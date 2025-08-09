# message_core.py
from __future__ import annotations
from dataclasses import dataclass, is_dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------
# Minimal message primitives
# ---------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool", "developer"]  # not enforced at runtime

Block = list[str, Any]                    # e.g., {"type": "text", "text": "..."}
Content = str | list[Block]         # string or multimodal-style blocks


@dataclass(frozen=True)
class Message:
    role: str
    content: Content
    name: str | None = None
    tool_call_id: str | None = None
    meta: dict[str, Any] | None = None


    def to_dict(self, include_meta: bool = False) -> dict[str, Any]:
        """Convert this Message into a dict suitable for API calls or logging."""
        d = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if include_meta and self.meta:
            d["meta"] = self.meta
        return d

def msg(role: str, content: Content, **kw) -> Message:
    """Primary factory. Lightweight and permissive by design."""
    return Message(role=role.lower(), content=content, **kw)

# sugar
def system(content: Content, **kw) -> Message:    return msg("system", content, **kw)
def user(content: Content, **kw) -> Message:      return msg("user", content, **kw)
def assistant(content: Content, **kw) -> Message: return msg("assistant", content, **kw)
def tool(content: Content, **kw) -> Message:      return msg("tool", content, **kw)
def developer(content: Content, **kw) -> Message: return msg("developer", content, **kw)


# ---------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------
def to_openai(
    messages: list[Message],
    *,
    use_instructions: bool = False,
) -> dict[str, Any]:
    """
    Render messages for OpenAI. Your program only uses system messages.
    If use_instructions=True, we 'lift' the FIRST system message into the
    'instructions' field, and send the rest as chat messages.

    - 'developer' role is mapped to 'system' for OpenAI compatibility.
    - Content passes through unchanged (str or list[blocks]).
    """
    instructions_content: Content | None = None
    rendered: list[dict[str, Any]] = []

    # If requested, lift the first system message into instructions
    if use_instructions:
        for i, m in enumerate(messages):
            if m.role == "system":
                instructions_content = m.content
                messages = messages[:i] + messages[i+1:]
                break

    for m in messages:
        rendered.append({
            "role": "system" if m.role == "developer" else m.role,
            "content": m.content,
            **({"name": m.name} if m.name else {}),
            **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
        })

    payload: dict[str, Any] = {"messages": rendered}
    if use_instructions and instructions_content is not None:
        payload["instructions"] = instructions_content
    return payload

# ---------------------------------------------------------------------
# Normalize input format
# ---------------------------------------------------------------------

MessagesIn = str | dict[str, Any] | list[Message | dict[str, Any]] # single message dict OR full payload with "messages"

def normalize(
    data: MessagesIn,
    *,
    strip_meta: bool = True,
) -> dict[str, Any]:
    """
    Normalize any accepted message input into {"messages": [...]} ready for LLM API.
    Accepted:
      1) str
      2) single dict {role, content}
      3) list of dicts
      4) single Message-like dataclass
      5) list of Message-like dataclasses
      6) full payload dict {"messages": [...], "instructions": ...}
    """
    # Case 6: full provider payload
    if isinstance(data, dict) and "messages" in data:
        if not isinstance(data["messages"], list):
            raise TypeError("'messages' must be a list")
        payload: dict[str, Any] = {
            "messages": [_coerce_message_dict(it, strip_meta=strip_meta) for it in data["messages"]]
        }
        if "instructions" in data:
            payload["instructions"] = data["instructions"]
        return payload

    # Case 1: bare string
    if isinstance(data, str):
        return {"messages": [{"role": "user", "content": data}]}

    # Case 4: single Message-like dataclass
    if _is_message_like_obj(data):
        return {"messages": [_message_obj_to_dict(data, strip_meta=strip_meta)]}

    # Case 2: single message dict
    if isinstance(data, dict):
        return {"messages": [_coerce_message_dict(data, strip_meta=strip_meta)]}

    # Case 3 & 5: list of dicts and/or Message-like dataclasses
    if isinstance(data, list):
        if not data:
            raise ValueError("Empty message list is not allowed")
        items: list[dict[str, Any]] = []
        for it in data:
            if _is_message_like_obj(it):
                items.append(_message_obj_to_dict(it, strip_meta=strip_meta))
            elif isinstance(it, dict):
                items.append(_coerce_message_dict(it, strip_meta=strip_meta))
            else:
                raise TypeError("List items must be dicts or dataclass with 'role' and 'content'")
        return {"messages": items}

    raise TypeError("Invalid input: must be str, dict, list, or Message-like dataclass")

# --------------------- internals ---------------------

def _is_message_like_obj(obj) -> bool:
    return is_dataclass(obj) and hasattr(obj, "role") and hasattr(obj, "content")

def _message_obj_to_dict(m: Any, *, strip_meta: bool) -> dict[str, Any]:
    # Works with your Message dataclass: role, content, name, tool_call_id, meta
    if not getattr(m, "role", None) or getattr(m, "content", None) is None:
        raise ValueError("Message object must have 'role' and 'content'")
    d = {
        "role": str(m.role).lower(),
        "content": m.content,
    }
    if getattr(m, "name", None):
        d["name"] = m.name
    if getattr(m, "tool_call_id", None):
        d["tool_call_id"] = m.tool_call_id
    if (not strip_meta) and getattr(m, "meta", None):
        d["meta"] = m.meta
    _validate_message_dict(d)  # light validation
    return d

def _coerce_message_dict(d: dict[str, Any], *, strip_meta: bool) -> dict[str, Any]:
    if not isinstance(d, dict):
        raise TypeError("Message item must be a dict")
    role = d.get("role", None)
    content = d.get("content", None)
    if not isinstance(role, str) or not role:
        raise ValueError("Each message dict must have non-empty string 'role'")
    if content is None:
        raise ValueError("Each message dict must have 'content'")

    out = {
        "role": role.lower(),
        "content": content,
    }
    # optional fields
    if "name" in d and d["name"] is not None:
        if not isinstance(d["name"], str):
            raise TypeError("'name' must be a string when provided")
        out["name"] = d["name"]
    if "tool_call_id" in d and d["tool_call_id"] is not None:
        if not isinstance(d["tool_call_id"], str):
            raise TypeError("'tool_call_id' must be a string when provided")
        out["tool_call_id"] = d["tool_call_id"]
    # optionally preserve meta
    if (not strip_meta) and ("meta" in d) and d["meta"] is not None:
        if not isinstance(d["meta"], dict):
            raise TypeError("'meta' must be a dict when provided")
        out["meta"] = d["meta"]

    _validate_message_dict(out)
    return out

def _validate_message_dict(d: dict[str, Any]) -> None:
    # role
    if not isinstance(d.get("role"), str) or not d["role"]:
        raise ValueError("Invalid 'role'")
    # content: allow str or list (blocks); reject other types
    content = d.get("content")
    if isinstance(content, str):
        if content == "":
            raise ValueError("Empty string content is not allowed")
    elif isinstance(content, list):
        # We don't enforce vendor block schemas—just ensure it's a list of dict-like parts
        for i, part in enumerate(content):
            if not isinstance(part, dict):
                raise TypeError(f"content blocks must be dicts; got {type(part).__name__} at index {i}")
    else:
        raise TypeError("content must be a string or a list of blocks (dicts)")
