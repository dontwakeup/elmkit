try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version('elmkit')
except PackageNotFoundError:
    __version__ = 'dev'