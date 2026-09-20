def ingest_and_prepare(*args, **kwargs):
    from .pipeline import ingest_and_prepare as _ingest_and_prepare

    return _ingest_and_prepare(*args, **kwargs)


__all__ = ["ingest_and_prepare"]
