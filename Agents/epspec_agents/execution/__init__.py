__all__ = ["SubprocessToolRunner"]


def __getattr__(name: str):
    if name == "SubprocessToolRunner":
        from .subprocess_runner import SubprocessToolRunner

        return SubprocessToolRunner
    raise AttributeError(name)
