__all__ = ["RuntimeRunner"]


def __getattr__(name: str):
    if name == "RuntimeRunner":
        from .runner import RuntimeRunner

        return RuntimeRunner
    raise AttributeError(name)
