__all__ = ["build_workflow"]


def __getattr__(name: str):
    if name == "build_workflow":
        from .workflow import build_workflow

        return build_workflow
    raise AttributeError(name)
