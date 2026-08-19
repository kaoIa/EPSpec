import ast
import tokenize
from pathlib import Path

from epspec_agents.evaluation import run_evaluation
from epspec_agents.services.tracing import redact


def test_planning_evaluation_suite() -> None:
    agents = Path(__file__).resolve().parents[1]
    result = run_evaluation(agents / "evals" / "cases.json")
    assert result["score"] == 1.0


def test_python_sources_have_no_comment_tokens() -> None:
    agents = Path(__file__).resolve().parents[1]
    violations = []
    for path in sorted(agents.rglob("*.py")):
        if {".venv", ".runtime", "runs", "build", "dist"}.intersection(path.parts):
            continue
        with path.open("rb") as stream:
            for token in tokenize.tokenize(stream.readline):
                if token.type == tokenize.COMMENT:
                    violations.append(f"{path}:{token.start[0]}")
    assert violations == []


def test_python_sources_have_no_docstrings() -> None:
    agents = Path(__file__).resolve().parents[1]
    violations = []
    node_types = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for path in sorted(agents.rglob("*.py")):
        if {".venv", ".runtime", "runs", "build", "dist"}.intersection(path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, node_types) and ast.get_docstring(node, clean=False) is not None:
                violations.append(f"{path}:{getattr(node, 'lineno', 1)}")
    assert violations == []


def test_configuration_sources_have_no_hash_comments() -> None:
    agents = Path(__file__).resolve().parents[1]
    names = {"Dockerfile", "Makefile", ".env.example", ".gitignore", ".gitattributes", "Dockerfile.dockerignore"}
    suffixes = {".toml", ".yml", ".yaml"}
    excluded = {".venv", ".runtime", "runs", "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    violations = []
    for path in sorted(item for item in agents.rglob("*") if item.is_file() and (item.name in names or item.suffix in suffixes)):
        if excluded.intersection(path.parts) or any(part.endswith(".egg-info") for part in path.parts):
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "#" in line:
                violations.append(f"{path}:{line_number}")
    assert violations == []


def test_nested_secret_redaction() -> None:
    value = redact({"planner_api_key": "secret", "nested": {"refresh-token": "secret", "value": "safe"}})
    assert value == {"planner_api_key": "[REDACTED]", "nested": {"refresh-token": "[REDACTED]", "value": "safe"}}
