import shutil
import sys
from pathlib import Path

import pytest

from epspec_agents.config import ModelProfile, RuntimeConfig


@pytest.fixture
def runtime_config(tmp_path: Path) -> RuntimeConfig:
    project = tmp_path / "project"
    agents = project / "Agents"
    prompts = agents / "epspec_agents" / "prompts"
    prompts.mkdir(parents=True)
    source_prompts = Path(__file__).resolve().parents[1] / "epspec_agents" / "prompts"
    for name in ("planning.txt", "interpretation.txt"):
        shutil.copyfile(source_prompts / name, prompts / name)
    raw = project / "Data" / "Raw Data"
    raw.mkdir(parents=True)
    for dataset in ("shootout", "corn", "soil", "tecator"):
        (raw / f"{dataset}.csv").write_text("wavelength,target\n1000,1\n1100,2\n", encoding="utf-8")
    preprocessing = project / "Baseline Algorithm" / "Preprocessing"
    regression = project / "Baseline Algorithm" / "Regression"
    wavelength = project / "Wavelength selection" / "Regression"
    sliding = project / "Experiments" / "Ablation" / "Code" / "滑动窗口和分段数"
    for directory in (preprocessing, regression, wavelength, sliding):
        directory.mkdir(parents=True)
    preprocess_source = (
        "from pathlib import Path\n"
        "import shutil\n"
        "def preprocess_file(input_path, output_path):\n"
        "    Path(output_path).parent.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copyfile(input_path, output_path)\n"
    )
    for name in ("savitzky_golay", "snv"):
        (preprocessing / f"{name}.py").write_text(preprocess_source, encoding="utf-8")
    model_source = (
        "from pathlib import Path\n"
        "import json\n"
        "def run_regression(input_path, out_dir, **kwargs):\n"
        "    output = Path(out_dir)\n"
        "    output.mkdir(parents=True, exist_ok=True)\n"
        "    payload = {'R2': {'mean': 0.9, 'std': 0.01}}\n"
        "    (output / 'summary.json').write_text(json.dumps(payload), encoding='utf-8')\n"
    )
    for name in ("plsr", "ipls_plsr", "ipls_plsr_no_full_lv_cap", "cars_plsr_no_full_lv_cap"):
        (regression / f"{name}.py").write_text(model_source, encoding="utf-8")
    (wavelength / "EPSpec_plsr_joink.py").write_text(model_source, encoding="utf-8")
    (sliding / "Sliding Window Segmentation Version.py").write_text(model_source, encoding="utf-8")
    prior = project / "RAG_Prior knowledge" / "Data" / "Functional Group.xlsx"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"fixture")
    profile = ModelProfile("test", "test-model", "test-key", "https://example.invalid/v1", 0.0, 2.0, 0)
    return RuntimeConfig(
        agents_dir=agents,
        project_root=project,
        planner=profile,
        interpreter=profile,
        scientific=profile,
        offline=True,
        execution_mode="simulate",
        worker_timeout_seconds=20.0,
        max_concurrency=3,
        allow_text_fallback=False,
        capture_prompts=False,
        sdk_tracing=False,
        auto_approve=True,
        server_token="",
        python_executable=sys.executable,
    )
