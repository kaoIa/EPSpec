import hashlib
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..schemas import ArtifactRef, ExperimentPlan, ExperimentResult, ModelFamily, ModelId, ModelRunResult, PreprocessConfig

METRICS = ["R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"]
TARGETS = {"shootout": "active pharmaceutical ingredient", "corn": "starch", "soil": "soil organic matter", "tecator": "fat"}


class ResultParser:
    def parse(self, plan: ExperimentPlan, simulated: bool = False) -> ExperimentResult:
        main = self._model_result("main_model", plan.step_model_main.method, plan.step_model_main.family, plan.step_model_main.out_dir)
        comparisons = [self._model_result(f"compare_model_{index}", step.method, step.family, step.out_dir) for index, step in enumerate(plan.step_model_compare.models, start=1)]
        prior = self._task_prior(plan.dataset_name, [main, *comparisons])
        return ExperimentResult(
            run_id=plan.run_id,
            dataset_name=plan.dataset_name,
            preprocess=PreprocessConfig(enabled=plan.step_preprocess.enabled, method=plan.step_preprocess.method),
            main_result=main,
            comparison_results=comparisons,
            task_level_prior=prior,
            simulated=simulated,
        )

    def _model_result(self, role: str, method: ModelId, family: ModelFamily, result_dir: Path) -> ModelRunResult:
        summary = self._read_json(self._find(result_dir, "summary.json")) or {}
        metrics_df = self._read_csv(self._find(result_dir, "metrics_per_fold.csv"))
        metrics_summary = self._metrics_summary(summary, metrics_df)
        records = self._records(metrics_df)
        context = self._epspec_context(result_dir) if method.startswith("EPSpec") else None
        if method.startswith("EPSpec"):
            selection = self._epspec_selection(result_dir, summary, metrics_df)
        elif method == "plsr":
            selection = {"selection_type": "full_spectrum", "note": "该模型为全谱 PLSR，不进行显式选段。"}
        else:
            selection = self._coefficient_selection(result_dir, method, summary)
        return ModelRunResult(
            role=role,
            method=method,
            family=family,
            result_dir=result_dir,
            metrics_summary=metrics_summary,
            metrics_per_fold=records,
            selection_details=selection,
            task_context=context,
            artifacts=self._artifacts(result_dir),
        )

    def _artifacts(self, root: Path) -> list[ArtifactRef]:
        if not root.is_dir():
            return []
        output = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            output.append(
                ArtifactRef(
                    name=str(path.relative_to(root)).replace("\\", "/"),
                    path=path,
                    media_type=media_type,
                    role="model-output",
                    sha256=digest.hexdigest(),
                    size_bytes=path.stat().st_size,
                )
            )
        return output

    def _find(self, root: Path, name: str) -> Path | None:
        if not root.is_dir():
            return None
        direct = root / name
        if direct.is_file():
            return direct
        return next((path for path in root.rglob(name) if path.is_file()), None)

    def _regex_files(self, root: Path, pattern: str) -> list[Path]:
        if not root.is_dir():
            return []
        regex = re.compile(pattern, flags=re.I)
        return sorted(path for path in root.rglob("*") if path.is_file() and regex.search(path.name))

    def _read_json(self, path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def _read_csv(self, path: Path | None) -> pd.DataFrame | None:
        if path is None:
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def _metrics_summary(self, summary: dict[str, Any], frame: pd.DataFrame | None) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for metric in METRICS:
            value = summary.get(metric)
            if isinstance(value, dict) and value.get("mean") is not None and value.get("std") is not None:
                output[metric] = {"mean": float(value["mean"]), "std": float(value["std"])}
            elif frame is not None and metric in frame.columns:
                series = pd.to_numeric(frame[metric], errors="coerce").dropna()
                if not series.empty:
                    output[metric] = {"mean": float(series.mean()), "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0}
        return output

    def _records(self, frame: pd.DataFrame | None) -> list[dict[str, Any]]:
        if frame is None or frame.empty:
            return []
        preferred = ["fold", "outer_fold", "pc", "best_k", "n_features", "n_selected_features", *METRICS]
        columns = [column for column in preferred if column in frame.columns] or list(frame.columns)
        records: list[dict[str, Any]] = []
        for row in frame[columns].to_dict(orient="records"):
            clean: dict[str, Any] = {}
            for key, value in row.items():
                if pd.isna(value):
                    clean[key] = None
                elif hasattr(value, "item"):
                    clean[key] = value.item()
                else:
                    clean[key] = value
            records.append(clean)
        return records

    def _fold_values(self, summary: dict[str, Any], frame: pd.DataFrame | None, key: str) -> list[int] | None:
        value = summary.get(key)
        if isinstance(value, dict):
            value = value.get("per_fold")
        if isinstance(value, list):
            return [int(item) for item in value]
        if frame is not None and key in frame.columns:
            series = pd.to_numeric(frame[key], errors="coerce").dropna()
            return [int(item) for item in series.tolist()] if not series.empty else None
        return None

    def _fold_id(self, path: Path) -> int | None:
        match = re.search(r"(?:outer)?fold0*([0-9]+)", path.name, flags=re.I)
        return int(match.group(1)) if match else None

    def _epspec_context(self, root: Path) -> dict[str, Any]:
        for path in self._regex_files(root, r"interval_metrics_outerfold.*\.json$"):
            value = self._read_json(path)
            if value:
                raw_info = value.get("info")
                info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}
                prior = str(value.get("global_prior_knowledge") or "").strip()
                return {
                    "dataset_key": info.get("dataset_key"),
                    "predicted_substance": info.get("predicted_substance"),
                    "n_intervals": info.get("n_intervals"),
                    "global_prior_knowledge": prior[:2200] + ("\n...[已截断]" if len(prior) > 2200 else "") if prior else None,
                }
        return {"dataset_key": None, "predicted_substance": None, "n_intervals": None, "global_prior_knowledge": None}

    def _epspec_selection(self, root: Path, summary: dict[str, Any], frame: pd.DataFrame | None) -> dict[str, Any]:
        metric_files = self._regex_files(root, r"interval_metrics_outerfold.*\.json$")
        ranking_files = self._regex_files(root, r"interval_ranking_outerfold(?!.*_nms).*\.json$")
        metric_objects = {fold: self._read_json(path) or {} for path in metric_files if (fold := self._fold_id(path)) is not None}
        ranking_objects = {fold: self._read_json(path) or {} for path in ranking_files if (fold := self._fold_id(path)) is not None}
        best_values = self._fold_values(summary, frame, "best_k")
        best_map = {index + 1: value for index, value in enumerate(best_values or [])}
        per_fold: list[dict[str, Any]] = []
        consensus: dict[str, dict[str, Any]] = {}
        for fold in sorted(set(metric_objects) | set(ranking_objects)):
            intervals = metric_objects.get(fold, {}).get("intervals", [])
            metric_index = {str(item.get("interval_id") or item.get("id")): item for item in intervals}
            ranking = ranking_objects.get(fold, {}).get("interval_ranking", [])
            ranking = sorted(ranking, key=lambda item: float(item.get("rank", 1e9)))
            if not ranking:
                ranking = [
                    {
                        "id": item.get("interval_id"),
                        "start": item.get("start_nm"),
                        "end": item.get("end_nm"),
                        "rank": index + 1,
                        "importance_level": "fallback",
                        "reason": "按 local_r2 恢复排序",
                    }
                    for index, item in enumerate(sorted(intervals, key=lambda item: float(item.get("local_r2", 0)), reverse=True))
                ]
            best_k = best_map.get(fold, min(5, len(ranking)))
            selected: list[dict[str, Any]] = []
            for item in ranking[:best_k]:
                metric = metric_index.get(str(item.get("id")), {})
                record = {
                    "interval_id": item.get("id"),
                    "start_nm": item.get("start"),
                    "end_nm": item.get("end"),
                    "rank": item.get("rank"),
                    "importance_level": item.get("importance_level"),
                    "reason": item.get("reason"),
                    **{key: metric.get(key) for key in ["local_r2", "stability", "snr_band", "artifact_risk", "corr_max", "corr_mean"] if key in metric},
                }
                selected.append(record)
                try:
                    key = f"{float(item['start']):.6f}-{float(item['end']):.6f}"
                    aggregate = consensus.setdefault(key, {"start_nm": float(item["start"]), "end_nm": float(item["end"]), "folds": [], "ranks": []})
                    aggregate["folds"].append(fold)
                    aggregate["ranks"].append(float(item.get("rank")))
                except Exception:
                    continue
            per_fold.append({"fold": fold, "best_k": best_k, "selected_topk_intervals": selected})
        consensus_rows: list[dict[str, Any]] = [
            {
                "start_nm": item["start_nm"],
                "end_nm": item["end_nm"],
                "selected_frequency": len(item["folds"]),
                "avg_rank": round(sum(item["ranks"]) / len(item["ranks"]), 4) if item["ranks"] else None,
            }
            for item in consensus.values()
        ]
        consensus_rows.sort(key=lambda item: (-int(item["selected_frequency"]), float(item["avg_rank"] or 1e9), float(item["start_nm"])))
        return {
            "selection_type": "epspec_topk_intervals",
            "best_k_per_fold": best_values,
            "per_fold_selected_topk_intervals": per_fold,
            "consensus_selected_intervals": consensus_rows[:12],
        }

    def _feature(self, value: Any) -> float | None:
        text = str(value).strip()
        if not text or text.lower() == "intercept":
            return None
        try:
            return float(text)
        except ValueError:
            match = re.search(r"(\d+(?:\.\d+)?)", text)
            return float(match.group(1)) if match else None

    def _ranges(self, values: list[float]) -> list[dict[str, Any]]:
        values = sorted(set(values))
        if not values:
            return []
        positive_diffs = [values[index + 1] - values[index] for index in range(len(values) - 1) if values[index + 1] > values[index]]
        step = min(positive_diffs) if positive_diffs else 1.0
        groups = [[values[0]]]
        for value in values[1:]:
            if value - groups[-1][-1] <= step * 1.5 + 1e-9:
                groups[-1].append(value)
            else:
                groups.append([value])
        return [{"start_nm": group[0], "end_nm": group[-1], "n_points": len(group)} for group in groups]

    def _coefficient_selection(self, root: Path, method: ModelId, summary: dict[str, Any]) -> dict[str, Any]:
        frame = self._read_csv(self._find(root, "coefficients.csv"))
        if frame is None or frame.empty or not {"fold", "feature"}.issubset(frame.columns):
            return {"selection_type": "unknown", "note": "当前结果中未读取到可恢复逐折选段的 coefficients.csv。"}
        frame = frame.copy()
        frame["parsed"] = frame["feature"].map(self._feature)
        frame = frame[frame["parsed"].notna()]
        per_fold: list[dict[str, Any]] = []
        for fold, subset in frame.groupby("fold"):
            values = sorted(set(float(value) for value in subset["parsed"].tolist()))
            per_fold.append({"fold": int(fold), "n_selected_features": len(values), "selected_ranges": self._ranges(values)})
        output: dict[str, Any] = {
            "selection_type": "selected_ranges_from_coefficients",
            "per_fold_selected_ranges": sorted(per_fold, key=lambda item: item["fold"]),
            "consensus_selected_ranges": self._ranges([float(value) for value in frame["parsed"].tolist()]),
        }
        if method == "ipls_plsr":
            output["ipls_interval_width"] = summary.get("ipls_interval_width")
        if method == "cars_plsr":
            output["cars_n_mc"] = summary.get("cars_n_mc")
            output["cars_ratio_samples"] = summary.get("cars_ratio_samples")
        return output

    def _task_prior(self, dataset: str, results: list[ModelRunResult]) -> dict[str, Any]:
        for result in results:
            context = result.task_context or {}
            if context.get("global_prior_knowledge"):
                return {
                    "available": True,
                    "source_model": result.method,
                    "dataset_key": context.get("dataset_key") or dataset,
                    "predicted_substance": context.get("predicted_substance") or TARGETS.get(dataset),
                    "n_intervals": context.get("n_intervals"),
                    "global_prior_knowledge": context.get("global_prior_knowledge"),
                }
        return {
            "available": False,
            "source_model": None,
            "dataset_key": dataset,
            "predicted_substance": TARGETS.get(dataset),
            "n_intervals": None,
            "global_prior_knowledge": None,
        }
