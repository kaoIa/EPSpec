import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

TARGETS = {
    "shootout": "active pharmaceutical ingredient",
    "corn": "starch",
    "soil": "soil organic matter",
    "tecator": "fat",
}


class ScientificSimulator:
    def preprocess(self, method: str, input_path: Path, output_path: Path) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, output_path)
        return {"status": "completed", "tool_id": method, "output_path": str(output_path), "simulated": True}

    def model(self, method: str, dataset: str, input_path: Path, output_dir: Path) -> dict[str, Any]:
        if not input_path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        seed = int(hashlib.sha256(f"{dataset}:{method}".encode()).hexdigest()[:8], 16)
        adjustment = {
            "plsr": 0.0,
            "ipls_plsr": 0.018,
            "cars_plsr": 0.024,
            "EPSpec_plsr_sliding": 0.038,
            "EPSpec_plsr": 0.045,
        }[method]
        base_r2 = 0.78 + (seed % 70) / 1000 + adjustment
        rows = []
        for fold in range(1, 6):
            variation = (fold - 3) * 0.004
            r2 = min(0.98, base_r2 + variation)
            rmse = max(0.01, 0.48 - r2 * 0.35)
            rows.append(
                {
                    "fold": fold,
                    "pc": 6 + fold % 3,
                    "best_k": 3 if method.startswith("EPSpec") else "",
                    "n_selected_features": 72 if method in {"ipls_plsr", "cars_plsr"} else "",
                    "R2": round(r2, 6),
                    "RMSE": round(rmse, 6),
                    "MAE": round(rmse * 0.79, 6),
                    "Bias": round((fold - 3) * 0.0008, 6),
                    "RPD": round(1.8 + r2, 6),
                    "RPIQ": round(2.1 + r2, 6),
                }
            )
        fieldnames = list(rows[0])
        with (output_dir / "metrics_per_fold.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary = {}
        for metric in ("R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"):
            values = [float(row[metric]) for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            summary[metric] = {"mean": round(mean, 6), "std": round(variance**0.5, 6)}
        if method.startswith("EPSpec"):
            summary["best_k"] = {"per_fold": [3, 3, 3, 3, 3]}
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if method.startswith("EPSpec"):
            self._epspec_files(output_dir, dataset)
        elif method in {"ipls_plsr", "cars_plsr"}:
            self._coefficient_file(output_dir)
        return {"status": "completed", "tool_id": method, "output_path": str(output_dir), "simulated": True}

    def _epspec_files(self, output_dir: Path, dataset: str) -> None:
        for fold in range(1, 6):
            intervals = [
                {
                    "interval_id": f"B{index}",
                    "start_nm": 1000 + index * 120,
                    "end_nm": 1080 + index * 120,
                    "local_r2": round(0.91 - index * 0.025 - (fold - 3) * 0.001, 4),
                    "stability": round(0.94 - index * 0.02 - (fold - 3) * 0.001, 4),
                    "snr_band": 18 + index + fold / 10,
                    "artifact_risk": round(index * 0.03, 4),
                    "corr_max": round(0.34 + index * 0.04, 4),
                    "corr_mean": round(0.2 + index * 0.03, 4),
                }
                for index in range(1, 6)
            ]
            metrics = {
                "info": {
                    "dataset_key": dataset,
                    "predicted_substance": TARGETS[dataset],
                    "n_intervals": len(intervals),
                },
                "global_prior_knowledge": f"Offline evidence fixture for {TARGETS[dataset]}",
                "intervals": intervals,
            }
            ranking = {
                "interval_ranking": [
                    {
                        "id": item["interval_id"],
                        "start": item["start_nm"],
                        "end": item["end_nm"],
                        "rank": index,
                        "importance_level": "high" if index <= 3 else "medium",
                        "reason": "Evidence-consistent simulated ranking",
                    }
                    for index, item in enumerate(intervals, start=1)
                ]
            }
            suffix = f"{fold:02d}"
            (output_dir / f"{dataset}_interval_metrics_outerfold{suffix}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            (output_dir / f"{dataset}_interval_ranking_outerfold{suffix}.json").write_text(json.dumps(ranking, ensure_ascii=False, indent=2), encoding="utf-8")

    def _coefficient_file(self, output_dir: Path) -> None:
        with (output_dir / "coefficients.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["fold", "feature", "coefficient"])
            writer.writeheader()
            for fold in range(1, 6):
                for wavelength in range(1120, 1281, 20):
                    writer.writerow({"fold": fold, "feature": wavelength, "coefficient": round((wavelength - 1200) / 10000, 6)})
