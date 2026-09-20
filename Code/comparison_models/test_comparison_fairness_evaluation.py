import csv
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_recommendations import main as evaluate_main  # noqa: E402
from summarize_dataset_results import (  # noqa: E402
    FAIRNESS_METRICS,
    TOP_KS,
    aggregate_results,
    collect_dataset_results,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


class ComparisonFairnessEvaluationTests(unittest.TestCase):
    def build_tiny_graph(self, root):
        dataset_root = root / "Tiny"
        graph_dir = dataset_root / "er_graph"
        raw_dir = dataset_root / "raw"
        graph_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        (graph_dir / "Q.txt").write_text("1,0\n0,1\n1,1\n", encoding="utf-8")
        write_json(graph_dir / "stu2know_mastery.json", [[0.8, 0.2], [0.3, 0.7]])
        (graph_dir / "test_triples.txt").write_text(
            "\n".join(
                [
                    "kc0\tmlkc0.8\tuid0",
                    "kc1\tmlkc0.2\tuid0",
                    "kc0\tmlkc0.3\tuid1",
                    "kc1\tmlkc0.7\tuid1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (graph_dir / "Tiny_uid_kc_response.txt").write_text(
            "uid0\t0,1\nuid1\t1\n",
            encoding="utf-8",
        )
        with (raw_dir / "interactions_all.csv").open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["uid", "entity_id", "entity_exercise_id", "question", "response", "split"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"uid": 0, "entity_id": "uid0", "entity_exercise_id": "ex0", "question": 0, "response": 1, "split": "train"},
                    {"uid": 0, "entity_id": "uid0", "entity_exercise_id": "ex2", "question": 2, "response": 1, "split": "train"},
                    {"uid": 1, "entity_id": "uid1", "entity_exercise_id": "ex1", "question": 1, "response": 1, "split": "train"},
                    {"uid": 1, "entity_id": "uid1", "entity_exercise_id": "ex2", "question": 2, "response": 0, "split": "test"},
                ]
            )
        return graph_dir

    def test_evaluation_writes_fairness_metrics_from_train_interactions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_dir = self.build_tiny_graph(root)
            scores_path = root / "scores.pkl"
            with scores_path.open("wb") as fp:
                pickle.dump([("uid0", [0.9, 0.1, 0.8]), ("uid1", [0.2, 0.7, 0.6])], fp)

            output_dir = root / "eval"
            evaluate_main(
                [
                    "--data-dir",
                    str(graph_dir),
                    "--scores-file",
                    str(scores_path),
                    "--output-dir",
                    str(output_dir),
                    "--dataset-name",
                    "Tiny",
                    "--model-name",
                    "TransE",
                    "--top-ks",
                    "1,2",
                    "--ep-top-k",
                    "1",
                ]
            )

            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("Fairness", metrics)
            self.assertEqual(
                metrics["FairnessDefinition"]["item_popularity_source"]["source"],
                "train_interactions",
            )
            self.assertIn("ItemExposureGini", metrics["Fairness"]["1"])
            self.assertTrue((output_dir / "fairness_metrics.csv").exists())

    def test_summary_carries_fairness_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            eval_dir = run_dir / "TransE" / "seed2024" / "eval"
            eval_dir.mkdir(parents=True)
            fairness = {
                str(top_k): {metric: 0.1 + index for index, metric in enumerate(FAIRNESS_METRICS)}
                for top_k in TOP_KS
            }
            payload = {
                "dataset": "Tiny",
                "model": "TransE",
                "seed": 2024,
                "Ada": {str(top_k): {"mean": 0.5, "std": 0.0} for top_k in TOP_KS},
                "NOV": {str(top_k): {"mean": 0.6, "std": 0.0} for top_k in TOP_KS},
                "Fairness": fairness,
                "Ep_sim": {"top_k": 10, "mean": 0.7, "std": 0.0},
            }
            write_json(eval_dir / "metrics.json", payload)

            rows, _ = collect_dataset_results("Tiny", run_dir)
            summary = aggregate_results(rows)

            self.assertEqual(rows[0]["ItemExposureGini@5"], 0.1)
            self.assertEqual(summary[0]["ItemExposureGini@5_mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
