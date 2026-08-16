import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EVALUATION = ROOT / "evaluation"


class PublicReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        sys.path.insert(0, str(SRC))
        sys.path.insert(0, str(EVALUATION))

    def test_redacts_signed_urls_and_credentials(self):
        from public_safety import redact_urls

        value = {
            "url": "https://cdn.example/video.mp4?token=secret",
            "Authorization": "Bearer secret",
        }
        self.assertEqual(
            redact_urls(value),
            {
                "url": "https://cdn.example/video.mp4<redacted-query>",
                "Authorization": "<redacted>",
            },
        )

    def test_playbooks_are_valid_json(self):
        for path in (ROOT / "playbooks").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload), {"strategies", "templates", "pitfalls"})
            self.assertTrue(all(isinstance(payload[key], list) for key in payload))

    def test_final_playbook_matches_competition_release(self):
        path = ROOT / "playbooks" / "playbook_final.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["strategies"]), 68)
        self.assertEqual(len(payload["templates"]), 6)
        self.assertEqual(len(payload["pitfalls"]), 5)
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "2537aff312a098b1de2bc94d44efae94b68020bbe08268d886026e2c8955e415",
        )

    def test_evaluation_score_and_aggregation(self):
        from aggregate_results import aggregate
        from score_candidates import score

        metrics = {
            "GME-Score": 0.60,
            "story_video_consistency": 0.90,
            "cur_score": 0.70,
            "arc_score": 0.65,
            "motion_smoothness": 0.95,
            "imaging_quality": 0.75,
        }
        self.assertAlmostEqual(score(metrics), 0.75)

        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir) / "id001_prompt0"
            result_dir.mkdir()
            (result_dir / "GME.json").write_text(
                json.dumps({"GME-Score": metrics["GME-Score"]}), encoding="utf-8"
            )
            (result_dir / "face_similarity.json").write_text(
                json.dumps({"cur_score": metrics["cur_score"], "arc_score": metrics["arc_score"]}),
                encoding="utf-8",
            )
            (result_dir / "story_video_consistency.json").write_text(
                json.dumps({"story_video_consistency": metrics["story_video_consistency"]}),
                encoding="utf-8",
            )
            (result_dir / "id001_prompt0_Vbench_eval_results.json").write_text(
                json.dumps(
                    {
                        "motion_smoothness": [metrics["motion_smoothness"]],
                        "imaging_quality": [metrics["imaging_quality"]],
                    }
                ),
                encoding="utf-8",
            )
            report = aggregate(Path(temp_dir))

        self.assertEqual(report["video_count"], 1)
        self.assertEqual(report["results"][0]["video_name"], "id001_prompt0")
        self.assertAlmostEqual(report["indicator_means"]["arc_score"], 0.65)


if __name__ == "__main__":
    unittest.main()
