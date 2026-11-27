import unittest

from rich_eval import paired_bootstrap


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_runs_have_zero_difference(self):
        docs = [
            {"id": "a", "strict_tp": 2, "strict_fp": 0, "strict_fn": 0},
            {"id": "b", "strict_tp": 1, "strict_fp": 1, "strict_fn": 0},
        ]
        result = paired_bootstrap(docs, docs, draws=50, seed=1)
        self.assertEqual(result["observed_diff"], 0.0)
        self.assertEqual(result["low"], 0.0)
        self.assertEqual(result["high"], 0.0)

    def test_pairs_documents_by_id(self):
        left = [
            {"id": "a", "strict_tp": 1, "strict_fp": 0, "strict_fn": 0},
            {"id": "b", "strict_tp": 0, "strict_fp": 0, "strict_fn": 1},
        ]
        right = [
            {"id": "b", "strict_tp": 1, "strict_fp": 0, "strict_fn": 0},
            {"id": "a", "strict_tp": 1, "strict_fp": 0, "strict_fn": 0},
        ]
        result = paired_bootstrap(left, right, draws=20, seed=1)
        self.assertEqual(result["documents"], 2)
        self.assertLess(result["observed_diff"], 0)


if __name__ == "__main__":
    unittest.main()
