"""Check meaningful time-series failure modes using independent synthetic data."""
import unittest

import numpy as np
import pandas as pd

from prepare import build_samples, split_masks


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-08-01", periods=100, freq="min")
        self.raw = pd.DataFrame({"日時": self.index, "稼働中": 1, "temperature": np.arange(100.)})
        self.cfg = {"time_column": "日時", "target": "temperature", "features": ["temperature"],
                    "lags_minutes": [0, 1, 10], "horizon_minutes": 5,
                    "train_end": str(self.index[40]), "validation_end": str(self.index[70])}

    def test_horizon_lags_and_purged_boundaries(self):
        data, _ = build_samples(self.raw, self.cfg)
        t = self.index[20]
        np.testing.assert_array_equal(data["X"].loc[t], [20, 19, 10])
        self.assertEqual(data["y"].loc[t], 25)
        masks = split_masks(data["X"].index, self.cfg)
        index = data["X"].index
        self.assertLess(index[masks["train"]].max() + pd.Timedelta(minutes=5), self.index[40])
        self.assertGreaterEqual(index[masks["validation"]].min(), self.index[40])
        self.assertLess(index[masks["validation"]].max() + pd.Timedelta(minutes=5), self.index[70])
        self.assertGreaterEqual(index[masks["test"]].min(), self.index[70])

    def test_conflicting_timestamp_is_excluded_without_choosing_a_value(self):
        conflict = self.raw.iloc[[30]].copy()
        conflict["temperature"] = 9999
        data, profile = build_samples(pd.concat([self.raw, conflict]), self.cfg)
        self.assertEqual(profile["conflicting_rows_excluded"], 2)
        # Entire history / forecast windows crossing that minute are excluded.
        self.assertFalse(data["X"].index.isin(self.index[25:41]).any())
        self.assertIn(self.index[24], data["X"].index)
        self.assertIn(self.index[41], data["X"].index)

    def test_future_edits_do_not_change_past_features(self):
        original, _ = build_samples(self.raw, self.cfg)
        modified = self.raw.copy()
        modified.loc[modified["日時"] > self.index[40], "temperature"] += 10000
        changed, _ = build_samples(modified, self.cfg)
        pd.testing.assert_frame_equal(original["X"].loc[:self.index[40]], changed["X"].loc[:self.index[40]])

    def test_missing_observation_is_not_interpolated(self):
        data, _ = build_samples(self.raw.drop(index=30), self.cfg)
        self.assertFalse(data["X"].index.isin(self.index[25:41]).any())


if __name__ == "__main__":
    unittest.main()
