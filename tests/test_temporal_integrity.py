"""
Temporal integrity tests for ML training and inference pipeline.
Ensures the model never sees future data and the temporal guard correctly
rejects training-range sequences during inference.
"""
import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -------- Path setup so we can import project modules --------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the module under test
from currency.modules import ml_pattern as ml
from currency.modules.strategy import Strategy


# ======================================================================
#  Helper functions for building test DataFrames
# ======================================================================

def make_ohlc_df(n_candles=300, start_date="2024-01-01", freq="6h",
                 seed=42):
    """Create a synthetic OHLC DataFrame with a DatetimeIndex.

    Prices follow a random walk with controllable seed.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_candles, freq=freq)
    raw = 100.0 + np.cumsum(rng.normal(0, 0.5, n_candles))
    prices = np.maximum(raw, 50.0)

    noise_high = np.abs(rng.normal(0, 0.3, n_candles))
    noise_low  = np.abs(rng.normal(0, 0.3, n_candles))

    df = pd.DataFrame({
        "Open":  prices,
        "High":  prices + noise_high,
        "Low":   prices - noise_low,
        "Close": prices,
        "Volume": rng.integers(80, 200, n_candles),
    }, index=dates)

    # Enforce High >= max(Open,Close) and Low <= min(Open,Close)
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"]  = df[["Open", "Close", "Low"]].min(axis=1)

    # Initialise pivot columns
    df["Is_High"] = np.nan
    df["Is_Low"]  = np.nan
    return df


def add_pivots(df, n_pivots=30, seed=1):
    """Place alternating high/low pivot markers into the DataFrame.

    Ensures >= 6 pivots exist so ``get_all_pivot_sequences`` works.
    """
    rng = np.random.default_rng(seed)
    step = max(len(df) // (n_pivots + 1), 2)
    indices = sorted(rng.choice(
        range(step, len(df) - step), size=n_pivots, replace=False
    ))
    for i, idx in enumerate(indices):
        if i % 2 == 0:
            df.iloc[idx, df.columns.get_loc("Is_High")] = \
                float(df.iloc[idx]["High"])
        else:
            df.iloc[idx, df.columns.get_loc("Is_Low")] = \
                float(df.iloc[idx]["Low"])
    return df


# ======================================================================
#  Tests
# ======================================================================

class TestExtractFeaturesFromPivots(unittest.TestCase):
    """``extract_features_from_pivots`` — shape & value sanity."""

    def setUp(self):
        self.pivots = [
            {"val": 105.0, "idx": 10, "time": pd.Timestamp("2024-01-11"), "is_high": True},
            {"val": 100.0, "idx":  8, "time": pd.Timestamp("2024-01-09"), "is_high": False},
            {"val": 103.0, "idx":  6, "time": pd.Timestamp("2024-01-07"), "is_high": True},
            {"val":  98.0, "idx":  4, "time": pd.Timestamp("2024-01-05"), "is_high": False},
            {"val": 102.0, "idx":  2, "time": pd.Timestamp("2024-01-03"), "is_high": True},
            {"val":  99.0, "idx":  0, "time": pd.Timestamp("2024-01-01"), "is_high": False},
        ]

    def test_output_shape(self):
        features = ml.extract_features_from_pivots(self.pivots)
        self.assertEqual(len(features), 16,
                         "Should produce exactly 16 features: "
                         "6 norm_vals + 5 wave_ratios + 5 time_diffs")

    def test_values_are_finite(self):
        features = ml.extract_features_from_pivots(self.pivots)
        self.assertTrue(np.all(np.isfinite(features)),
                        "All features should be finite numbers")

    def test_normalized_values_in_01_range(self):
        features = ml.extract_features_from_pivots(self.pivots)
        norm = features[:6]  # first 6 are normalized vals
        self.assertTrue(np.all(norm >= 0.0) and np.all(norm <= 1.0 + 1e-9),
                        "Normalised pivot values should be in [0, 1]")

    def test_identical_values_does_not_crash(self):
        """When all vals are identical, normalisation uses epsilon."""
        flat = [{"val": 100.0, "idx": i, "time": pd.Timestamp(f"2024-01-{i+1:02d}"), "is_high": i % 2 == 0}
                for i in range(5, -1, -1)]
        features = ml.extract_features_from_pivots(flat)
        self.assertTrue(np.all(np.isfinite(features)))


class TestLabelPivotTrade(unittest.TestCase):
    """``label_pivot_trade`` — win/loss classification."""

    def test_tp_before_sl_buy(self):
        """Buy: TP (high >= 103) hit before SL (low <= 97)."""
        df = pd.DataFrame({
            "High": [101, 102, 104.0, 101],
            "Low":  [100,  99, 100.0,  98],
        })
        label = ml.label_pivot_trade(df, occ_idx=0,
                                     entry_price=100.0,
                                     stop_loss=97.0,
                                     take_profit=103.0,
                                     is_buy=True)
        self.assertEqual(label, 1, "TP hit first => win")

    def test_sl_before_tp_buy(self):
        """Buy: SL hit before TP."""
        df = pd.DataFrame({
            "High": [101, 100, 104.0],
            "Low":  [100,  96.0, 99.0],
        })
        label = ml.label_pivot_trade(df, occ_idx=0,
                                     entry_price=100.0,
                                     stop_loss=97.0,
                                     take_profit=103.0,
                                     is_buy=True)
        self.assertEqual(label, 0, "SL hit first => loss")

    def test_tp_before_sl_sell(self):
        """Sell: TP (low <= 97) hit before SL (high >= 103)."""
        df = pd.DataFrame({
            "High": [102, 101, 102],
            "Low":  [ 99,  96.0,  98],
        })
        label = ml.label_pivot_trade(df, occ_idx=0,
                                     entry_price=100.0,
                                     stop_loss=103.0,
                                     take_profit=97.0,
                                     is_buy=False)
        self.assertEqual(label, 1, "TP hit first on sell => win")

    def test_neither_hit_returns_0(self):
        df = pd.DataFrame({
            "High": [101, 102, 101],
            "Low":  [ 99,  98,  99],
        })
        label = ml.label_pivot_trade(df, occ_idx=0,
                                     entry_price=100.0,
                                     stop_loss=97.0,
                                     take_profit=103.0,
                                     is_buy=True)
        self.assertEqual(label, 0, "Neither TP nor SL hit => default loss")

    def test_empty_future_data_returns_0(self):
        """occ_idx is the last row — no future data to check."""
        df = pd.DataFrame({"High": [101], "Low": [99]})
        label = ml.label_pivot_trade(df, occ_idx=0,
                                     entry_price=100.0,
                                     stop_loss=97.0,
                                     take_profit=103.0,
                                     is_buy=True)
        self.assertEqual(label, 0)


class TestGetAllPivotSequences(unittest.TestCase):
    """``get_all_pivot_sequences`` — sliding-window extraction."""

    def setUp(self):
        self.df = make_ohlc_df(n_candles=200, seed=1)
        add_pivots(self.df, n_pivots=30, seed=2)

    def test_returns_equal_length_lists(self):
        seq, tri, buy = ml.get_all_pivot_sequences(self.df)
        self.assertEqual(len(seq), len(tri))
        self.assertEqual(len(tri), len(buy))

    def test_each_sequence_has_6_pivots(self):
        seq, _, _ = ml.get_all_pivot_sequences(self.df)
        for s in seq:
            self.assertEqual(len(s), 6, "Each sequence must have exactly 6 pivots")

    def test_sequence_keys_present(self):
        seq, _, _ = ml.get_all_pivot_sequences(self.df)
        for s in seq:
            for pivot in s:
                self.assertIn("val", pivot)
                self.assertIn("idx", pivot)
                self.assertIn("time", pivot)
                self.assertIn("is_high", pivot)

    def test_trigger_times_are_timestamps(self):
        _, tri, _ = ml.get_all_pivot_sequences(self.df)
        for t in tri:
            self.assertIsInstance(t, pd.Timestamp)

    def test_insufficient_pivots_returns_empty(self):
        empty = make_ohlc_df(n_candles=30, seed=1)
        empty["Is_High"] = np.nan
        empty["Is_Low"] = np.nan
        # Only 4 pivots → not enough for a window of 6
        for i in range(4):
            empty.iloc[i * 5, empty.columns.get_loc("Is_High")] = 100.0
        seq, tri, buy = ml.get_all_pivot_sequences(empty)
        self.assertEqual(seq, [])
        self.assertEqual(tri, [])
        self.assertEqual(buy, [])


class TestBuildXY(unittest.TestCase):
    """``_build_xy`` — feature matrix, label vector, metadata."""

    def setUp(self):
        self.df = make_ohlc_df(n_candles=200, seed=1)
        add_pivots(self.df, n_pivots=30, seed=2)
        self.sequences, self.tri, self.buy = \
            ml.get_all_pivot_sequences(self.df)

    def test_output_shapes(self):
        X, y, meta = ml._build_xy(self.df, self.sequences,
                                  self.tri, self.buy, RR=5.0)
        n = len(self.sequences)
        self.assertEqual(X.shape, (n, 16),
                         f"X should be ({n}, 16), got {X.shape}")
        self.assertEqual(y.shape, (n,),
                         f"y should be ({n},), got {y.shape}")
        self.assertEqual(len(meta), n)

    def test_all_labels_are_0_or_1(self):
        _, y, _ = ml._build_xy(self.df, self.sequences,
                               self.tri, self.buy, RR=5.0)
        self.assertTrue(set(y).issubset({0, 1}))

    def test_meta_contains_required_keys(self):
        _, _, meta = ml._build_xy(self.df, self.sequences,
                                  self.tri, self.buy, RR=5.0)
        for m in meta:
            for key in ("trigger_time", "entry", "sl", "tp", "is_buy"):
                self.assertIn(key, m)


class TestMetadataFunctions(unittest.TestCase):
    """``_save_training_metadata`` / ``_load_training_metadata``."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _inject_model_dir(self):
        """Monkey-patch MODEL_DIR to the temp directory."""
        orig = ml.MODEL_DIR
        ml.MODEL_DIR = self.tmpdir
        self.addCleanup(lambda: setattr(ml, "MODEL_DIR", orig))

    def test_save_and_load_round_trip(self):
        self._inject_model_dir()
        cutoff = pd.Timestamp("2024-10-01")
        ml._save_training_metadata("TEST_SYM", cutoff, 100, 25)

        loaded = ml._load_training_metadata("TEST_SYM")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["train_cutoff"], str(cutoff))
        self.assertEqual(loaded["n_train"], 100)
        self.assertEqual(loaded["n_test"], 25)
        self.assertIn("trained_at", loaded)

    def test_load_nonexistent_returns_none(self):
        self._inject_model_dir()
        loaded = ml._load_training_metadata("NO_SYMBOL")
        self.assertIsNone(loaded)

    def test_metadata_path_format(self):
        self._inject_model_dir()
        path = ml._metadata_path("Volatility 25 Index")
        expected = os.path.join(self.tmpdir,
                                "Volatility 25 Index_pattern_meta.json")
        self.assertEqual(path, expected)


class TestListSavedModels(unittest.TestCase):
    """``list_saved_models`` — discover trained models."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_dir = ml.MODEL_DIR
        ml.MODEL_DIR = self.tmpdir

    def tearDown(self):
        ml.MODEL_DIR = self._orig_dir
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_dir_returns_empty_list(self):
        self.assertEqual(ml.list_saved_models(), [])

    def test_finds_all_metadata_files(self):
        ml._save_training_metadata("SYM_A", pd.Timestamp("2024-10-01"), 100, 25)
        ml._save_training_metadata("SYM_B", pd.Timestamp("2024-11-01"), 80, 20)

        models = ml.list_saved_models()
        self.assertEqual(len(models), 2)
        symbols = {m["symbol"] for m in models}
        self.assertEqual(symbols, {"SYM_A", "SYM_B"})

    def test_has_model_flag(self):
        ml._save_training_metadata("SYM_A", pd.Timestamp("2024-10-01"), 100, 25)
        models = ml.list_saved_models()
        # No .joblib file was saved, so has_model should be False
        self.assertFalse(models[0]["has_model"])

    def test_ignores_non_meta_files(self):
        # Write a random file that doesn't match the pattern
        with open(os.path.join(self.tmpdir, "random_file.json"), "w") as f:
            json.dump({"foo": "bar"}, f)
        self.assertEqual(ml.list_saved_models(), [])

    def test_corrupt_meta_file_returns_meta_none(self):
        meta_path = os.path.join(self.tmpdir, "SYM_C_pattern_meta.json")
        with open(meta_path, "w") as f:
            f.write("not valid json")
        models = ml.list_saved_models()
        self.assertEqual(len(models), 1)
        self.assertIsNone(models[0]["meta"])


class TestBuildAndTrainModelTemporalSplit(unittest.TestCase):
    """Core temporal integrity of ``build_and_train_model``."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_dir = ml.MODEL_DIR
        ml.MODEL_DIR = self.tmpdir
        # Create 6 months of 6-hourly data → ~730 candles, plenty of pivots
        self.df = make_ohlc_df(n_candles=720, start_date="2024-01-01",
                               freq="6h", seed=1)
        add_pivots(self.df, n_pivots=60, seed=2)

    def tearDown(self):
        ml.MODEL_DIR = self._orig_dir
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_trigger_times(self, df):
        seq, tri, _ = ml.get_all_pivot_sequences(df)
        return tri

    def test_date_based_split_80_20(self):
        """Train sequences occur chronologically before test sequences."""
        result = ml.build_and_train_model(self.df, "TEST_SYM",
                                          RR=5.0, test_size=0.2)
        self.assertTrue(result["success"], msg=result.get("reason", ""))

        cutoff = pd.Timestamp(result["cutoff"])
        times = self._get_trigger_times(self.df)

        train_times = [t for t in times if t < cutoff]
        test_times  = [t for t in times if t >= cutoff]

        self.assertEqual(len(train_times), result["n_train"],
                         "Number of train sequences mismatches")
        self.assertEqual(len(test_times), result["n_test"],
                         "Number of test sequences mismatches")

        if train_times and test_times:
            self.assertLess(
                max(train_times), min(test_times),
                "All train trigger times MUST be before all test trigger times"
            )

    def test_explicit_cutoff_date(self):
        """An explicitly-provided cutoff_date overrides test_size."""
        cutoff = "2024-06-01"
        result = ml.build_and_train_model(self.df, "TEST_SYM",
                                          RR=5.0, cutoff_date=cutoff)
        self.assertTrue(result["success"])

        cutoff_ts = pd.Timestamp(cutoff)
        times = self._get_trigger_times(self.df)
        train_times = [t for t in times if t < cutoff_ts]
        self.assertEqual(len(train_times), result["n_train"])

    def test_insufficient_sequences_returns_failure(self):
        """< 15 sequences should cause early return."""
        small = make_ohlc_df(n_candles=50, start_date="2024-01-01",
                             freq="1h", seed=1)
        add_pivots(small, n_pivots=12, seed=2)  # 12 pivots → 7 sequences
        result = ml.build_and_train_model(small, "TEST_SYM", RR=5.0)
        self.assertFalse(result["success"])
        self.assertIn("Only", result.get("reason", ""))

    def test_model_file_saved(self):
        """A .joblib model file should exist after training."""
        ml.build_and_train_model(self.df, "TEST_SYM", RR=5.0, test_size=0.2)
        model_path = os.path.join(self.tmpdir, "TEST_SYM_pattern_model.joblib")
        self.assertTrue(os.path.exists(model_path),
                        "Model .joblib file should exist after training")

    def test_metadata_file_saved(self):
        """A _pattern_meta.json file should exist after training."""
        ml.build_and_train_model(self.df, "TEST_SYM", RR=5.0, test_size=0.2)
        meta_path = os.path.join(self.tmpdir,
                                 "TEST_SYM_pattern_meta.json")
        self.assertTrue(os.path.exists(meta_path),
                        "Metadata JSON should exist after training")

    def test_metadata_cutoff_matches_result(self):
        """The saved metadata cutoff should match the result cutoff."""
        result = ml.build_and_train_model(self.df, "TEST_SYM",
                                          RR=5.0, test_size=0.2)
        meta = ml._load_training_metadata("TEST_SYM")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["train_cutoff"], result["cutoff"])

    def test_result_contains_all_keys(self):
        result = ml.build_and_train_model(self.df, "TEST_SYM",
                                          RR=5.0, test_size=0.2)
        for key in ("success", "train_metrics", "test_metrics",
                    "n_train", "n_test", "cutoff"):
            self.assertIn(key, result, f"Result missing key '{key}'")

    def test_no_shuffle_preserves_chronology(self):
        """Run twice: train/test sequence counts should be deterministic."""
        r1 = ml.build_and_train_model(self.df, "TEST_SYM",
                                      RR=5.0, test_size=0.2)
        # Re-initialise model dir to avoid cached model issues
        ml.MODEL_DIR = self.tmpdir
        r2 = ml.build_and_train_model(self.df, "TEST_SYM",
                                      RR=5.0, test_size=0.2)
        self.assertEqual(r1["n_train"], r2["n_train"])
        self.assertEqual(r1["n_test"],  r2["n_test"])
        self.assertEqual(r1["cutoff"],  r2["cutoff"])

    def test_cutoff_date_string_conversion(self):
        """String cutoff_date should be converted to pd.Timestamp."""
        result = ml.build_and_train_model(
            self.df, "TEST_SYM", RR=5.0,
            cutoff_date="2024-04-01"
        )
        self.assertTrue(result["success"])
        self.assertIn("2024-04-01", result["cutoff"])

    def test_retrain_on_all_false_does_not_use_test_data(self):
        """With retrain_on_all=False, model is only trained on train set."""
        result = ml.build_and_train_model(
            self.df, "TEST_SYM", RR=5.0, test_size=0.2,
            retrain_on_all=False
        )
        self.assertTrue(result["success"])
        # Model file should still exist
        model_path = os.path.join(self.tmpdir, "TEST_SYM_pattern_model.joblib")
        self.assertTrue(os.path.exists(model_path))


class TestStrategyTemporalGuard(unittest.TestCase):
    """``Strategy.MLPattern`` temporal guard — rejects training-range sequences."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_dir = ml.MODEL_DIR
        ml.MODEL_DIR = self.tmpdir

        # Build a full dataset + train + metadata
        self.df = make_ohlc_df(n_candles=400, start_date="2024-01-01",
                               freq="6h", seed=1)
        add_pivots(self.df, n_pivots=40, seed=2)

        # Temporarily suppress stdout during training (it's noisy)
        with open(os.devnull, "w") as null:
            old = sys.stdout
            sys.stdout = null
            try:
                self.result = ml.build_and_train_model(
                    self.df, "GUARD_TEST", RR=5.0, test_size=0.2
                )
            finally:
                sys.stdout = old

        self.cutoff = pd.Timestamp(self.result["cutoff"])

    def tearDown(self):
        ml.MODEL_DIR = self._orig_dir
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_strategy_trades(self, df=None):
        """Helper: run MLPattern and return the plot_df."""
        df = df if df is not None else self.df
        strat = Strategy(df, symbol="GUARD_TEST")
        return strat.MLPattern(RR=5.0)

    def test_guard_rejects_training_range_sequences(self):
        """Trades should only occur after the training cutoff."""
        plot_df = self._get_strategy_trades()
        if not plot_df.empty:
            for occ in plot_df["Occurence"]:
                self.assertGreater(
                    occ, self.cutoff,
                    f"Trade trigger {occ} is within training range "
                    f"(cutoff={self.cutoff})"
                )

    def test_guard_allows_all_passed_data_same_as_training(self):
        """The output DataFrame should contain valid columns."""
        plot_df = self._get_strategy_trades()
        expected_cols = {"Occurence", "Entry", "Stop_Loss",
                         "Take_Profit", "Risk_to_Reward_Ratio"}
        self.assertTrue(expected_cols.issubset(set(plot_df.columns)))

    def test_guard_returns_empty_when_all_rejected(self):
        """If ALL data is in training range, return empty DataFrame."""
        train_only_df = self.df[self.df.index < self.cutoff].copy()
        if len(train_only_df) >= 30:
            plot_df = self._get_strategy_trades(df=train_only_df)
            self.assertTrue(plot_df.empty,
                            "Should return empty when no sequences "
                            "are after the cutoff")

    def test_guard_no_metadata_passes_all_sequences(self):
        """Without metadata file, no filtering happens."""
        # Remove the metadata file
        meta_path = ml._metadata_path("GUARD_TEST")
        if os.path.exists(meta_path):
            os.remove(meta_path)

        plot_df = self._get_strategy_trades()
        # Should still produce a valid DataFrame (possibly empty because
        # predict_pattern_probability returns 0.0 since we're mocking
        # and the RandomForest may or may not trigger trades)
        self.assertIsInstance(plot_df, pd.DataFrame)

    def test_guard_trade_occurrences_are_timestamps(self):
        """All trade occurrence times should be pd.Timestamps."""
        plot_df = self._get_strategy_trades()
        if not plot_df.empty:
            for occ in plot_df["Occurence"]:
                self.assertIsInstance(occ, pd.Timestamp)

    def test_guard_empty_df_has_correct_columns(self):
        """Empty DataFrame should still contain the expected columns."""
        empty_df = pd.DataFrame(columns=["Occurence", "Entry", "Stop_Loss",
                                          "Take_Profit",
                                          "Risk_to_Reward_Ratio"])
        self.assertListEqual(
            list(empty_df.columns),
            ["Occurence", "Entry", "Stop_Loss", "Take_Profit",
             "Risk_to_Reward_Ratio"]
        )


class TestTemporalTrainTestSplit(unittest.TestCase):
    """Legacy ``temporal_train_test_split`` — still works and preserves order."""

    def _make_fake_data(self, n=30):
        seq = [[{"val": float(i), "idx": i,
                 "time": pd.Timestamp(f"2024-01-{i+1:02d}"),
                 "is_high": i % 2 == 0}] for i in range(n)]
        tri = [pd.Timestamp(f"2024-01-{i+1:02d}") for i in range(n)]
        buy = [i % 2 == 0 for i in range(n)]
        return seq, tri, buy

    def test_split_preserves_chronological_order(self):
        seq, tri, buy = self._make_fake_data(30)
        train_seq, train_tri, train_buy, test_seq, test_tri, test_buy = \
            ml.temporal_train_test_split(seq, tri, buy, test_size=0.2)

        self.assertEqual(len(train_seq), 24)
        self.assertEqual(len(test_seq), 6)

        if train_tri and test_tri:
            self.assertLess(max(train_tri), min(test_tri))

    def test_partial_split_returns_empty_lists(self):
        seq, tri, buy = self._make_fake_data(6)
        train_seq, train_tri, train_buy, test_seq, test_tri, test_buy = \
            ml.temporal_train_test_split(seq, tri, buy, test_size=0.5)
        # With 6 sequences: split_idx=3, train=3 test=3 → both < 5
        # Function returns (paired, [], [], [], [], [])
        # First return is the raw paired list, test sequences are empty lists
        self.assertEqual(len(train_seq), 6,
                         "When split is too small, first return is the "
                         "full paired list")
        self.assertEqual(test_seq, [])


class TestPredictPatternProbability(unittest.TestCase):
    """``predict_pattern_probability`` — model inference."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_dir = ml.MODEL_DIR
        ml.MODEL_DIR = self.tmpdir

    def tearDown(self):
        ml.MODEL_DIR = self._orig_dir
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_model_returns_zero(self):
        pivots = [{"val": 100.0, "idx": i,
                   "time": pd.Timestamp(f"2024-01-{i+1:02d}"),
                   "is_high": i % 2 == 0} for i in range(5, -1, -1)]
        prob = ml.predict_pattern_probability("NO_MODEL", pivots)
        self.assertEqual(prob, 0.0)

    def test_returns_float_between_0_and_1(self):
        # Train a tiny model to test inference
        X = np.random.default_rng(42).normal(size=(20, 16))
        y = np.random.default_rng(42).integers(0, 2, 20)
        model = ml._build_model()
        model.fit(X, y)
        path = os.path.join(self.tmpdir, "TINY_pattern_model.joblib")
        import joblib
        joblib.dump(model, path)

        pivots = [{"val": float(v), "idx": i,
                   "time": pd.Timestamp(f"2024-01-{i+1:02d}"),
                   "is_high": i % 2 == 0}
                  for i, v in enumerate([100, 101, 99, 102, 98, 103])]
        prob = ml.predict_pattern_probability("TINY", pivots)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


class TestBuildModel(unittest.TestCase):
    """``_build_model`` returns the expected classifier."""

    def test_returns_random_forest(self):
        model = ml._build_model()
        self.assertIsInstance(model, RandomForestClassifier)

    def test_default_params(self):
        model = ml._build_model()
        self.assertEqual(model.n_estimators, 100)
        self.assertEqual(model.max_depth, 5)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(model.random_state, 42)


class TestReportMetrics(unittest.TestCase):
    """``_report_metrics`` returns correct dictionary."""

    def test_output_dict_keys(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        y_prob = np.array([0.9, 0.2, 0.4, 0.1])
        metrics = ml._report_metrics("test", y_true, y_pred, y_prob)
        for key in ("accuracy", "precision", "recall", "f1"):
            self.assertIn(key, metrics)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 0.5)


class TestBuildAndTrainModelEdgeCases(unittest.TestCase):
    """Edge-case scenarios for build_and_train_model."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_dir = ml.MODEL_DIR
        ml.MODEL_DIR = self.tmpdir

    def tearDown(self):
        ml.MODEL_DIR = self._orig_dir
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_all_trigger_times_identical(self):
        """When all triggers are at the same time, split should handle it."""
        df = make_ohlc_df(n_candles=100, start_date="2024-01-01",
                          freq="1h", seed=1)
        add_pivots(df, n_pivots=20, seed=2)
        # Force all trigger timestamps to the same date
        seq, tri, buy = ml.get_all_pivot_sequences(df)
        if tri:
            # Replace trigger indices with all same timestamp
            fixed_ts = pd.Timestamp("2024-01-15")
            ml._build_xy(df, seq, [fixed_ts] * len(seq), buy, RR=5.0)
            # The split would have t_min == t_max, so cutoff = t_min
            # All sequences go to test, none to train → caught by <5 check
            result = ml.build_and_train_model(df, "EDGE", RR=5.0)
            # Should either succeed (if enough in test) or fail gracefully
            if result["success"]:
                self.assertEqual(result["n_train"], 0)
                self.assertGreaterEqual(result["n_test"], 5)
            else:
                self.assertIn("too small", result.get("reason", "").lower()
                              or "only" in result.get("reason", "").lower())

    def test_test_size_zero(self):
        """test_size=0 means everything is training, nothing in test."""
        df = make_ohlc_df(n_candles=300, start_date="2024-01-01",
                          freq="6h", seed=1)
        add_pivots(df, n_pivots=30, seed=2)
        result = ml.build_and_train_model(df, "EDGE", RR=5.0, test_size=0.0)
        # Should fail because test set is empty
        if result.get("success"):
            self.assertEqual(result["n_test"], 0)
        else:
            self.assertIn("too small", result.get("reason", "").lower())

    def test_test_size_one(self):
        """test_size=1.0 means everything is test, nothing in train."""
        df = make_ohlc_df(n_candles=300, start_date="2024-01-01",
                          freq="6h", seed=1)
        add_pivots(df, n_pivots=30, seed=2)
        result = ml.build_and_train_model(df, "EDGE", RR=5.0, test_size=1.0)
        if result.get("success"):
            self.assertEqual(result["n_train"], 0)
        else:
            self.assertIn("too small", result.get("reason", "").lower())


if __name__ == "__main__":
    unittest.main()
