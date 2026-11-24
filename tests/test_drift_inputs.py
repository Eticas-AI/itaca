import unittest

import numpy as np
import pandas as pd

from eticas.metrics.tdx_inconsistency import Tdx_inconsistency


class DriftInputTests(unittest.TestCase):
    def test_production_uses_its_own_index(self):
        for separate_index, complex_group in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(separate_index=separate_index, complex_group=complex_group):
                development = pd.DataFrame({
                    "group": [1] * 40 + [2] * 40,
                    "feature": [0.0] * 80,
                    "outcome": [0, 1] * 40,
                })
                production = development.copy()
                if separate_index:
                    production.index = range(1000, 1080)
                base = {"type": "simple", "columns": [{"name": "group", "privileged": [1]}]}
                sensitive = {"group": base}
                key = "group"
                if complex_group:
                    sensitive = {"combined": {"type": "complex", "groups": ["group"]}, "group": base}
                    key = "combined"
                result = Tdx_inconsistency().compute(
                    sensitive_attrs=sensitive,
                    input_features=["feature"],
                    input_data_dev=development, output_column_dev="outcome",
                    input_data_prod=production, output_column_prod="outcome",
                )
                self.assertNotIn("error", result[key])
                self.assertGreater(result[key]["accuracy"], 0)

    def test_combined_group_is_the_intersection(self):
        """A combined attribute is audited on the rows underprivileged on every component."""
        sensitive = {
            "a": {"type": "simple", "columns": [{"name": "a", "underprivileged": [2]}]},
            "b": {"type": "simple", "columns": [{"name": "b", "privileged": [1]}]},
            "a_b": {"type": "complex", "groups": ["a", "b"]},
        }

        def dataset(seed, shifted_rows=None):
            rng = np.random.default_rng(seed)
            data = pd.DataFrame({
                "a": rng.integers(1, 3, 4000),
                "b": rng.integers(1, 4, 4000),
                "feature": rng.normal(0, 1, 4000),
                "outcome": rng.integers(0, 2, 4000),
            })
            if shifted_rows is not None:
                data.loc[shifted_rows(data), "feature"] += 3
            return data

        def drift(shifted_rows):
            return Tdx_inconsistency().compute(
                sensitive_attrs=sensitive,
                input_features=["feature"],
                input_data_dev=dataset(1), output_column_dev="outcome",
                input_data_prod=dataset(2, shifted_rows), output_column_prod="outcome",
            )

        # shift in rows that are underprivileged on b only: outside the combined group
        outside = drift(lambda d: (d.a == 1) & (d.b != 1))
        self.assertEqual(outside["b"]["bias_level"], "High bias")
        self.assertEqual(outside["a_b"]["bias_level"], "Low/no bias")

        # shift in the rows underprivileged on both: inside the combined group
        inside = drift(lambda d: (d.a == 2) & (d.b != 1))
        self.assertEqual(inside["a_b"]["bias_level"], "High bias")
        self.assertNotIn("error", inside["a_b"])
        self.assertNotIn("error", outside["a_b"])

    def test_inputs_are_not_modified(self):
        development = pd.DataFrame({
            "group": [1] * 40 + [2] * 40,
            "feature": [0.0] * 80,
            "outcome": [0, 1] * 40,
        })
        production = development.copy()
        before_dev, before_prod = development.copy(), production.copy()
        Tdx_inconsistency().compute(
            sensitive_attrs={"group": {"type": "simple", "columns": [{"name": "group", "underprivileged": [2]}]}},
            input_features=["feature"],
            input_data_dev=development, output_column_dev="outcome",
            input_data_prod=production, output_column_prod="outcome",
        )
        pd.testing.assert_frame_equal(development, before_dev)
        pd.testing.assert_frame_equal(production, before_prod)

    def test_group_size_change_is_not_drift(self):
        """A group that changes size, with unchanged features, is not reported as drifting."""
        sensitive = {"group": {"type": "simple", "columns": [{"name": "group", "underprivileged": [2]}]}}

        def dataset(seed, share, shift=0.0):
            rng = np.random.default_rng(seed)
            data = pd.DataFrame({
                "group": np.where(rng.random(6000) < share, 2, 1),
                "feature_1": rng.normal(0, 1, 6000),
                "feature_2": rng.normal(0, 1, 6000),
                "outcome": rng.integers(0, 2, 6000),
            })
            data.loc[data.group == 2, "feature_1"] += shift
            return data

        def drift(production):
            return Tdx_inconsistency().compute(
                sensitive_attrs=sensitive,
                input_features=["feature_1", "feature_2"],
                input_data_dev=dataset(1, 0.5), output_column_dev="outcome",
                input_data_prod=production, output_column_prod="outcome",
            )["group"]

        # the group goes from half to a quarter of the rows; its features do not change
        smaller = drift(dataset(2, 0.25))
        self.assertEqual(smaller["proportion"], 0.5)
        self.assertEqual(smaller["bias_level"], "Low/no bias")
        self.assertGreater(smaller["normalized_risk"], 90)

        # same change of size, and the group's features do shift
        shifted = drift(dataset(2, 0.25, shift=3.0))
        self.assertEqual(shifted["bias_level"], "High bias")

    def test_complete_drift_detected_whatever_the_sizes(self):
        """Complete drift is 'High bias' even when production is much smaller than development."""
        sensitive = {"group": {"type": "simple", "columns": [{"name": "group", "underprivileged": [2]}]}}

        def dataset(seed, rows, shift):
            rng = np.random.default_rng(seed)
            return pd.DataFrame({
                "group": [2] * rows,
                "feature": rng.normal(0, 1, rows) + shift,
                "outcome": rng.integers(0, 2, rows),
            })

        for development_rows in (1000, 5000, 9000, 20000):
            for shift, expected in ((0.0, "Low/no bias"), (10.0, "High bias")):
                result = Tdx_inconsistency().compute(
                    sensitive_attrs=sensitive,
                    input_features=["feature"],
                    input_data_dev=dataset(1, development_rows, 0.0), output_column_dev="outcome",
                    input_data_prod=dataset(2, 1000, shift), output_column_prod="outcome",
                )
                for key in ("overall", "group"):
                    self.assertEqual(result[key]["bias_level"], expected,
                                     msg=f"{development_rows}:1000 rows, shift {shift}, {key}: {result[key]}")
                if shift:
                    self.assertLess(result["group"]["normalized_risk"], 5)
