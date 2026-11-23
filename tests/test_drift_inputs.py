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
