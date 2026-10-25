import unittest

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
