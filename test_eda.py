"""Tests for the internet access EDA functions."""

import unittest

import pandas as pd

from data_processing import clean_and_merge


class TestCleanAndMerge(unittest.TestCase):
    """Check that the merge function matches country-year observations."""

    def setUp(self):
        """Create small example datasets with known matching rows."""
        self.internet = pd.DataFrame({
            "entity": ["A", "A", "B"],
            "code": ["AAA", "AAA", "BBB"],
            "year": [2000, 2001, 2000],
            "it_net_user_zs": [10.0, 20.0, 30.0]
        })
        self.gdp = pd.DataFrame({
            "entity": ["A", "A", "B"],
            "code": ["AAA", "AAA", "BBB"],
            "year": [2000, 2001, 2001],
            "ny_gdp_pcap_pp_kd": [1000.0, 1100.0, 2000.0],
            "owid_region": ["Europe", "Europe", "Asia"]
        })
        self.life = pd.DataFrame({
            "entity": ["A", "B", "Aggregate"],
            "code": ["AAA", "BBB", None],
            "year": [2000, 2001, 2000],
            "life_expectancy_0": [70.0, 75.0, 65.0]
        })

    def test_merge_uses_matching_country_and_year(self):
        """Only rows with the same code and year should be joined."""
        internet_gdp, internet_life, all_three = clean_and_merge(
            self.internet, self.gdp, self.life
        )
        self.assertEqual(len(internet_gdp), 2)
        self.assertEqual(len(internet_life), 1)
        self.assertEqual(len(all_three), 1)
        self.assertEqual(
            all_three.iloc[0]["code"], "AAA"
        )
        self.assertEqual(
            all_three.iloc[0]["year"], 2000
        )

    def test_aggregate_without_code_is_excluded(self):
        """Rows without country codes should not enter country comparisons."""
        _, _, all_three = clean_and_merge(
            self.internet, self.gdp, self.life
        )
        self.assertNotIn(None, all_three["code"].tolist())

    def test_no_duplicate_country_years_after_merge(self):
        """The merged data should have one row per country and year."""
        internet_gdp, internet_life, all_three = clean_and_merge(
            self.internet, self.gdp, self.life
        )
        for data in [internet_gdp, internet_life, all_three]:
            self.assertEqual(
                data.duplicated(["code", "year"]).sum(), 0
            )

    def test_regional_aggregates_are_excluded(self):
        """Codes such as OWID_WRL should not enter country comparisons."""
        internet = pd.concat([
            self.internet,
            pd.DataFrame({
                "entity": ["World"], "code": ["OWID_WRL"],
                "year": [2000], "it_net_user_zs": [20.0]
            })
        ], ignore_index=True)
        gdp = pd.concat([
            self.gdp,
            pd.DataFrame({
                "entity": ["World"], "code": ["OWID_WRL"],
                "year": [2000], "ny_gdp_pcap_pp_kd": [5000.0],
                "owid_region": [None]
            })
        ], ignore_index=True)
        _, _, all_three = clean_and_merge(
            internet, gdp, self.life
        )
        self.assertNotIn("OWID_WRL", all_three["code"].tolist())


if __name__ == "__main__":
    unittest.main()
