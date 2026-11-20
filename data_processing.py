"""Load, inspect, clean, and merge the project datasets."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
INTERNET_FILE = DATA_DIR / "share-of-individuals-using-the-internet.csv"
GDP_FILE = DATA_DIR / "gdp-per-capita-worldbank.csv"
LIFE_FILE = DATA_DIR / "life-expectancy.csv"


def load_data():
    """Load the three source datasets and return them as DataFrames."""
    internet = pd.read_csv(INTERNET_FILE)
    gdp = pd.read_csv(GDP_FILE)
    life = pd.read_csv(LIFE_FILE)
    return internet, gdp, life


def clean_and_merge(internet, gdp, life):
    """Merge country-year observations without missing country codes."""
    # Keep three-letter country codes and exclude regional aggregates.
    internet = internet[
        internet["code"].str.fullmatch(r"[A-Z]{3}", na=False)
    ].copy()
    gdp = gdp[
        gdp["code"].str.fullmatch(r"[A-Z]{3}", na=False)
    ].copy()
    life = life[
        life["code"].str.fullmatch(r"[A-Z]{3}", na=False)
    ].copy()

    internet_gdp = internet.merge(
        gdp, on=["code", "year"], how="inner",
        suffixes=("_internet", "_gdp")
    )
    internet_life = internet.merge(
        life, on=["code", "year"], how="inner",
        suffixes=("_internet", "_life")
    )
    all_three = internet_gdp.merge(
        life, on=["code", "year"], how="inner"
    )
    return internet_gdp, internet_life, all_three