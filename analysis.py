"""Run the exploratory and statistical analysis for the project."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

from data_processing import clean_and_merge, load_data

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
INTERNET_COL = "it_net_user_zs"
GDP_COL = "ny_gdp_pcap_pp_kd"
LIFE_COL = "life_expectancy_0"
COMPARISON_YEAR = 2023
START_YEAR = 2000
END_YEAR = 2023


def save_dataset_inspection(datasets):
    """Save dimensions, missing values, and duplicate counts."""
    rows = []
    for name, data in datasets.items():
        country_mask = data["code"].str.fullmatch(
            r"[A-Z]{3}", na=False
        )
        country_rows = data[country_mask]
        rows.append({
            "dataset": name,
            "rows": len(data),
            "columns": len(data.columns),
            "missing_cells": int(data.isna().sum().sum()),
            "missing_code_rows": int(data["code"].isna().sum()),
            "aggregate_code_rows": int(
                ((~country_mask) & data["code"].notna()).sum()
            ),
            "duplicate_rows": int(data.duplicated().sum()),
            "duplicate_country_year": int(
                country_rows.duplicated(["code", "year"]).sum()
            ),
            "first_year": int(data["year"].min()),
            "last_year": int(data["year"].max()),
            "unique_entities": int(data["entity"].nunique())
        })
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "dataset_inspection.csv", index=False
    )

    missing_rows = []
    for name, data in datasets.items():
        for column in data.columns:
            missing_rows.append({
                "dataset": name,
                "column": column,
                "missing_count": int(data[column].isna().sum()),
                "missing_percent": (
                    data[column].isna().mean() * 100
                )
            })
    pd.DataFrame(missing_rows).to_csv(
        OUTPUT_DIR / "missing_values.csv", index=False
    )


def save_summary_tables(internet_gdp, internet_life, all_three):
    """Save summaries for each merged dataset in the common year."""
    datasets = {
        "internet_gdp": (
            internet_gdp[internet_gdp["year"] == COMPARISON_YEAR],
            [INTERNET_COL, GDP_COL]
        ),
        "internet_life": (
            internet_life[internet_life["year"] == COMPARISON_YEAR],
            [INTERNET_COL, LIFE_COL]
        ),
        "all_three": (
            all_three[all_three["year"] == COMPARISON_YEAR],
            [INTERNET_COL, GDP_COL, LIFE_COL]
        )
    }
    summary_rows = []
    for name, (data, variables) in datasets.items():
        for variable in variables:
            stats = data[variable].describe()
            summary_rows.append({
                "dataset": name,
                "variable": variable,
                "count": stats["count"],
                "mean": stats["mean"],
                "standard_deviation": stats["std"],
                "minimum": stats["min"],
                "first_quartile": stats["25%"],
                "median": stats["50%"],
                "third_quartile": stats["75%"],
                "maximum": stats["max"]
            })
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "summary_statistics_2023.csv", index=False
    )

    current = internet_gdp[
        internet_gdp["year"] == COMPARISON_YEAR
    ].copy()
    current["owid_region"].value_counts(dropna=False).rename_axis(
        "region"
    ).reset_index(name="count").to_csv(
        OUTPUT_DIR / "region_counts_2023.csv", index=False
    )


def save_correlations(internet_gdp, internet_life, all_three):
    """Calculate Pearson correlations for each intended data combination."""
    comparisons = [
        (
            "Internet access and GDP per capita",
            internet_gdp[internet_gdp["year"] == COMPARISON_YEAR],
            INTERNET_COL, GDP_COL
        ),
        (
            "Internet access and life expectancy",
            internet_life[internet_life["year"] == COMPARISON_YEAR],
            INTERNET_COL, LIFE_COL
        ),
        (
            "GDP per capita and life expectancy",
            all_three[all_three["year"] == COMPARISON_YEAR],
            GDP_COL, LIFE_COL
        )
    ]
    results = []
    for label, data, first, second in comparisons:
        pair_data = data[[first, second]].dropna()
        coefficient, p_value = pearsonr(
            pair_data[first], pair_data[second]
        )
        results.append({
            "comparison": label,
            "first_variable": first,
            "second_variable": second,
            "observations": len(pair_data),
            "pearson_r": coefficient,
            "p_value": p_value
        })
    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "pearson_correlations_2023.csv", index=False
    )



def save_change_analysis(internet_gdp):
    """Calculate country-level changes between 2000 and 2023."""
    selected = internet_gdp[
        internet_gdp["year"].isin([START_YEAR, END_YEAR])
    ]
    wide = selected.pivot(
        index=["code", "entity_internet"],
        columns="year",
        values=[INTERNET_COL, GDP_COL]
    )
    required = [
        (INTERNET_COL, START_YEAR),
        (INTERNET_COL, END_YEAR),
        (GDP_COL, START_YEAR),
        (GDP_COL, END_YEAR)
    ]
    complete = wide.dropna(subset=required).copy()
    complete["internet_change"] = (
        complete[(INTERNET_COL, END_YEAR)]
        - complete[(INTERNET_COL, START_YEAR)]
    )
    complete["gdp_change"] = (
        complete[(GDP_COL, END_YEAR)]
        - complete[(GDP_COL, START_YEAR)]
    )
    complete.to_csv(OUTPUT_DIR / "country_changes_2000_2023.csv")

    summary = complete[["internet_change", "gdp_change"]].describe().T
    summary.to_csv(OUTPUT_DIR / "change_summary_2000_2023.csv")

    coefficient, p_value = pearsonr(
        complete["internet_change"], complete["gdp_change"]
    )
    pd.DataFrame([{
        "observations": len(complete),
        "pearson_r": coefficient,
        "p_value": p_value
    }]).to_csv(OUTPUT_DIR / "change_correlation_2000_2023.csv",
               index=False)


def create_figures(internet_gdp, internet_life, all_three):
    """Create plots for each merged dataset and the change analysis."""
    ig = internet_gdp[
        internet_gdp["year"] == COMPARISON_YEAR
    ].copy()
    il = internet_life[
        internet_life["year"] == COMPARISON_YEAR
    ].copy()
    current = all_three[
        all_three["year"] == COMPARISON_YEAR
    ].copy()

    # Internet access and GDP per capita.
    plt.figure(figsize=(8, 5))
    plt.scatter(ig[INTERNET_COL], ig[GDP_COL], alpha=0.7)
    plt.xlabel("Population using the internet (%)")
    plt.ylabel("GDP per capita")
    plt.title("Internet Access and GDP per Capita Across Countries (2023)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "internet_gdp_2023.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    ig.boxplot(column=GDP_COL, by="owid_region", rot=25)
    plt.title("GDP per Capita by Region (2023)")
    plt.suptitle("")
    plt.xlabel("World region")
    plt.ylabel("GDP per capita")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "gdp_by_region_2023.png", dpi=200)
    plt.close()

    # Internet access and life expectancy.
    plt.figure(figsize=(8, 5))
    plt.scatter(il[INTERNET_COL], il[LIFE_COL], alpha=0.7)
    plt.xlabel("Population using the internet (%)")
    plt.ylabel("Life expectancy at birth (years)")
    plt.title("Internet Access and Life Expectancy Across Countries (2023)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "internet_life_2023.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(il[LIFE_COL], bins=15, edgecolor="black")
    plt.title("Distribution of Life Expectancy Across Countries (2023)")
    plt.xlabel("Life expectancy at birth (years)")
    plt.ylabel("Number of countries")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "life_expectancy_distribution_2023.png", dpi=200)
    plt.close()

    # Three-variable comparison.
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        current[GDP_COL], current[LIFE_COL],
        c=current[INTERNET_COL], alpha=0.75
    )
    plt.colorbar(scatter, label="Population using the internet (%)")
    plt.xlabel("GDP per capita")
    plt.ylabel("Life expectancy at birth (years)")
    plt.title(
        "GDP per Capita and Life Expectancy, Colored by Internet Use "
        "(2023)"
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "three_variable_comparison_2023.png", dpi=200)
    plt.close()

    corr = current[[INTERNET_COL, GDP_COL, LIFE_COL]].corr()
    plt.figure(figsize=(7, 5))
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(label="Pearson correlation")
    labels = ["Internet use", "GDP per capita", "Life expectancy"]
    plt.xticks(range(3), labels, rotation=20, ha="right")
    plt.yticks(range(3), labels)
    for row in range(3):
        for column in range(3):
            plt.text(
                column, row, f"{corr.iloc[row, column]:.2f}",
                ha="center", va="center"
            )
    plt.title("Correlation Between the Three Variables (2023)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_matrix_2023.png", dpi=200)
    plt.close()

    # Changes from 2000 to 2023, using the internet-GDP pair.
    selected = internet_gdp[
        internet_gdp["year"].isin([START_YEAR, END_YEAR])
    ]
    wide = selected.pivot(
        index=["code", "entity_internet"],
        columns="year",
        values=[INTERNET_COL, GDP_COL]
    )
    complete = wide.dropna(subset=[
        (INTERNET_COL, START_YEAR), (INTERNET_COL, END_YEAR),
        (GDP_COL, START_YEAR), (GDP_COL, END_YEAR)
    ]).copy()
    complete["internet_change"] = (
        complete[(INTERNET_COL, END_YEAR)]
        - complete[(INTERNET_COL, START_YEAR)]
    )
    complete["gdp_change"] = (
        complete[(GDP_COL, END_YEAR)]
        - complete[(GDP_COL, START_YEAR)]
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(
        complete["internet_change"], complete["gdp_change"], alpha=0.7
    )
    plt.xlabel("Change in internet use (percentage points)")
    plt.ylabel("Change in GDP per capita")
    plt.title("Changes in Internet Access and GDP per Capita (2000–2023)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "changes_internet_gdp_2000_2023.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    complete[["internet_change", "gdp_change"]].plot(
        kind="box", figsize=(8, 5)
    )
    plt.title("Distribution of Country-Level Changes (2000–2023)")
    plt.ylabel("Change in measurement (different units)")
    plt.xlabel("Variable")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "change_distributions_2000_2023.png", dpi=200)
    plt.close()



def main():
    """Run the complete exploratory data analysis."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    internet, gdp, life = load_data()
    datasets = {
        "internet": internet,
        "gdp": gdp,
        "life_expectancy": life
    }
    save_dataset_inspection(datasets)

    internet_gdp, internet_life, all_three = clean_and_merge(
        internet, gdp, life
    )
    pd.DataFrame([
        {
            "merged_dataset": "internet_gdp",
            "rows": len(internet_gdp),
            "unique_countries": internet_gdp["code"].nunique()
        },
        {
            "merged_dataset": "internet_life",
            "rows": len(internet_life),
            "unique_countries": internet_life["code"].nunique()
        },
        {
            "merged_dataset": "all_three",
            "rows": len(all_three),
            "unique_countries": all_three["code"].nunique()
        }
    ]).to_csv(OUTPUT_DIR / "merge_inspection.csv", index=False)

    save_summary_tables(internet_gdp, internet_life, all_three)
    save_correlations(internet_gdp, internet_life, all_three)
    save_change_analysis(internet_gdp)
    create_figures(internet_gdp, internet_life, all_three)


if __name__ == "__main__":
    main()
