# Does Internet Access Reflect Economic Development and Quality of Life?

CSE 163 — Autumn 2026 | Author: Sia Razdan

## Project overview
This project studies associations among internet use, GDP per capita, and life
expectancy across coded country/entity observations. Cross-country comparisons
use 2023, the latest year represented in all three source datasets. The change
analysis compares 2000 with 2023. The analysis is descriptive and does not
establish causation.

## Files
- `analysis.py`: runnable analysis program. Loads the data through the data
  processing module, creates summary tables and Pearson correlations, performs
  the 2000–2023 change analysis, and generates all figures.
- `data_processing.py`: reads the source CSV files, filters to three-letter
  country/entity codes, and merges datasets using both `code` and `year`.
- `test_eda.py`: automated unit tests for the data-cleaning and merge functions.
- `data/`: the three original Our World in Data CSV files. Do not edit these
  source files manually.
- `outputs/`: generated CSV summaries and figures. Running `analysis.py`
  regenerates these outputs.
- `requirements.txt`: Python package dependencies.

## Requirements and setup
Use Python 3 and install the packages from the project directory:

```bash
python -m pip install -r requirements.txt
```

The project uses pandas for data processing, Matplotlib for plots, SciPy for
Pearson correlation tests, and the Python standard library for file paths and
testing.

## Run the analysis
Open a terminal and change into the `internet_access_eda_project` folder. Run:

```bash
python analysis.py
```

The program reads the CSV files from `data/` and writes tables to `outputs/`
and plots to `outputs/figures/`. Run it from the project root so the relative
file paths resolve correctly.

## Run the tests
From the same project root, run:

```bash
python -m unittest test_eda.py -v
```

The tests use small in-memory DataFrames and check that merges match both
country/entity code and year, missing codes are excluded, aggregate codes are
excluded, and country-year observations are not duplicated in the test data.

## Check code style
Install Flake8 and run it on all submitted Python modules:

```bash
python -m pip install flake8
flake8 analysis.py data_processing.py test_eda.py
```

## Reproducibility notes
The analysis retains rows with three-letter uppercase codes. This removes rows
without usable codes and many aggregate codes, although some territories or
other coded entities may remain. All merges use both `code` and `year`. The
2023 comparisons use pairwise complete observations; the change analysis
requires measurements for both variables in both 2000 and 2023. The GDP
per-capita variable is used in the units provided by the source dataset.
