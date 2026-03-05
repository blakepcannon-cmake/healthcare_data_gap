# healthcare_data_gap | Notebook Collection
Tracing dollars through the drug ecosystem - inspired by Mark Cuban and Eric Bricker

# Healthcare Data Case Study: Notebook Collection

## Overview

Six Databricks notebooks that trace a pharmaceutical dollar from FDA registration through
drug pricing, distribution, claims adjudication, and patient outcomes. Built entirely on
free public datasets and Databricks Free Edition.

## Notebook Sequence

| # | File | Layer | What It Does |
|---|------|-------|--------------|
| 1 | `01_bronze_ingestion.py` | Bronze | Loads raw CSVs into Delta tables. No transformations. |
| 2 | `02_ndc_normalization_drug_dim.py` | Silver | Normalizes NDC formats across datasets. Builds dim_drug. |
| 3 | `03_claims_pricing_silver.py` | Silver | Transforms beneficiary, claims, and pricing tables. |
| 4 | `04_gold_layer_models.py` | Gold | Joins across ecosystem boundaries. Four analytical tables. |
| 5 | `05_exploratory_analysis.py` | Analysis | Four product hypotheses derived from data patterns. |
| 6 | `06_ai_ml_extension.py` | ML | Two proof-of-concept models (adherence + pricing anomaly). |

## Datasets to Download Before Starting

1. **FDA NDC Directory** (product.csv + package.csv)
   - https://open.fda.gov/data/ndc/
   - Note: These are tab-delimited, not comma-delimited

2. **NADAC 2024** (CSV)
   - https://data.medicaid.gov (search "NADAC 2024")
   - Full year is ~1.5M rows; filter to Q1 if needed for compute

3. **CMS DE-SynPUF Sample 1** (8 CSV files)
   - https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf
   - Or search "CMS SynPUF" on Databricks Marketplace
   - Download ONLY Sample 1: Beneficiary (3 files), PDE, Inpatient, Outpatient, Carrier (2 files)

4. **Medicare Part D Spending by Drug** (CSV)
   - https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-medicaid-spending-by-drug

## Databricks Setup

1. Create a free account at https://signup.databricks.com
2. Create a volume: `CREATE VOLUME IF NOT EXISTS workspace.default.healthcare_data;`
3. Upload all CSVs to that volume via the Databricks UI
4. Run notebooks in order (1 through 6)

## Schemas Created

- `workspace.bronze` -- Raw ingested data
- `workspace.silver` -- Cleaned, typed, normalized tables
- `workspace.gold` -- Business-ready analytical models

## How to Import These Notebooks

These files use Databricks source format with `# COMMAND ----------` cell separators.
You can either:
- **Import directly:** In Databricks, go to Workspace > Import > select the .py file
- **Copy/paste cells:** Open the file, split at each `# COMMAND ----------` line,
  and paste each section into a new cell. Lines starting with `# MAGIC %md` are
  markdown cells; lines starting with `# MAGIC %sql` are SQL cells.

## Key Design Decisions

- **Medallion architecture** (bronze/silver/gold) matches Databricks best practices
  and the kind of data platform McKesson already uses
- **NDC normalization** is isolated in its own notebook because it is the single
  most critical piece of cross-domain join logic
- **Quality checks** are embedded in silver-layer notebooks, not separate
- **Gold tables** each cross at least two ecosystem boundaries
- **ML models** use scikit-learn (not Spark ML) to stay within Free Edition constraints
- **Synthetic data caveat:** SynPUF uses synthetic NDCs, so cross-dataset join rates
  will be lower than real-world. The architecture is the point, not the match rates.
