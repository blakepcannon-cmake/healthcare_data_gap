# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 1: Data Ingestion (Bronze Layer)
# MAGIC
# MAGIC **Purpose:** Load raw public healthcare datasets into Delta tables with zero transformations.  
# MAGIC The bronze layer preserves source data exactly as downloaded. All cleaning happens in silver.
# MAGIC
# MAGIC **Datasets:**
# MAGIC - FDA National Drug Code (NDC) Directory (product + package files)
# MAGIC - NADAC (National Average Drug Acquisition Cost) 2024
# MAGIC - CMS DE-SynPUF Sample 1 (beneficiary, PDE, inpatient, outpatient, carrier)
# MAGIC - Medicare Part D Spending by Drug
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC 1. Download CSVs from the URLs listed in each section below
# MAGIC 2. Upload them to your Databricks volume at `/Volumes/workspace/default/healthcare_data/`
# MAGIC 3. If the volume doesn't exist yet, run the setup cell first

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Setup

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create the volume for raw file uploads if it doesn't exist
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.default.healthcare_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a schema to hold all bronze tables
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.bronze;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify File Uploads
# MAGIC
# MAGIC Run this cell after uploading your CSVs to confirm they're visible.  
# MAGIC You should see all your uploaded files listed.

# COMMAND ----------

import os

volume_path = "/Volumes/workspace/default/healthcare_data"

print("Files in volume:")
print("-" * 60)
for f in os.listdir(volume_path):
    size_mb = os.path.getsize(os.path.join(volume_path, f)) / (1024 * 1024)
    print(f"  {f:<50} {size_mb:>8.1f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. FDA NDC Directory
# MAGIC
# MAGIC **Download from:** https://open.fda.gov/data/ndc/  
# MAGIC **Files needed:** `product.csv`, `package.csv`  
# MAGIC **Expected sizes:** ~130K product rows, ~180K package rows
# MAGIC
# MAGIC The NDC Directory contains every drug product registered with the FDA.
# MAGIC Each product has a labeler (manufacturer), product details, and one or more packages.
# MAGIC The PRODUCTNDC field links product to package files.

# COMMAND ----------

# FDA NDC Product file
df_ndc_product = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("sep", "\t")  # FDA NDC files are tab-delimited
    #.csv(f"{volume_path}/product.csv") # original
    .csv(f"{volume_path}/product.txt")
)

row_count = df_ndc_product.count()
print(f"FDA NDC Product: {row_count:,} rows, {len(df_ndc_product.columns)} columns")
print(f"Columns: {df_ndc_product.columns}")

(
    df_ndc_product.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.bronze.fda_ndc_product")
)

# COMMAND ----------

# FDA NDC Package file
df_ndc_package = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("sep", "\t")  # Also tab-delimited
    # .csv(f"{volume_path}/package.csv") # original
    .csv(f"{volume_path}/package.txt")
)

row_count = df_ndc_package.count()
print(f"FDA NDC Package: {row_count:,} rows, {len(df_ndc_package.columns)} columns")
print(f"Columns: {df_ndc_package.columns}")

(
    df_ndc_package.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.bronze.fda_ndc_package")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. NADAC (National Average Drug Acquisition Cost)
# MAGIC
# MAGIC **Download from:** https://data.medicaid.gov (search "NADAC 2024")  
# MAGIC **Dataset ID:** 99315a95-37ac-4eee-946a-3c523b4c481e  
# MAGIC **Full year:** ~1.5M rows. For Free Edition, filter to Q1 2024 before uploading, 
# MAGIC or load the full file and we'll filter in silver.
# MAGIC
# MAGIC NADAC represents what pharmacies actually pay to acquire drugs from distributors.
# MAGIC This is the closest public proxy to McKesson's sell-through price.

# COMMAND ----------

# DBTITLE 1,Cell 11
df_nadac = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{volume_path}/nadac*.csv")  # Glob pattern catches any NADAC file naming
)

row_count = df_nadac.count()
print(f"NADAC: {row_count:,} rows, {len(df_nadac.columns)} columns")
print(f"Columns: {df_nadac.columns}")

(
    df_nadac.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.bronze.nadac_pricing")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. CMS DE-SynPUF Sample 1
# MAGIC
# MAGIC **Download from:** https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf  
# MAGIC **Alternative:** Search "CMS SynPUF" on Databricks Marketplace  
# MAGIC **Use Sample 1 only** to stay within Free Edition compute limits.
# MAGIC
# MAGIC Sample 1 contains ~116K synthetic Medicare beneficiaries and all their linked claims:
# MAGIC - Beneficiary Summary (one file per year: 2008, 2009, 2010)
# MAGIC - Inpatient Claims (3-year file)
# MAGIC - Outpatient Claims (3-year file)
# MAGIC - Carrier Claims (split into two files: Carrier A and Carrier B)
# MAGIC - Prescription Drug Events / PDE (3-year file)
# MAGIC
# MAGIC All files for a given beneficiary share the same DESYNPUF_ID.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Beneficiary Summary Files
# MAGIC
# MAGIC Three files (2008, 2009, 2010) with time-varying fields like chronic conditions and 
# MAGIC reimbursement amounts. We load all three and union them.

# COMMAND ----------

# MAGIC %skip
# MAGIC
# MAGIC ### Modified many of thethe 3a/b/c cells to skip file reads and pull directly from catalog since connected to Databricks marketplace
# MAGIC
# MAGIC # Load all beneficiary files and tag with source year
# MAGIC # File naming convention: DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv
# MAGIC from pyspark.sql.functions import lit, input_file_name, regexp_extract
# MAGIC
# MAGIC df_bene = (
# MAGIC     spark.read
# MAGIC     .option("header", "true")
# MAGIC     .option("inferSchema", "true")
# MAGIC     .csv(f"{volume_path}/DE1_0_*_Beneficiary_Summary_File_Sample_1.csv")
# MAGIC     .withColumn("_source_file", input_file_name())
# MAGIC     .withColumn(
# MAGIC         "_source_year",
# MAGIC         regexp_extract("_source_file", r"(\d{4})_Beneficiary", 1)
# MAGIC     )
# MAGIC )
# MAGIC
# MAGIC row_count = df_bene.count()
# MAGIC print(f"Beneficiary Summary (all years): {row_count:,} rows")
# MAGIC print(f"Columns: {df_bene.columns}")
# MAGIC
# MAGIC (
# MAGIC     df_bene.write
# MAGIC     .mode("overwrite")
# MAGIC     .option("overwriteSchema", "true")
# MAGIC     .saveAsTable("workspace.bronze.synpuf_beneficiary")
# MAGIC )

# COMMAND ----------

# Load beneficiary summary from external table
df_bene = spark.table("databricks_cms_synthetic_public_use_files_synpuf.cms_synpuf_ext.ben_sum")

row_count = df_bene.count()
print(f"Beneficiary Summary (all years): {row_count:,} rows")
print(f"Columns: {df_bene.columns}")

(
    df_bene.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.bronze.synpuf_beneficiary")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Prescription Drug Events (PDE)
# MAGIC
# MAGIC This is the core claims file for this case study. Each row is a single prescription fill.
# MAGIC Key fields: DESYNPUF_ID (patient), PROD_SRVC_ID (NDC code), QTY_DSPNSD_NUM, 
# MAGIC DAYS_SUPLY_NUM, PTNT_PAY_AMT, TOT_RX_CST_AMT.

# COMMAND ----------

###
# df_pde = (
#    spark.read
#    .option("header", "true")
#    .option("inferSchema", "true")
#    .csv(f"{volume_path}/DE1_0_2008_2010_Prescription_Drug_Events_Sample_1.csv")
#)
df_pde = spark.table("databricks_cms_synthetic_public_use_files_synpuf.cms_synpuf_ext.rx_claims")


row_count = df_pde.count()
print(f"Prescription Drug Events: {row_count:,} rows")
print(f"Columns: {df_pde.columns}")

(
    df_pde.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.bronze.synpuf_pde")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c. Inpatient Claims

# COMMAND ----------

#df_inpatient = (
#    spark.read
#    .option("header", "true")
##    .option("inferSchema", "true")
#    .csv(f"{volume_path}/DE1_0_2008_2010_Inpatient_Claims_Sample_1.csv")
#)

df_inpatient = spark.table("databricks_cms_synthetic_public_use_files_synpuf.cms_synpuf_ext.inp_claims")


row_count = df_inpatient.count()
print(f"Inpatient Claims: {row_count:,} rows")
print(f"Columns: {df_inpatient.columns}")

(
    df_inpatient.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.bronze.synpuf_inpatient")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3d. Outpatient Claims

# COMMAND ----------

#df_outpatient = (
#    spark.read
#    .option("header", "true")
#    .option("inferSchema", "true")
#    .csv(f"{volume_path}/DE1_0_2008_2010_Outpatient_Claims_Sample_1.csv")
#)

df_outpatient = spark.table("databricks_cms_synthetic_public_use_files_synpuf.cms_synpuf_ext.out_claims")


row_count = df_outpatient.count()
print(f"Outpatient Claims: {row_count:,} rows")
print(f"Columns: {df_outpatient.columns}")

(
    df_outpatient.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.bronze.synpuf_outpatient")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3e. Carrier Claims
# MAGIC
# MAGIC Carrier claims are split into two CSV files per sample due to size.
# MAGIC Both must be loaded and unioned.

# COMMAND ----------

#df_carrier = (
#    spark.read
#    .option("header", "true")
#    .option("inferSchema", "true")
#    .csv(f"{volume_path}/DE1_0_2008_2010_Carrier_Claims_Sample_1*.csv")
#)

df_carrier = spark.table("databricks_cms_synthetic_public_use_files_synpuf.cms_synpuf_ext.car_claims")


row_count = df_carrier.count()
print(f"Carrier Claims (A + B combined): {row_count:,} rows")
print(f"Columns: {df_carrier.columns}")

(
    df_carrier.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.bronze.synpuf_carrier")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Medicare Part D Spending by Drug
# MAGIC
# MAGIC **Download from:** https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-medicaid-spending-by-drug  
# MAGIC **Size:** ~4,500 drugs per year
# MAGIC
# MAGIC Aggregate spending and utilization by drug across the entire Medicare Part D program.
# MAGIC Provides a macro view that complements the claim-level SynPUF PDE data.

# COMMAND ----------

df_partd = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    #.csv(f"{volume_path}/*part*d*spend*.csv")  # Flexible glob for various file names #not flexible enough
    .csv(f"{volume_path}/DSD_PTD_RY25_P04_V10_DY23_BGM.csv") 
    
)

row_count = df_partd.count()
print(f"Part D Spending by Drug: {row_count:,} rows")
print(f"Columns: {df_partd.columns}")

(
    df_partd.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.bronze.partd_spending")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Layer Inventory
# MAGIC
# MAGIC Final validation: list all bronze tables and their row counts.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     'fda_ndc_product' AS table_name, COUNT(*) AS row_count FROM workspace.bronze.fda_ndc_product
# MAGIC UNION ALL
# MAGIC SELECT 'fda_ndc_package', COUNT(*) FROM workspace.bronze.fda_ndc_package
# MAGIC UNION ALL
# MAGIC SELECT 'nadac_pricing', COUNT(*) FROM workspace.bronze.nadac_pricing
# MAGIC UNION ALL
# MAGIC SELECT 'synpuf_beneficiary', COUNT(*) FROM workspace.bronze.synpuf_beneficiary
# MAGIC UNION ALL
# MAGIC SELECT 'synpuf_pde', COUNT(*) FROM workspace.bronze.synpuf_pde
# MAGIC UNION ALL
# MAGIC SELECT 'synpuf_inpatient', COUNT(*) FROM workspace.bronze.synpuf_inpatient
# MAGIC UNION ALL
# MAGIC SELECT 'synpuf_outpatient', COUNT(*) FROM workspace.bronze.synpuf_outpatient
# MAGIC UNION ALL
# MAGIC SELECT 'synpuf_carrier', COUNT(*) FROM workspace.bronze.synpuf_carrier
# MAGIC UNION ALL
# MAGIC SELECT 'partd_spending', COUNT(*) FROM workspace.bronze.partd_spending
# MAGIC ORDER BY table_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Source Documentation
# MAGIC
# MAGIC | Dataset | Source URL | Download Date | Notes |
# MAGIC |---------|-----------|---------------|-------|
# MAGIC | FDA NDC Product | https://open.fda.gov/data/ndc/ | 2026-03-04 | Tab-delimited |
# MAGIC | FDA NDC Package | https://open.fda.gov/data/ndc/ | 2026-03-04 | Tab-delimited |
# MAGIC | NADAC 2024 | https://data.medicaid.gov | 2026-03-04 | May need to filter to Q1 for compute |
# MAGIC | DE-SynPUF Sample 1 | https://www.cms.gov (SynPUF page) | 2026-03-04 | Sample 1 of 20 |
# MAGIC | Part D Spending | https://data.cms.gov | 2026-03-04 | Most recent available year |
# MAGIC
# MAGIC **Bronze layer complete.** Proceed to Notebook 2 for NDC normalization and the drug dimension.
