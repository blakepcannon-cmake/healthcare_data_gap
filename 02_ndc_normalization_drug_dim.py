# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 2: NDC Normalization & Drug Dimension (Silver Layer)
# MAGIC
# MAGIC **Purpose:** Build a clean, normalized drug dimension table that serves as the backbone 
# MAGIC for joining all downstream datasets.
# MAGIC
# MAGIC **The core problem this notebook solves:**  
# MAGIC The National Drug Code (NDC) is the universal drug identifier, but it appears in different 
# MAGIC formats across every system in the healthcare ecosystem. The FDA stores it as a hyphenated 
# MAGIC 3-segment code (e.g., `0002-1433-01`). CMS claims data stores it as an 11-digit padded string 
# MAGIC (e.g., `00002143301`). NADAC uses yet another variant. If you cannot normalize these reliably, 
# MAGIC you cannot join across ecosystem nodes.
# MAGIC
# MAGIC **NDC Format Background:**  
# MAGIC The NDC has three segments: Labeler (manufacturer), Product, and Package.  
# MAGIC The FDA allows three configurations that all sum to 10 digits:
# MAGIC - **4-4-2:** 4-digit labeler, 4-digit product, 2-digit package
# MAGIC - **5-3-2:** 5-digit labeler, 3-digit product, 2-digit package
# MAGIC - **5-4-1:** 5-digit labeler, 4-digit product, 1-digit package
# MAGIC
# MAGIC The 11-digit billing format (used by CMS, NADAC, pharmacies) standardizes to **5-4-2** by 
# MAGIC zero-padding the short segment:
# MAGIC - 4-4-2 becomes 04444-4444-22 (pad labeler)
# MAGIC - 5-3-2 becomes 55555-0333-22 (pad product)
# MAGIC - 5-4-1 becomes 55555-4444-01 (pad package)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.silver;

# COMMAND ----------

# MAGIC %md
# MAGIC ## NDC Normalization Function
# MAGIC
# MAGIC This is the single most important piece of reusable logic in the project.
# MAGIC It handles conversion from any FDA-format NDC to the 11-digit billing format.

# COMMAND ----------

from pyspark.sql.functions import (
    col, udf, when, length, lpad, regexp_replace, trim, concat, lit,
    split, size, coalesce
)
from pyspark.sql.types import StringType


def normalize_ndc_to_11(ndc_raw: str) -> str:
    """
    Convert any NDC format to the 11-digit billing format (5-4-2).
    
    Handles:
    - Hyphenated FDA format: "0002-1433-01" (any of 4-4-2, 5-3-2, 5-4-1)
    - Already 11-digit: "00002143301" (passthrough)
    - 10-digit no hyphens: attempts 5-4-1 assumption (most common)
    - Whitespace and leading/trailing junk
    
    Returns None if input is null or unparseable.
    """
    if ndc_raw is None:
        return None
    
    cleaned = ndc_raw.strip().replace(" ", "")
    
    if not cleaned:
        return None
    
    # Case 1: Hyphenated format (most reliable, segments are explicit)
    if "-" in cleaned:
        parts = cleaned.split("-")
        if len(parts) != 3:
            return None
        
        labeler, product, package = parts
        
        # Determine which 10-digit config this is and pad to 5-4-2
        seg_lengths = (len(labeler), len(product), len(package))
        
        if seg_lengths == (4, 4, 2):
            labeler = labeler.zfill(5)
        elif seg_lengths == (5, 3, 2):
            product = product.zfill(4)
        elif seg_lengths == (5, 4, 1):
            package = package.zfill(2)
        elif seg_lengths == (5, 4, 2):
            pass  # Already in 11-digit format with hyphens
        else:
            # Non-standard segment lengths; attempt best-effort padding
            labeler = labeler.zfill(5)
            product = product.zfill(4)
            package = package.zfill(2)
        
        return f"{labeler}{product}{package}"
    
    # Case 2: No hyphens, already 11 digits
    if len(cleaned) == 11 and cleaned.isdigit():
        return cleaned
    
    # Case 3: No hyphens, 10 digits -- ambiguous, but 5-4-1 is most common
    # Pad the package segment (last digit) to get 5-4-2
    if len(cleaned) == 10 and cleaned.isdigit():
        return cleaned[:9] + "0" + cleaned[9]
    
    # Case 4: Fewer than 10 digits -- left-pad entire string to 11
    if cleaned.isdigit() and len(cleaned) < 11:
        return cleaned.zfill(11)
    
    return None


# Register as UDF for use in Spark SQL and DataFrames
normalize_ndc_udf = udf(normalize_ndc_to_11, StringType())
spark.udf.register("normalize_ndc", normalize_ndc_to_11, StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unit Tests for NDC Normalization
# MAGIC
# MAGIC Before building anything on top of this function, verify it handles known edge cases.

# COMMAND ----------

# Test cases: (input, expected_output)
test_cases = [
    # Standard hyphenated formats
    ("0002-1433-01", "00002143301"),   # 4-4-2 -> pad labeler
    ("55513-101-01", "55513010101"),   # 5-3-2 -> pad product
    ("60429-1127-1", "60429112701"),   # 5-4-1 -> pad package
    ("12345-6789-01", "12345678901"),  # 5-4-2 -> already correct
    
    # No hyphens
    ("00002143301", "00002143301"),    # Already 11 digits
    ("0002143301", "00021433001"),     # 10 digits, assume 5-4-1 padding
    
    # Edge cases
    (None, None),
    ("", None),
    ("  0002-1433-01  ", "00002143301"),  # Whitespace
]

print(f"{'Input':<25} {'Expected':<15} {'Got':<15} {'Pass?'}")
print("-" * 70)

all_passed = True
for raw_input, expected in test_cases:
    result = normalize_ndc_to_11(raw_input)
    passed = result == expected
    if not passed:
        all_passed = False
    display_input = repr(raw_input)
    print(f"{display_input:<25} {str(expected):<15} {str(result):<15} {'OK' if passed else 'FAIL'}")

print()
print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED -- review before proceeding")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build dim_drug from FDA NDC Directory
# MAGIC
# MAGIC Join product and package files on PRODUCTNDC, normalize the full NDC 
# MAGIC (NDCPACKAGECODE) to 11-digit format, and select the fields needed downstream.

# COMMAND ----------

df_product = spark.table("workspace.bronze.fda_ndc_product")
df_package = spark.table("workspace.bronze.fda_ndc_package")

# Inspect the join key
print("Product PRODUCTNDC samples:")
df_product.select("PRODUCTNDC").show(5, truncate=False)

print("Package PRODUCTNDC samples:")
df_package.select("PRODUCTNDC", "NDCPACKAGECODE").show(5, truncate=False)

# COMMAND ----------

# Join product + package on PRODUCTNDC
# One product can have multiple packages (different counts/sizes of the same drug)
# The grain of dim_drug is one row per unique 11-digit NDC (package level)

df_drug_raw = (
    df_package
    .join(df_product, on="PRODUCTNDC", how="left")
    .withColumn("ndc_11", normalize_ndc_udf(col("NDCPACKAGECODE")))
)

# Check how many NDCs we successfully normalized
total = df_drug_raw.count()
normalized = df_drug_raw.filter(col("ndc_11").isNotNull()).count()
failed = total - normalized

print(f"Total package rows:       {total:,}")
print(f"Successfully normalized:  {normalized:,} ({normalized/total*100:.1f}%)")
print(f"Failed normalization:     {failed:,} ({failed/total*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select and rename fields for the dimension table

# COMMAND ----------

dim_drug = (
    df_drug_raw
    .filter(col("ndc_11").isNotNull())
    .select(
        col("ndc_11").alias("ndc"),
        col("PRODUCTNDC").alias("product_ndc"),
        col("NDCPACKAGECODE").alias("ndc_package_code_raw"),
        col("PROPRIETARYNAME").alias("brand_name"),
        col("NONPROPRIETARYNAME").alias("generic_name"),
        col("LABELERNAME").alias("labeler_name"),
        col("DOSAGEFORMNAME").alias("dosage_form"),
        col("ROUTENAME").alias("route"),
        col("SUBSTANCENAME").alias("active_ingredients"),
        col("ACTIVE_NUMERATOR_STRENGTH").alias("strength"),
        col("ACTIVE_INGRED_UNIT").alias("strength_unit"),
        col("PHARM_CLASSES").alias("pharm_classes"),
        col("DEASCHEDULE").alias("dea_schedule"),
        col("PRODUCTTYPENAME").alias("product_type"),
        col("PACKAGEDESCRIPTION").alias("package_description"),
    )
    .dropDuplicates(["ndc"])  # Ensure one row per 11-digit NDC
)

print(f"dim_drug final row count: {dim_drug.count():,}")
dim_drug.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Quality Checks on dim_drug

# COMMAND ----------

from pyspark.sql.functions import count, sum as spark_sum, when as spark_when

# Null rates by column
total_rows = dim_drug.count()

null_checks = dim_drug.select(
    *[
        (spark_sum(spark_when(col(c).isNull(), 1).otherwise(0)) / total_rows * 100)
        .alias(c)
        for c in dim_drug.columns
    ]
)

print("Null rates (%) for dim_drug:")
print("-" * 50)
null_row = null_checks.collect()[0]
for c in dim_drug.columns:
    pct = null_row[c]
    flag = " <-- HIGH" if pct > 20 else ""
    print(f"  {c:<35} {pct:>6.1f}%{flag}")

# COMMAND ----------

# Check for duplicate NDCs (should be zero after dropDuplicates)
dup_check = (
    dim_drug
    .groupBy("ndc")
    .count()
    .filter(col("count") > 1)
    .count()
)
print(f"Duplicate NDCs in dim_drug: {dup_check}")

# COMMAND ----------

# DBTITLE 1,Sample output
# Sample output
display(dim_drug.show(10, truncate=40))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write dim_drug to Silver

# COMMAND ----------

(
    dim_drug.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.silver.dim_drug")
)

print("workspace.silver.dim_drug written successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate NDC Coverage Against Downstream Datasets
# MAGIC
# MAGIC Before moving on, check what percentage of NDCs in the PDE and NADAC files 
# MAGIC can be resolved against dim_drug. Low match rates signal a data quality issue 
# MAGIC that would need to be addressed before building gold-layer products.

# COMMAND ----------

# PDE coverage check
df_pde = spark.table("workspace.bronze.synpuf_pde")

pde_ndcs = (
    df_pde
    .select(col("PROD_SRVC_ID").alias("raw_ndc"))
    .distinct()
    .withColumn("ndc_11", normalize_ndc_udf(col("raw_ndc")))
)

pde_total = pde_ndcs.count()
pde_matched = (
    pde_ndcs
    .join(dim_drug.select("ndc"), pde_ndcs.ndc_11 == dim_drug.ndc, "inner")
    .count()
)

print(f"PDE unique NDCs:           {pde_total:,}")
print(f"Matched to dim_drug:       {pde_matched:,} ({pde_matched/pde_total*100:.1f}%)")
print(f"Unmatched:                 {pde_total - pde_matched:,}")
print()
print("Note: The SynPUF uses synthetic NDCs, so match rates will be lower")
print("than you'd see with real claims data. The architecture is what matters here.")

# COMMAND ----------

# DBTITLE 1,Cell 20
# NADAC coverage check

df_nadac = spark.table("workspace.bronze.nadac_pricing")

nadac_ndcs = (
    df_nadac
    .select(col("NDC").alias("raw_ndc"))
    .distinct()
    .withColumn(
        "ndc_11",
        normalize_ndc_udf(col("raw_ndc").cast("string"))  # Cast to string for UDF
    )
)

nadac_total = nadac_ndcs.count()
nadac_matched = (
    nadac_ndcs
    .join(dim_drug.select("ndc"), nadac_ndcs.ndc_11 == dim_drug.ndc, "inner")
    .count()
)

print(f"NADAC unique NDCs:         {nadac_total:,}")
print(f"Matched to dim_drug:       {nadac_matched:,} ({nadac_matched/nadac_total*100:.1f}%)")
print(f"Unmatched:                 {nadac_total - nadac_matched:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What this notebook produced:**
# MAGIC - `workspace.silver.dim_drug`: One row per unique 11-digit NDC with product details
# MAGIC - A reusable `normalize_ndc` UDF registered for both Python and SQL use
# MAGIC - Coverage validation against PDE and NADAC source data
# MAGIC
# MAGIC **What this demonstrates:**
# MAGIC - Deep familiarity with the NDC format problem (a real pain point in healthcare data)
# MAGIC - Defensive coding: unit tests, null handling, quality checks before downstream use
# MAGIC - The dim_drug table is the bridge that connects manufacturer data (FDA) to 
# MAGIC   pharmacy data (NADAC) to claims data (SynPUF)
# MAGIC
# MAGIC **Next:** Notebook 3 transforms the remaining bronze tables into silver.
