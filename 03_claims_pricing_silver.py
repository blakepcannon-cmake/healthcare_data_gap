# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 3: Claims & Pricing Transformation (Silver Layer)
# MAGIC
# MAGIC **Purpose:** Transform remaining bronze tables into clean, typed, analytics-ready silver tables.
# MAGIC Each table gets: proper data types, normalized keys, consistent naming, and embedded quality checks.
# MAGIC
# MAGIC **Tables built in this notebook:**
# MAGIC - `silver.fact_drug_pricing` (from NADAC)
# MAGIC - `silver.fact_prescription_events` (from SynPUF PDE)
# MAGIC - `silver.dim_beneficiary` (from SynPUF Beneficiary)
# MAGIC - `silver.fact_inpatient_claims` (from SynPUF Inpatient)
# MAGIC - `silver.fact_outpatient_claims` (from SynPUF Outpatient)
# MAGIC - `silver.fact_carrier_claims` (from SynPUF Carrier)
# MAGIC - `silver.agg_partd_spending` (from Part D Spending by Drug)

# COMMAND ----------

from pyspark.sql.functions import (
    col, to_date, trim, when, lit, regexp_replace, lpad, lower,
    count, sum as spark_sum, avg, min as spark_min, max as spark_max,
    year, month, datediff, expr, coalesce, concat_ws
)
from pyspark.sql.types import IntegerType, DoubleType, StringType, DateType

# Re-register the NDC normalization UDF (in case cluster restarted between notebooks)
def normalize_ndc_to_11(ndc_raw: str) -> str:
    if ndc_raw is None:
        return None
    cleaned = ndc_raw.strip().replace(" ", "")
    if not cleaned:
        return None
    if "-" in cleaned:
        parts = cleaned.split("-")
        if len(parts) != 3:
            return None
        labeler, product, package = parts
        seg_lengths = (len(labeler), len(product), len(package))
        if seg_lengths == (4, 4, 2):
            labeler = labeler.zfill(5)
        elif seg_lengths == (5, 3, 2):
            product = product.zfill(4)
        elif seg_lengths == (5, 4, 1):
            package = package.zfill(2)
        elif seg_lengths == (5, 4, 2):
            pass
        else:
            labeler = labeler.zfill(5)
            product = product.zfill(4)
            package = package.zfill(2)
        return f"{labeler}{product}{package}"
    if len(cleaned) == 11 and cleaned.isdigit():
        return cleaned
    if len(cleaned) == 10 and cleaned.isdigit():
        return cleaned[:9] + "0" + cleaned[9]
    if cleaned.isdigit() and len(cleaned) < 11:
        return cleaned.zfill(11)
    return None

from pyspark.sql.functions import udf
normalize_ndc_udf = udf(normalize_ndc_to_11, StringType())
spark.udf.register("normalize_ndc", normalize_ndc_to_11, StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper: Data Quality Report
# MAGIC
# MAGIC Reusable function that prints null rates and basic stats for any DataFrame.

# COMMAND ----------

def quality_report(df, table_name):
    """Print null rates and basic stats for a DataFrame."""
    total = df.count()
    print(f"\n{'='*60}")
    print(f"Quality Report: {table_name}")
    print(f"Total rows: {total:,}")
    print(f"{'='*60}")
    
    print(f"\n{'Column':<40} {'Null %':>8} {'Distinct':>10}")
    print("-" * 60)
    
    for c in df.columns:
        null_count = df.filter(col(c).isNull()).count()
        null_pct = (null_count / total * 100) if total > 0 else 0
        distinct = df.select(c).distinct().count()
        flag = " ***" if null_pct > 50 else ""
        print(f"  {c:<38} {null_pct:>7.1f}% {distinct:>10,}{flag}")
    
    return total

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. fact_drug_pricing (from NADAC)
# MAGIC
# MAGIC NADAC represents the national average of what pharmacies pay to acquire drugs.
# MAGIC This is the closest public data point to the distributor sell-through price
# MAGIC (where McKesson sits in the supply chain).

# COMMAND ----------

df_nadac = spark.table("workspace.bronze.nadac_pricing")

# Inspect column names (they vary slightly across download years)
print("NADAC columns:")
for c in df_nadac.columns:
    print(f"  {c}")

# COMMAND ----------

# DBTITLE 1,Transform NADAC pricing to silver
# Transform NADAC to silver
# Column names from the 2024 NADAC download; adjust if your download differs

fact_drug_pricing = (
    df_nadac
    .withColumn("ndc", normalize_ndc_udf(col("NDC").cast("string")))
    .filter(col("ndc").isNotNull())
    .select(
        col("ndc"),
        trim(col("NDC_Description")).alias("ndc_description"),
        col("NADAC_Per_Unit").cast(DoubleType()).alias("nadac_per_unit"),
        to_date(col("Effective_Date"), "MM/dd/yyyy").alias("effective_date"),
        trim(col("Pricing_Unit")).alias("pricing_unit"),
        trim(col("Pharmacy_Type_Indicator")).alias("pharmacy_type"),
        trim(col("OTC")).alias("otc_flag"),
        trim(col("Explanation_Code")).alias("explanation_code"),
        trim(col("Classification_for_Rate_Setting")).alias("rate_classification"),
        col("Corresponding_Generic_Drug_NADAC_Per_Unit")
            .cast(DoubleType()).alias("generic_nadac_per_unit"),
        to_date(col("Corresponding_Generic_Drug_Effective_Date"), "MM/dd/yyyy")
            .alias("generic_effective_date"),
    )
)

quality_report(fact_drug_pricing, "fact_drug_pricing")

# COMMAND ----------

(
    fact_drug_pricing.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.silver.fact_drug_pricing")
)

print("workspace.silver.fact_drug_pricing written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. fact_prescription_events (from SynPUF PDE)
# MAGIC
# MAGIC Each row is a single prescription fill for a Medicare beneficiary.
# MAGIC This is the claim-level data that connects the patient to the drug to the cost.
# MAGIC
# MAGIC SynPUF dates are stored as integer strings in YYYYMMDD format.

# COMMAND ----------

df_pde = spark.table("workspace.bronze.synpuf_pde")

print("PDE columns:")
for c in df_pde.columns:
    print(f"  {c}")

# COMMAND ----------

def parse_synpuf_date(col_ref):
    """Convert SynPUF integer date (YYYYMMDD) to proper date type."""
    return to_date(col_ref.cast(StringType()), "yyyyMMdd")

fact_prescription_events = (
    df_pde
    .withColumn("ndc", normalize_ndc_udf(col("PROD_SRVC_ID").cast(StringType())))
    .select(
        col("DESYNPUF_ID").alias("beneficiary_id"),
        col("PDE_ID").alias("pde_id"),
        parse_synpuf_date(col("SRVC_DT")).alias("service_date"),
        col("ndc"),
        col("PROD_SRVC_ID").cast(StringType()).alias("ndc_raw"),
        col("QTY_DSPNSD_NUM").cast(DoubleType()).alias("quantity_dispensed"),
        col("DAYS_SUPLY_NUM").cast(IntegerType()).alias("days_supply"),
        col("PTNT_PAY_AMT").cast(DoubleType()).alias("patient_pay_amount"),
        col("TOT_RX_CST_AMT").cast(DoubleType()).alias("total_rx_cost"),
    )
    .withColumn(
        "payer_paid_amount",
        col("total_rx_cost") - col("patient_pay_amount")
    )
    .withColumn("service_year", year(col("service_date")))
    .withColumn("service_month", month(col("service_date")))
)

quality_report(fact_prescription_events, "fact_prescription_events")

# COMMAND ----------

# Cost distribution sanity check
fact_prescription_events.select(
    spark_min("total_rx_cost").alias("min_cost"),
    avg("total_rx_cost").alias("avg_cost"),
    spark_max("total_rx_cost").alias("max_cost"),
    avg("patient_pay_amount").alias("avg_patient_pay"),
    avg("payer_paid_amount").alias("avg_payer_paid"),
    avg("days_supply").alias("avg_days_supply"),
).show(truncate=False)

# COMMAND ----------

(
    fact_prescription_events.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.silver.fact_prescription_events")
)

print("workspace.silver.fact_prescription_events written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. dim_beneficiary (from SynPUF Beneficiary)
# MAGIC
# MAGIC The patient dimension. Uses the most recent year's record for each beneficiary
# MAGIC to capture the latest chronic condition flags and demographics.
# MAGIC
# MAGIC Chronic condition codes: 1 = yes, 2 = no (counterintuitive but that's the SynPUF spec).

# COMMAND ----------

df_bene = spark.table("workspace.bronze.synpuf_beneficiary")

print("Beneficiary columns:")
for c in df_bene.columns:
    print(f"  {c}")

# COMMAND ----------

# MAGIC %skip
# MAGIC
# MAGIC /*
# MAGIC
# MAGIC A little looking around to figure out what my source data was. Went back to bronze layer to add in source year column; anotehr item that was cinluded on teh csv read source but not within the table creation to pullj from teh databricks marektplace.
# MAGIC
# MAGIC */
# MAGIC
# MAGIC
# MAGIC --SELECT * FROM databricks_cms_synthetic_public_use_files_synpuf.cms_synpuf_ext.car_claims limit 10
# MAGIC WITH DateFormat as (
# MAGIC
# MAGIC   SELECT
# MAGIC     SUBSTR(CLM_FROM_DT, 1, 4) || '-' ||
# MAGIC     SUBSTR(CLM_FROM_DT, 5, 2) || '-' ||
# MAGIC     SUBSTR(CLM_FROM_DT, 7, 2) AS DateFormatted,
# MAGIC     CLM_FROM_DT
# MAGIC   FROM databricks_cms_synthetic_public_use_files_synpuf.cms_synpuf_ext.car_claims 
# MAGIC )
# MAGIC SELECT DATE_PART('year', DateFormatted::DATE),count(*) FROM DateFormat group by  DATE_PART('year', DateFormatted::DATE) order by  DATE_PART('year', DateFormatted::DATE) desc
# MAGIC --SELECT * FROM dateformat limit 10
# MAGIC
# MAGIC
# MAGIC /*
# MAGIC SELECT EXTRACT(YEAR FROM TO_DATE(CLM_FROM_DT, 'YYYYMMDD')),COUNT(*) FROM databricks_cms_synthetic_public_use_files_synpuf.cms_synpuf_ext.car_claims 
# MAGIC group by EXTRACT(YEAR FROM TO_DATE(CLM_FROM_DT, 'YYYYMMDD')) order by EXTRACT(YEAR FROM TO_DATE(CLM_FROM_DT, 'YYYYMMDD')) desc
# MAGIC */
# MAGIC

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, desc

# Take the most recent year's record per beneficiary
window_latest = Window.partitionBy("DESYNPUF_ID").orderBy(desc("_source_year"))

# Chronic condition recode: 1=Yes, 2=No -> boolean-friendly 1/0
def recode_chronic(col_name, alias):
    return when(col(col_name) == 1, 1).otherwise(0).alias(alias)

dim_beneficiary = (
    df_bene
    .withColumn("_rank", row_number().over(window_latest))
    .filter(col("_rank") == 1)
    .select(
        col("DESYNPUF_ID").alias("beneficiary_id"),
        parse_synpuf_date(col("BENE_BIRTH_DT")).alias("birth_date"),
        parse_synpuf_date(col("BENE_DEATH_DT")).alias("death_date"),
        when(col("BENE_SEX_IDENT_CD") == 1, "Male")
            .when(col("BENE_SEX_IDENT_CD") == 2, "Female")
            .otherwise("Unknown").alias("sex"),
        col("BENE_RACE_CD").cast(IntegerType()).alias("race_code"),
        col("SP_STATE_CODE").cast(IntegerType()).alias("state_code"),
        col("BENE_COUNTY_CD").cast(IntegerType()).alias("county_code"),
        col("BENE_HI_CVRAGE_TOT_MONS").cast(IntegerType()).alias("part_a_coverage_months"),
        col("BENE_SMI_CVRAGE_TOT_MONS").cast(IntegerType()).alias("part_b_coverage_months"),
        col("PLAN_CVRG_MOS_NUM").cast(IntegerType()).alias("part_d_coverage_months"),
        
        # Chronic conditions (recoded to 1=yes, 0=no)
        recode_chronic("SP_ALZHDMTA", "cc_alzheimers"),
        recode_chronic("SP_CHF", "cc_heart_failure"),
        recode_chronic("SP_CHRNKIDN", "cc_chronic_kidney"),
        recode_chronic("SP_CNCR", "cc_cancer"),
        recode_chronic("SP_COPD", "cc_copd"),
        recode_chronic("SP_DEPRESSN", "cc_depression"),
        recode_chronic("SP_DIABETES", "cc_diabetes"),
        recode_chronic("SP_ISCHMCHT", "cc_ischemic_heart"),
        recode_chronic("SP_OSTEOPRS", "cc_osteoporosis"),
        recode_chronic("SP_RA_OA", "cc_arthritis"),
        recode_chronic("SP_STRKETIA", "cc_stroke_tia"),
        
        # Annual reimbursement totals
        col("MEDREIMB_IP").cast(DoubleType()).alias("inpatient_reimbursement"),
        col("MEDREIMB_OP").cast(DoubleType()).alias("outpatient_reimbursement"),
        col("MEDREIMB_CAR").cast(DoubleType()).alias("carrier_reimbursement"),
        col("BENRES_IP").cast(DoubleType()).alias("inpatient_beneficiary_resp"),
        col("BENRES_OP").cast(DoubleType()).alias("outpatient_beneficiary_resp"),
        col("BENRES_CAR").cast(DoubleType()).alias("carrier_beneficiary_resp"),
        
        col("_source_year").cast(IntegerType()).alias("record_year"),
    )
    .withColumn(
        "chronic_condition_count",
        col("cc_alzheimers") + col("cc_heart_failure") + col("cc_chronic_kidney") +
        col("cc_cancer") + col("cc_copd") + col("cc_depression") + col("cc_diabetes") +
        col("cc_ischemic_heart") + col("cc_osteoporosis") + col("cc_arthritis") +
        col("cc_stroke_tia")
    )
    .withColumn(
        "is_deceased",
        when(col("death_date").isNotNull(), 1).otherwise(0)
    )
)

quality_report(dim_beneficiary, "dim_beneficiary")

# COMMAND ----------

# Chronic condition prevalence
print("\nChronic Condition Prevalence:")
print("-" * 40)
total_benes = dim_beneficiary.count()
for cc in [c for c in dim_beneficiary.columns if c.startswith("cc_")]:
    pos = dim_beneficiary.filter(col(cc) == 1).count()
    print(f"  {cc:<30} {pos/total_benes*100:>5.1f}%")

# COMMAND ----------

(
    dim_beneficiary.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.silver.dim_beneficiary")
)

print("workspace.silver.dim_beneficiary written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. fact_inpatient_claims
# MAGIC
# MAGIC Hospital admissions. Key for downstream analysis: do patients on expensive 
# MAGIC drug regimens have lower hospitalization rates?

# COMMAND ----------

df_ip = spark.table("workspace.bronze.synpuf_inpatient")

fact_inpatient = (
    df_ip
    .select(
        col("DESYNPUF_ID").alias("beneficiary_id"),
        col("CLM_ID").alias("claim_id"),
        parse_synpuf_date(col("CLM_FROM_DT")).alias("claim_start_date"),
        parse_synpuf_date(col("CLM_THRU_DT")).alias("claim_end_date"),
        col("PRVDR_NUM").alias("provider_id"),
        col("CLM_PMT_AMT").cast(DoubleType()).alias("claim_payment_amount"),
        col("NCH_PRMRY_PYR_CLM_PD_AMT").cast(DoubleType()).alias("primary_payer_paid"),
        parse_synpuf_date(col("CLM_ADMSN_DT")).alias("admission_date"),
        parse_synpuf_date(col("NCH_BENE_DSCHRG_DT")).alias("discharge_date"),
        col("CLM_UTLZTN_DAY_CNT").cast(IntegerType()).alias("utilization_days"),
        col("CLM_DRG_CD").alias("drg_code"),
        col("ADMTNG_ICD9_DGNS_CD").alias("admitting_diagnosis"),
        col("NCH_BENE_IP_DDCTBL_AMT").cast(DoubleType()).alias("deductible_amount"),
        col("NCH_BENE_PTA_COINSRNC_LBLTY_AM").cast(DoubleType()).alias("coinsurance_amount"),
        # Diagnosis codes (up to 10)
        col("ICD9_DGNS_CD_1").alias("diag_1"),
        col("ICD9_DGNS_CD_2").alias("diag_2"),
        col("ICD9_DGNS_CD_3").alias("diag_3"),
    )
    .withColumn("service_year", year(col("claim_start_date")))
    .withColumn(
        "length_of_stay",
        when(
            col("admission_date").isNotNull() & col("discharge_date").isNotNull(),
            datediff(col("discharge_date"), col("admission_date"))
        )
    )
)

row_ct = quality_report(fact_inpatient, "fact_inpatient_claims")

(
    fact_inpatient.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.silver.fact_inpatient_claims")
)

print(f"\nworkspace.silver.fact_inpatient_claims written ({row_ct:,} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. fact_outpatient_claims

# COMMAND ----------

df_op = spark.table("workspace.bronze.synpuf_outpatient")

fact_outpatient = (
    df_op
    .select(
        col("DESYNPUF_ID").alias("beneficiary_id"),
        col("CLM_ID").alias("claim_id"),
        parse_synpuf_date(col("CLM_FROM_DT")).alias("claim_start_date"),
        parse_synpuf_date(col("CLM_THRU_DT")).alias("claim_end_date"),
        col("PRVDR_NUM").alias("provider_id"),
        col("CLM_PMT_AMT").cast(DoubleType()).alias("claim_payment_amount"),
        col("NCH_PRMRY_PYR_CLM_PD_AMT").cast(DoubleType()).alias("primary_payer_paid"),
        col("NCH_BENE_PTB_DDCTBL_AMT").cast(DoubleType()).alias("deductible_amount"),
        col("NCH_BENE_BLOOD_DDCTBL_LBLTY_AM").cast(DoubleType()).alias("blood_deductible"),
        col("ICD9_DGNS_CD_1").alias("diag_1"),
        col("ICD9_DGNS_CD_2").alias("diag_2"),
        col("ICD9_DGNS_CD_3").alias("diag_3"),
        col("HCPCS_CD_1").alias("hcpcs_1"),
    )
    .withColumn("service_year", year(col("claim_start_date")))
)

row_ct = quality_report(fact_outpatient, "fact_outpatient_claims")

(
    fact_outpatient.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.silver.fact_outpatient_claims")
)

print(f"\nworkspace.silver.fact_outpatient_claims written ({row_ct:,} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. fact_carrier_claims
# MAGIC
# MAGIC Physician/supplier claims. These are the professional services claims 
# MAGIC (office visits, specialist encounters, procedures).

# COMMAND ----------

df_car = spark.table("workspace.bronze.synpuf_carrier")

fact_carrier = (
    df_car
    .select(
        col("DESYNPUF_ID").alias("beneficiary_id"),
        col("CLM_ID").alias("claim_id"),
        parse_synpuf_date(col("CLM_FROM_DT")).alias("claim_start_date"),
        parse_synpuf_date(col("CLM_THRU_DT")).alias("claim_end_date"),

        ###
        # Assumptions below since lack of data; these columns don't appear to be available on carrier claims

        #col("PRVDR_NUM").alias("provider_id"),
        col("PRF_PHYSN_NPI_1").alias("provider_id"),


        #col("CLM_PMT_AMT").cast(DoubleType()).alias("claim_payment_amount"),
        col("LINE_NCH_PMT_AMT_1").cast(DoubleType()).alias("claim_payment_amount"),


        #col("NCH_PRMRY_PYR_CLM_PD_AMT").cast(DoubleType()).alias("primary_payer_paid"),
        col("LINE_NCH_PMT_AMT_1").cast(DoubleType()).alias("primary_payer_paid"),

        #col("TAX_NUM").alias("provider_tax_id"),
        col("TAX_NUM_1").alias("provider_tax_id"),
        ### 

        col("LINE_NCH_PMT_AMT_1").cast(DoubleType()).alias("line_1_payment"),
        col("LINE_BENE_PTB_DDCTBL_AMT_1").cast(DoubleType()).alias("line_1_deductible"),
        col("LINE_ALOWD_CHRG_AMT_1").cast(DoubleType()).alias("line_1_allowed_amount"),
        col("LINE_COINSRNC_AMT_1").cast(DoubleType()).alias("line_1_coinsurance"),
        col("HCPCS_CD_1").alias("hcpcs_1"),
        col("LINE_ICD9_DGNS_CD_1").alias("line_1_diagnosis"),
    )
    .withColumn("service_year", year(col("claim_start_date")))
)

row_ct = quality_report(fact_carrier, "fact_carrier_claims")

(
    fact_carrier.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.silver.fact_carrier_claims")
)

print(f"\nworkspace.silver.fact_carrier_claims written ({row_ct:,} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. agg_partd_spending
# MAGIC
# MAGIC Aggregate Part D drug spending data. This provides the macro view 
# MAGIC (total program-level spending by drug) that complements the claim-level PDE data.
# MAGIC
# MAGIC Column names vary by download year. Inspect and adjust as needed.

# COMMAND ----------

df_partd = spark.table("workspace.bronze.partd_spending")

print("Part D Spending columns:")
for c in df_partd.columns:
    print(f"  {c}")

# COMMAND ----------

# The Part D file column names vary by year. Below is a general mapping.
# Adjust the column names to match your specific download.
# Common columns across years:
#   Brand Name, Generic Name, Total Spending, Total Dosage Units,
#   Total Claims, Total Beneficiaries, Average Cost Per Dosage Unit,
#   Average Spending Per Claim, Average Spending Per Beneficiary

# Attempt a flexible select -- pick columns that exist
available_cols = [c.lower().replace(" ", "_") for c in df_partd.columns]

agg_partd = df_partd

# Rename all columns to snake_case for consistency
for c in agg_partd.columns:
    agg_partd = agg_partd.withColumnRenamed(c, c.lower().replace(" ", "_").replace("-", "_"))

print("Renamed columns:")
for c in agg_partd.columns:
    print(f"  {c}")

# COMMAND ----------

(
    agg_partd.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.silver.agg_partd_spending")
)

print("workspace.silver.agg_partd_spending written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Layer Inventory

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'dim_drug' AS table_name, COUNT(*) AS rows FROM workspace.silver.dim_drug
# MAGIC UNION ALL SELECT 'dim_beneficiary', COUNT(*) FROM workspace.silver.dim_beneficiary
# MAGIC UNION ALL SELECT 'fact_drug_pricing', COUNT(*) FROM workspace.silver.fact_drug_pricing
# MAGIC UNION ALL SELECT 'fact_prescription_events', COUNT(*) FROM workspace.silver.fact_prescription_events
# MAGIC UNION ALL SELECT 'fact_inpatient_claims', COUNT(*) FROM workspace.silver.fact_inpatient_claims
# MAGIC UNION ALL SELECT 'fact_outpatient_claims', COUNT(*) FROM workspace.silver.fact_outpatient_claims
# MAGIC UNION ALL SELECT 'fact_carrier_claims', COUNT(*) FROM workspace.silver.fact_carrier_claims
# MAGIC UNION ALL SELECT 'agg_partd_spending', COUNT(*) FROM workspace.silver.agg_partd_spending
# MAGIC ORDER BY table_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **Silver layer complete.** All source data is now:
# MAGIC - Properly typed (dates are dates, numbers are numbers)
# MAGIC - Keyed on normalized 11-digit NDC where applicable
# MAGIC - Named with consistent snake_case conventions
# MAGIC - Quality-checked with null rates and distribution stats
# MAGIC
# MAGIC **Key schema relationships:**
# MAGIC - `dim_drug.ndc` joins to `fact_drug_pricing.ndc` and `fact_prescription_events.ndc`
# MAGIC - `dim_beneficiary.beneficiary_id` joins to all fact tables on `beneficiary_id`
# MAGIC - `agg_partd_spending` joins to `dim_drug` via drug name (no NDC in aggregate file)
# MAGIC
# MAGIC **Next:** Notebook 4 builds the gold-layer analytical models.
