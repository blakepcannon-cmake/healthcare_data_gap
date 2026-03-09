# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 4: Gold Layer Models
# MAGIC 
# MAGIC **Purpose:** Build business-ready analytical tables by joining silver tables across 
# MAGIC ecosystem domains. These gold tables are what a product team would build features on.
# MAGIC 
# MAGIC **Gold tables:**
# MAGIC 1. `gold.drug_cost_journey` -- end-to-end cost per drug (acquisition to reimbursement)
# MAGIC 2. `gold.beneficiary_drug_utilization` -- patient-level Rx patterns and adherence signals
# MAGIC 3. `gold.drug_cost_spread_analysis` -- margin analysis by drug over time
# MAGIC 4. `gold.high_cost_drug_cohort` -- high-spend patients and their downstream utilization
# MAGIC 
# MAGIC Each table crosses at least two ecosystem boundaries, demonstrating the cross-domain 
# MAGIC data fluency that Amy's team needs.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.gold;

# COMMAND ----------

from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg, min as spark_min,
    max as spark_max, when, datediff, lag, lead, row_number, desc,
    percentile_approx, year, month, round as spark_round, lit, expr,
    first, collect_list, size, array_distinct
)
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. drug_cost_journey
# MAGIC 
# MAGIC **What it answers:** For a given drug, what does the pharmacy pay to acquire it (NADAC), 
# MAGIC what does the claim say it costs (SynPUF), what does the patient pay, and what does the 
# MAGIC payer reimburse? The spread between these numbers is where the economics of drug 
# MAGIC distribution live.
# MAGIC 
# MAGIC **Ecosystem nodes joined:** Manufacturer (dim_drug) + Pharmacy (NADAC) + Payer/Patient (PDE)
# MAGIC 
# MAGIC **Important caveat:** The SynPUF uses synthetic NDCs, so the join to NADAC and dim_drug 
# MAGIC will not be 1:1. This is a known limitation of the synthetic data. In production with 
# MAGIC real claims, this join would be tight. The architecture and logic are what matter here.

# COMMAND ----------

# Load silver tables
dim_drug = spark.table("workspace.silver.dim_drug")
fact_pricing = spark.table("workspace.silver.fact_drug_pricing")
fact_rx = spark.table("workspace.silver.fact_prescription_events")

# For NADAC: get the most recent price per NDC 
# (NADAC updates weekly; we want the latest effective price for each drug)
pricing_window = Window.partitionBy("ndc").orderBy(desc("effective_date"))

latest_pricing = (
    fact_pricing
    .withColumn("_rank", row_number().over(pricing_window))
    .filter(col("_rank") == 1)
    .drop("_rank")
    .select(
        col("ndc"),
        col("nadac_per_unit"),
        col("pricing_unit"),
        col("pharmacy_type"),
        col("rate_classification"),
        col("effective_date").alias("nadac_effective_date"),
    )
)

print(f"Unique NDCs in latest NADAC pricing: {latest_pricing.count():,}")
print(f"Unique NDCs in PDE claims:           {fact_rx.select('ndc').distinct().count():,}")

# COMMAND ----------

# Aggregate PDE to the drug level first (one row per NDC)
rx_by_drug = (
    fact_rx
    .groupBy("ndc")
    .agg(
        count("*").alias("total_claims"),
        countDistinct("beneficiary_id").alias("unique_patients"),
        spark_sum("total_rx_cost").alias("total_gross_cost"),
        spark_sum("patient_pay_amount").alias("total_patient_pay"),
        spark_sum("payer_paid_amount").alias("total_payer_paid"),
        avg("total_rx_cost").alias("avg_claim_cost"),
        avg("patient_pay_amount").alias("avg_patient_pay"),
        avg("quantity_dispensed").alias("avg_quantity"),
        avg("days_supply").alias("avg_days_supply"),
    )
)

# Join: drug dimension + NADAC pricing + claims aggregates
drug_cost_journey = (
    rx_by_drug
    .join(dim_drug.select(
        "ndc", "brand_name", "generic_name", "labeler_name",
        "dosage_form", "route", "dea_schedule", "product_type"
    ), on="ndc", how="left")
    .join(latest_pricing, on="ndc", how="left")
    .withColumn(
        "cost_per_unit_on_claim",
        when(col("avg_quantity") > 0, col("avg_claim_cost") / col("avg_quantity"))
    )
    .withColumn(
        "acquisition_to_claim_spread",
        when(
            col("nadac_per_unit").isNotNull() & col("cost_per_unit_on_claim").isNotNull(),
            col("cost_per_unit_on_claim") - col("nadac_per_unit")
        )
    )
    .withColumn(
        "patient_cost_share_pct",
        when(col("total_gross_cost") > 0,
             spark_round(col("total_patient_pay") / col("total_gross_cost") * 100, 1))
    )
    .select(
        "ndc",
        "brand_name",
        "generic_name",
        "labeler_name",
        "dosage_form",
        "route",
        "product_type",
        "rate_classification",
        "total_claims",
        "unique_patients",
        "nadac_per_unit",
        "cost_per_unit_on_claim",
        "acquisition_to_claim_spread",
        "avg_claim_cost",
        "avg_patient_pay",
        "patient_cost_share_pct",
        "total_gross_cost",
        "total_patient_pay",
        "total_payer_paid",
        "avg_days_supply",
    )
)

drug_cost_journey.cache()
print(f"drug_cost_journey: {drug_cost_journey.count():,} rows")

# How many have both NADAC and claims data? (the cross-domain join)
both = drug_cost_journey.filter(
    col("nadac_per_unit").isNotNull() & col("total_claims").isNotNull()
).count()
print(f"Rows with both NADAC and claims data: {both:,}")

# COMMAND ----------

# Preview the top drugs by total spend
drug_cost_journey.orderBy(desc("total_gross_cost")).show(15, truncate=30)

# COMMAND ----------

(
    drug_cost_journey.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.gold.drug_cost_journey")
)

print("workspace.gold.drug_cost_journey written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. beneficiary_drug_utilization
# MAGIC 
# MAGIC **What it answers:** For each patient, what is their prescription pattern? 
# MAGIC How many drugs, how consistently are they filling, what is their total spend, 
# MAGIC and what chronic conditions do they carry?
# MAGIC 
# MAGIC **Ecosystem nodes joined:** Patient demographics (beneficiary) + Claims (PDE)
# MAGIC 
# MAGIC This table is the foundation for a medication adherence product.

# COMMAND ----------

dim_bene = spark.table("workspace.silver.dim_beneficiary")

# Per-beneficiary prescription utilization summary
bene_rx_summary = (
    fact_rx
    .groupBy("beneficiary_id")
    .agg(
        count("*").alias("total_rx_fills"),
        countDistinct("ndc").alias("unique_drugs"),
        spark_sum("total_rx_cost").alias("total_rx_spend"),
        spark_sum("patient_pay_amount").alias("total_patient_pay"),
        spark_sum("payer_paid_amount").alias("total_payer_paid"),
        avg("days_supply").alias("avg_days_supply"),
        spark_sum("days_supply").alias("total_days_supply"),
        spark_min("service_date").alias("first_fill_date"),
        spark_max("service_date").alias("last_fill_date"),
        countDistinct("service_year").alias("active_years"),
    )
)

# Calculate crude adherence proxy: 
# total days supply / calendar days between first and last fill
# Values near 1.0 suggest consistent filling; low values suggest gaps
beneficiary_drug_utilization = (
    bene_rx_summary
    .join(dim_bene, on="beneficiary_id", how="inner")
    .withColumn(
        "observation_days",
        datediff(col("last_fill_date"), col("first_fill_date"))
    )
    .withColumn(
        "days_supply_coverage_ratio",
        when(
            col("observation_days") > 0,
            spark_round(col("total_days_supply") / col("observation_days"), 2)
        )
    )
    .select(
        "beneficiary_id",
        "sex",
        "race_code",
        "state_code",
        "chronic_condition_count",
        "cc_diabetes",
        "cc_heart_failure",
        "cc_copd",
        "cc_depression",
        "cc_cancer",
        "is_deceased",
        "total_rx_fills",
        "unique_drugs",
        "total_rx_spend",
        "total_patient_pay",
        "avg_days_supply",
        "total_days_supply",
        "observation_days",
        "days_supply_coverage_ratio",
        "first_fill_date",
        "last_fill_date",
        "active_years",
        # Medical spend from beneficiary summary (non-Rx)
        "inpatient_reimbursement",
        "outpatient_reimbursement",
        "carrier_reimbursement",
    )
    .withColumn(
        "total_medical_spend",
        coalesce(col("inpatient_reimbursement"), lit(0)) +
        coalesce(col("outpatient_reimbursement"), lit(0)) +
        coalesce(col("carrier_reimbursement"), lit(0))
    )
    .withColumn(
        "rx_share_of_total_spend",
        when(
            (col("total_rx_spend") + col("total_medical_spend")) > 0,
            spark_round(
                col("total_rx_spend") / 
                (col("total_rx_spend") + col("total_medical_spend")) * 100,
                1
            )
        )
    )
)

from pyspark.sql.functions import coalesce

beneficiary_drug_utilization.cache()
print(f"beneficiary_drug_utilization: {beneficiary_drug_utilization.count():,} rows")

# COMMAND ----------

# Distribution of key metrics
beneficiary_drug_utilization.select(
    avg("total_rx_fills").alias("avg_fills"),
    avg("unique_drugs").alias("avg_drugs"),
    avg("total_rx_spend").alias("avg_rx_spend"),
    avg("days_supply_coverage_ratio").alias("avg_coverage_ratio"),
    avg("chronic_condition_count").alias("avg_chronic_conditions"),
    avg("rx_share_of_total_spend").alias("avg_rx_pct_of_spend"),
).show(truncate=False)

# COMMAND ----------

(
    beneficiary_drug_utilization.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.gold.beneficiary_drug_utilization")
)

print("workspace.gold.beneficiary_drug_utilization written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. drug_cost_spread_analysis
# MAGIC 
# MAGIC **What it answers:** For each drug, how has the gap between acquisition cost 
# MAGIC and reimbursement behaved over time? Narrowing spreads = margin compression. 
# MAGIC Volatile spreads = pricing instability worth monitoring.
# MAGIC 
# MAGIC **Ecosystem nodes joined:** Pharmacy acquisition (NADAC over time) + Payer (PDE claims)
# MAGIC 
# MAGIC This is the foundation for a drug pricing intelligence product.

# COMMAND ----------

# NADAC monthly trend: aggregate to monthly granularity per NDC
monthly_nadac = (
    fact_pricing
    .filter(col("effective_date").isNotNull())
    .withColumn("price_year", year(col("effective_date")))
    .withColumn("price_month", month(col("effective_date")))
    .groupBy("ndc", "price_year", "price_month")
    .agg(
        avg("nadac_per_unit").alias("avg_nadac_per_unit"),
        spark_min("nadac_per_unit").alias("min_nadac_per_unit"),
        spark_max("nadac_per_unit").alias("max_nadac_per_unit"),
        count("*").alias("price_observations"),
        first("rate_classification").alias("rate_classification"),
    )
    .withColumn(
        "monthly_price_range",
        col("max_nadac_per_unit") - col("min_nadac_per_unit")
    )
)

# Claims monthly trend: aggregate PDE to monthly per NDC
monthly_claims = (
    fact_rx
    .filter(col("service_date").isNotNull())
    .groupBy("ndc", "service_year", "service_month")
    .agg(
        count("*").alias("claim_count"),
        avg("total_rx_cost").alias("avg_claim_cost"),
        avg("patient_pay_amount").alias("avg_patient_pay"),
        spark_sum("quantity_dispensed").alias("total_units"),
    )
    .withColumn(
        "avg_cost_per_unit",
        when(col("total_units") > 0, col("avg_claim_cost") / (col("total_units") / col("claim_count")))
    )
)

# Join NADAC monthly to claims monthly
drug_cost_spread = (
    monthly_nadac
    .join(
        monthly_claims,
        (monthly_nadac.ndc == monthly_claims.ndc) &
        (monthly_nadac.price_year == monthly_claims.service_year) &
        (monthly_nadac.price_month == monthly_claims.service_month),
        how="full"
    )
    .select(
        coalesce(monthly_nadac.ndc, monthly_claims.ndc).alias("ndc"),
        coalesce(monthly_nadac.price_year, monthly_claims.service_year).alias("year"),
        coalesce(monthly_nadac.price_month, monthly_claims.service_month).alias("month"),
        monthly_nadac.avg_nadac_per_unit,
        monthly_nadac.rate_classification,
        monthly_nadac.monthly_price_range,
        monthly_claims.claim_count,
        monthly_claims.avg_claim_cost,
        monthly_claims.avg_patient_pay,
        monthly_claims.avg_cost_per_unit.alias("avg_claim_cost_per_unit"),
    )
    .withColumn(
        "spread",
        when(
            col("avg_claim_cost_per_unit").isNotNull() & col("avg_nadac_per_unit").isNotNull(),
            spark_round(col("avg_claim_cost_per_unit") - col("avg_nadac_per_unit"), 4)
        )
    )
)

# Add drug name from dim_drug for readability
drug_cost_spread_analysis = (
    drug_cost_spread
    .join(
        dim_drug.select("ndc", "brand_name", "generic_name"),
        on="ndc",
        how="left"
    )
)

drug_cost_spread_analysis.cache()
print(f"drug_cost_spread_analysis: {drug_cost_spread_analysis.count():,} rows")

# COMMAND ----------

(
    drug_cost_spread_analysis.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.gold.drug_cost_spread_analysis")
)

print("workspace.gold.drug_cost_spread_analysis written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. high_cost_drug_cohort
# MAGIC 
# MAGIC **What it answers:** Who are the highest-spend drug patients, and what does their 
# MAGIC overall healthcare utilization look like? If high Rx spend correlates with lower 
# MAGIC hospitalizations, that is evidence for a "total cost of care" data product.
# MAGIC 
# MAGIC **Ecosystem nodes joined:** Patient (beneficiary) + Rx claims (PDE) + 
# MAGIC Medical claims (inpatient + outpatient)

# COMMAND ----------

bdu = spark.table("workspace.gold.beneficiary_drug_utilization")

# Define "high cost" as top 10% of total Rx spend
p90 = bdu.approxQuantile("total_rx_spend", [0.9], 0.01)[0]
print(f"90th percentile of total Rx spend: ${p90:,.2f}")

# COMMAND ----------

# Compute inpatient and outpatient utilization per beneficiary
fact_ip = spark.table("workspace.silver.fact_inpatient_claims")
fact_op = spark.table("workspace.silver.fact_outpatient_claims")

ip_util = (
    fact_ip
    .groupBy("beneficiary_id")
    .agg(
        count("*").alias("inpatient_claims"),
        spark_sum("claim_payment_amount").alias("inpatient_total_paid"),
        avg("length_of_stay").alias("avg_length_of_stay"),
        spark_sum("utilization_days").alias("total_inpatient_days"),
    )
)

op_util = (
    fact_op
    .groupBy("beneficiary_id")
    .agg(
        count("*").alias("outpatient_claims"),
        spark_sum("claim_payment_amount").alias("outpatient_total_paid"),
    )
)

# Build the cohort table
high_cost_cohort = (
    bdu
    .withColumn(
        "spend_tier",
        when(col("total_rx_spend") >= p90, "high_cost_top_10pct")
        .otherwise("standard")
    )
    .join(ip_util, on="beneficiary_id", how="left")
    .join(op_util, on="beneficiary_id", how="left")
    .select(
        "beneficiary_id",
        "spend_tier",
        "sex",
        "chronic_condition_count",
        "total_rx_fills",
        "unique_drugs",
        "total_rx_spend",
        "total_patient_pay",
        "days_supply_coverage_ratio",
        "total_medical_spend",
        "rx_share_of_total_spend",
        coalesce(col("inpatient_claims"), lit(0)).alias("inpatient_claims"),
        coalesce(col("inpatient_total_paid"), lit(0)).alias("inpatient_total_paid"),
        coalesce(col("avg_length_of_stay"), lit(0)).alias("avg_length_of_stay"),
        coalesce(col("outpatient_claims"), lit(0)).alias("outpatient_claims"),
        coalesce(col("outpatient_total_paid"), lit(0)).alias("outpatient_total_paid"),
    )
    .withColumn(
        "total_all_spend",
        col("total_rx_spend") + col("total_medical_spend")
    )
)

high_cost_cohort.cache()
print(f"high_cost_drug_cohort: {high_cost_cohort.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cohort Comparison: High-Cost vs. Standard
# MAGIC 
# MAGIC This is the analysis that would support (or refute) a "total cost of care" product hypothesis.

# COMMAND ----------

cohort_comparison = (
    high_cost_cohort
    .groupBy("spend_tier")
    .agg(
        count("*").alias("beneficiary_count"),
        spark_round(avg("total_rx_spend"), 2).alias("avg_rx_spend"),
        spark_round(avg("total_medical_spend"), 2).alias("avg_medical_spend"),
        spark_round(avg("total_all_spend"), 2).alias("avg_total_spend"),
        spark_round(avg("rx_share_of_total_spend"), 1).alias("avg_rx_pct"),
        spark_round(avg("chronic_condition_count"), 1).alias("avg_chronic_conditions"),
        spark_round(avg("unique_drugs"), 1).alias("avg_unique_drugs"),
        spark_round(avg("inpatient_claims"), 2).alias("avg_ip_claims"),
        spark_round(avg("avg_length_of_stay"), 1).alias("avg_los"),
        spark_round(avg("outpatient_claims"), 1).alias("avg_op_claims"),
        spark_round(avg("days_supply_coverage_ratio"), 2).alias("avg_adherence_ratio"),
    )
)

cohort_comparison.show(truncate=False)

# COMMAND ----------

(
    high_cost_cohort.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.gold.high_cost_drug_cohort")
)

print("workspace.gold.high_cost_drug_cohort written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Layer Inventory

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'drug_cost_journey' AS table_name, COUNT(*) AS rows FROM workspace.gold.drug_cost_journey
# MAGIC UNION ALL SELECT 'beneficiary_drug_utilization', COUNT(*) FROM workspace.gold.beneficiary_drug_utilization
# MAGIC UNION ALL SELECT 'drug_cost_spread_analysis', COUNT(*) FROM workspace.gold.drug_cost_spread_analysis
# MAGIC UNION ALL SELECT 'high_cost_drug_cohort', COUNT(*) FROM workspace.gold.high_cost_drug_cohort
# MAGIC ORDER BY table_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC **Gold layer complete.** Four analytical tables, each crossing ecosystem boundaries:
# MAGIC 
# MAGIC | Table | Ecosystem Nodes Joined | Product Concept |
# MAGIC |-------|----------------------|-----------------|
# MAGIC | drug_cost_journey | Manufacturer + Pharmacy + Payer | Drug cost transparency tool |
# MAGIC | beneficiary_drug_utilization | Patient + Payer (Rx) + Provider (medical) | Medication adherence monitor |
# MAGIC | drug_cost_spread_analysis | Pharmacy (NADAC) + Payer (claims) over time | Drug pricing intelligence |
# MAGIC | high_cost_drug_cohort | Patient + Rx + Inpatient + Outpatient | Total cost of care analytics |
# MAGIC 
# MAGIC **Next:** Notebook 5 runs exploratory analysis and identifies product opportunities.
