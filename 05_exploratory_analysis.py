# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 5: Exploratory Analysis & Product Opportunity Identification
# MAGIC 
# MAGIC **Purpose:** Run analyses that surface actionable product hypotheses from the gold-layer data.
# MAGIC Each analysis mirrors what would happen in a product discovery session on Amy's team:
# MAGIC start with the data, find a pattern, form a hypothesis, articulate the product concept.
# MAGIC 
# MAGIC **Four analyses:**
# MAGIC 1. Generic vs. Brand Cost Spread (pharmacy margin intelligence)
# MAGIC 2. Chronic Condition Drug Adherence Patterns (adherence monitoring product)
# MAGIC 3. Geographic Pricing Variation (pricing intelligence)
# MAGIC 4. High-Cost Drug Downstream Utilization (total cost of care product)

# COMMAND ----------

from pyspark.sql.functions import (
    col, count, countDistinct, sum as spark_sum, avg, min as spark_min,
    max as spark_max, when, desc, asc, spark_round, lit, percentile_approx,
    stddev, coalesce, concat, expr, ntile
)

# Convenience for rounding display values
def rd(c, n=2):
    return spark_round(col(c), n).alias(c) if isinstance(c, str) else spark_round(c, n)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Analysis 1: Generic vs. Brand Cost Spread
# MAGIC 
# MAGIC **Question:** Where do generic alternatives exist but the acquisition-to-reimbursement 
# MAGIC spread is surprisingly small? These are drugs where pharmacies (and by extension, 
# MAGIC distributors like McKesson) face margin pressure despite a generic being available.
# MAGIC 
# MAGIC **Product hypothesis:** A "margin optimization recommender" that alerts pharmacy 
# MAGIC purchasing teams to therapeutically equivalent drugs with better spreads.

# COMMAND ----------

dcj = spark.table("workspace.gold.drug_cost_journey")

# Split by rate classification (generic vs brand)
# NADAC rate_classification: 'G' = generic, 'B' = brand (or similar codes)
brand_generic_spread = (
    dcj
    .filter(col("rate_classification").isNotNull())
    .groupBy("rate_classification")
    .agg(
        count("*").alias("drug_count"),
        spark_sum("total_claims").alias("total_claims"),
        rd(avg("nadac_per_unit"), 4).alias("avg_nadac_per_unit"),
        rd(avg("avg_claim_cost"), 2).alias("avg_claim_cost"),
        rd(avg("acquisition_to_claim_spread"), 4).alias("avg_spread"),
        rd(avg("patient_cost_share_pct"), 1).alias("avg_patient_cost_share_pct"),
    )
    .orderBy("rate_classification")
)

print("=== Generic vs. Brand: Cost Spread Overview ===")
brand_generic_spread.show(truncate=False)

# COMMAND ----------

# Find specific drugs where generics have unusually thin spreads
# These represent margin risk for pharmacies and distributors
thin_margin_generics = (
    dcj
    .filter(
        (col("rate_classification") == "G") &  # Generic
        (col("acquisition_to_claim_spread").isNotNull()) &
        (col("total_claims") >= 10)  # Minimum volume for relevance
    )
    .orderBy(asc("acquisition_to_claim_spread"))
    .select(
        "ndc",
        "generic_name",
        "labeler_name",
        "total_claims",
        "unique_patients",
        rd("nadac_per_unit", 4),
        rd("cost_per_unit_on_claim", 4),
        rd("acquisition_to_claim_spread", 4),
        rd("patient_cost_share_pct", 1),
    )
    .limit(25)
)

print("=== Top 25 Generic Drugs with Thinnest Margins ===")
print("(Lowest acquisition-to-reimbursement spread)")
thin_margin_generics.show(25, truncate=30)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis 1 Findings
# MAGIC 
# MAGIC **Pattern:** _[Describe what you observe after running on real data. Look for:]_
# MAGIC - Whether generic spreads are meaningfully different from brand spreads
# MAGIC - Specific drug categories where generic margins are compressed
# MAGIC - Whether high-volume generics have thinner margins than low-volume ones
# MAGIC 
# MAGIC **Product Concept: Margin Optimization Recommender**  
# MAGIC A tool that monitors acquisition cost and reimbursement data for a pharmacy's formulary,
# MAGIC flags drugs where margins are eroding, and suggests therapeutically equivalent alternatives
# MAGIC with better spread. For a distributor like McKesson, this could be a value-added service
# MAGIC offered alongside the distribution relationship.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Analysis 2: Chronic Condition Drug Adherence Patterns
# MAGIC 
# MAGIC **Question:** Do patients with certain chronic conditions show worse medication adherence 
# MAGIC (measured by days-supply coverage ratio)? Which conditions are most associated with 
# MAGIC therapy abandonment?
# MAGIC 
# MAGIC **Product hypothesis:** A medication adherence risk scoring model that identifies 
# MAGIC patients likely to stop filling prescriptions, enabling proactive intervention.

# COMMAND ----------

bdu = spark.table("workspace.gold.beneficiary_drug_utilization")

# Adherence by chronic condition
# For each condition, compare adherence of patients WITH vs WITHOUT the condition
chronic_conditions = [
    ("cc_diabetes", "Diabetes"),
    ("cc_heart_failure", "Heart Failure"),
    ("cc_copd", "COPD"),
    ("cc_depression", "Depression"),
    ("cc_cancer", "Cancer"),
]

print("=== Adherence by Chronic Condition (Days Supply Coverage Ratio) ===")
print(f"{'Condition':<20} {'With Condition':>15} {'Without':>10} {'Delta':>8} {'N (with)':>10}")
print("-" * 70)

for cc_col, cc_name in chronic_conditions:
    with_cc = (
        bdu
        .filter((col(cc_col) == 1) & col("days_supply_coverage_ratio").isNotNull())
        .agg(
            avg("days_supply_coverage_ratio").alias("avg_ratio"),
            count("*").alias("n"),
        )
        .collect()[0]
    )
    
    without_cc = (
        bdu
        .filter((col(cc_col) == 0) & col("days_supply_coverage_ratio").isNotNull())
        .agg(avg("days_supply_coverage_ratio").alias("avg_ratio"))
        .collect()[0]
    )
    
    with_val = with_cc["avg_ratio"] or 0
    without_val = without_cc["avg_ratio"] or 0
    delta = with_val - without_val
    n = with_cc["n"] or 0
    
    print(f"  {cc_name:<18} {with_val:>14.3f} {without_val:>10.3f} {delta:>+8.3f} {n:>10,}")

# COMMAND ----------

# Adherence by chronic condition count (comorbidity burden)
adherence_by_burden = (
    bdu
    .filter(col("days_supply_coverage_ratio").isNotNull())
    .groupBy("chronic_condition_count")
    .agg(
        count("*").alias("beneficiary_count"),
        rd(avg("days_supply_coverage_ratio"), 3).alias("avg_adherence"),
        rd(avg("unique_drugs"), 1).alias("avg_drugs"),
        rd(avg("total_rx_spend"), 2).alias("avg_rx_spend"),
        rd(avg("total_rx_fills"), 1).alias("avg_fills"),
    )
    .orderBy("chronic_condition_count")
)

print("\n=== Adherence by Chronic Condition Burden ===")
adherence_by_burden.show(15, truncate=False)

# COMMAND ----------

# Identify low-adherence cohort for potential intervention targeting
low_adherence = (
    bdu
    .filter(
        (col("days_supply_coverage_ratio").isNotNull()) &
        (col("days_supply_coverage_ratio") < 0.5) &  # Less than 50% coverage
        (col("total_rx_fills") >= 3)  # Had enough fills to establish a pattern
    )
    .agg(
        count("*").alias("low_adherence_count"),
        rd(avg("chronic_condition_count"), 1).alias("avg_chronic_conditions"),
        rd(avg("unique_drugs"), 1).alias("avg_unique_drugs"),
        rd(avg("total_rx_spend"), 2).alias("avg_total_rx_spend"),
        rd(avg("total_medical_spend"), 2).alias("avg_medical_spend"),
    )
)

total_benes = bdu.filter(col("days_supply_coverage_ratio").isNotNull()).count()
low_count = low_adherence.collect()[0]["low_adherence_count"]

print(f"\n=== Low Adherence Cohort (Coverage Ratio < 0.5) ===")
print(f"Count: {low_count:,} of {total_benes:,} ({low_count/total_benes*100:.1f}%)")
low_adherence.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis 2 Findings
# MAGIC 
# MAGIC **Pattern:** _[Describe what you observe. Look for:]_
# MAGIC - Which conditions have the lowest adherence
# MAGIC - Whether higher comorbidity burden helps or hurts adherence
# MAGIC - The size of the low-adherence cohort and their downstream medical spend
# MAGIC 
# MAGIC **Product Concept: Medication Adherence Risk Score**  
# MAGIC A predictive model (built in Notebook 6) that scores patients on their likelihood 
# MAGIC of discontinuing therapy. Inputs: chronic condition burden, fill history gaps, 
# MAGIC number of unique drugs, demographics. Output: risk tier (high/medium/low).
# MAGIC For a pharmacy or health plan, this enables proactive outreach before the patient 
# MAGIC lapses. For McKesson, this is a data product that strengthens the pharmacy relationship.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Analysis 3: Geographic Pricing Variation
# MAGIC 
# MAGIC **Question:** Does the acquisition cost for the same drug vary meaningfully by 
# MAGIC pharmacy type (chain vs. independent)? Combined with beneficiary geography, 
# MAGIC are there regional pricing pockets worth flagging?
# MAGIC 
# MAGIC **Product hypothesis:** A geographic pricing intelligence tool that identifies 
# MAGIC regions where acquisition costs are above the national average.

# COMMAND ----------

fact_pricing = spark.table("workspace.silver.fact_drug_pricing")

# NADAC by pharmacy type (chain vs. independent)
pharmacy_type_comparison = (
    fact_pricing
    .filter(
        col("nadac_per_unit").isNotNull() &
        col("pharmacy_type").isNotNull()
    )
    .groupBy("pharmacy_type")
    .agg(
        countDistinct("ndc").alias("unique_drugs"),
        count("*").alias("price_observations"),
        rd(avg("nadac_per_unit"), 4).alias("avg_nadac_per_unit"),
        rd(percentile_approx("nadac_per_unit", 0.5), 4).alias("median_nadac"),
        rd(stddev("nadac_per_unit"), 4).alias("stddev_nadac"),
    )
)

print("=== NADAC by Pharmacy Type ===")
pharmacy_type_comparison.show(truncate=False)

# COMMAND ----------

# For drugs that appear in BOTH pharmacy types, compare prices head-to-head
from pyspark.sql.functions import collect_set, array_contains

# Get NDCs present in both pharmacy types
both_types = (
    fact_pricing
    .filter(col("pharmacy_type").isNotNull())
    .groupBy("ndc")
    .agg(collect_set("pharmacy_type").alias("types"))
    .filter(size(col("types")) > 1)
    .select("ndc")
)

# Pivot: one row per NDC with chain and independent prices
pricing_pivot = (
    fact_pricing
    .join(both_types, on="ndc", how="inner")
    .groupBy("ndc")
    .pivot("pharmacy_type")
    .agg(avg("nadac_per_unit"))
)

# The pivot column names will be the pharmacy_type values
print("Pricing pivot columns:", pricing_pivot.columns)
pricing_pivot.show(5, truncate=False)

# COMMAND ----------

# Beneficiary geographic distribution (for context)
dim_bene = spark.table("workspace.silver.dim_beneficiary")

state_distribution = (
    dim_bene
    .groupBy("state_code")
    .agg(
        count("*").alias("beneficiary_count"),
        rd(avg("chronic_condition_count"), 1).alias("avg_chronic_conditions"),
    )
    .orderBy(desc("beneficiary_count"))
)

print("=== Beneficiary Distribution by State Code ===")
state_distribution.show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis 3 Findings
# MAGIC 
# MAGIC **Pattern:** _[Describe what you observe. Look for:]_
# MAGIC - Meaningful price differences between chain and independent pharmacies
# MAGIC - Specific drug categories where the gap is largest
# MAGIC - Geographic concentration of beneficiaries (useful for market sizing)
# MAGIC 
# MAGIC **Product Concept: Geographic Drug Pricing Intelligence**  
# MAGIC A dashboard that overlays NADAC pricing data with geographic pharmacy data to identify
# MAGIC regions where acquisition costs diverge from the national average. For McKesson's 
# MAGIC pharmacy customers, this helps with purchasing decisions. For McKesson's distribution 
# MAGIC strategy, it identifies markets where pricing pressure differs.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Analysis 4: High-Cost Drug Downstream Utilization
# MAGIC 
# MAGIC **Question:** Do patients who spend the most on prescriptions have lower or higher 
# MAGIC rates of hospitalization and outpatient visits? This is the "total cost of care" question.
# MAGIC 
# MAGIC **Product hypothesis:** If higher Rx spend correlates with lower acute care utilization, 
# MAGIC that supports a data product that quantifies the ROI of medication investment.

# COMMAND ----------

cohort = spark.table("workspace.gold.high_cost_drug_cohort")

# Compare high-cost vs. standard across all dimensions
cohort_summary = (
    cohort
    .groupBy("spend_tier")
    .agg(
        count("*").alias("n"),
        rd(avg("total_rx_spend"), 2).alias("avg_rx_spend"),
        rd(avg("total_medical_spend"), 2).alias("avg_medical_spend"),
        rd(avg("inpatient_claims"), 2).alias("avg_ip_admissions"),
        rd(avg("inpatient_total_paid"), 2).alias("avg_ip_cost"),
        rd(avg("avg_length_of_stay"), 1).alias("avg_los"),
        rd(avg("outpatient_claims"), 1).alias("avg_op_visits"),
        rd(avg("outpatient_total_paid"), 2).alias("avg_op_cost"),
        rd(avg("chronic_condition_count"), 1).alias("avg_chronic_conditions"),
        rd(avg("unique_drugs"), 1).alias("avg_unique_drugs"),
        rd(avg("days_supply_coverage_ratio"), 2).alias("avg_adherence"),
    )
)

print("=== High-Cost vs. Standard Cohort Comparison ===")
cohort_summary.show(truncate=False)

# COMMAND ----------

# Compute the "Rx investment ratio": for every dollar spent on drugs, 
# how many dollars are spent on acute care?
investment_ratio = (
    cohort
    .filter(col("total_rx_spend") > 0)
    .groupBy("spend_tier")
    .agg(
        rd(
            spark_sum("inpatient_total_paid") / spark_sum("total_rx_spend"),
            3
        ).alias("ip_cost_per_rx_dollar"),
        rd(
            spark_sum("outpatient_total_paid") / spark_sum("total_rx_spend"),
            3
        ).alias("op_cost_per_rx_dollar"),
        rd(
            (spark_sum("inpatient_total_paid") + spark_sum("outpatient_total_paid")) 
            / spark_sum("total_rx_spend"),
            3
        ).alias("total_medical_per_rx_dollar"),
    )
)

print("=== Medical Spend per Rx Dollar ===")
investment_ratio.show(truncate=False)

# COMMAND ----------

# Break high-cost cohort into quartiles to see if the relationship is linear
high_cost_only = cohort.filter(col("spend_tier") == "high_cost_top_10pct")

quartile_analysis = (
    high_cost_only
    .withColumn("rx_quartile", ntile(4).over(Window.orderBy("total_rx_spend")))
    .groupBy("rx_quartile")
    .agg(
        count("*").alias("n"),
        rd(avg("total_rx_spend"), 2).alias("avg_rx_spend"),
        rd(avg("inpatient_claims"), 2).alias("avg_ip_admissions"),
        rd(avg("inpatient_total_paid"), 2).alias("avg_ip_cost"),
        rd(avg("outpatient_claims"), 1).alias("avg_op_visits"),
        rd(avg("chronic_condition_count"), 1).alias("avg_conditions"),
    )
    .orderBy("rx_quartile")
)

from pyspark.sql.window import Window

print("=== Within High-Cost Cohort: Rx Spend Quartiles vs. Acute Utilization ===")
quartile_analysis.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis 4 Findings
# MAGIC 
# MAGIC **Pattern:** _[Describe what you observe. The key question is:]_
# MAGIC - Do higher Rx spenders have proportionally lower acute care utilization?
# MAGIC - Or does high Rx spend simply correlate with sicker patients who also use more acute care?
# MAGIC - Does adherence (days supply coverage ratio) differ between tiers?
# MAGIC 
# MAGIC **Product Concept: Total Cost of Care Analytics**  
# MAGIC A product that integrates pharmaceutical and medical claims to compute total cost of care
# MAGIC by patient cohort. For health plans, this supports formulary decisions (is it worth paying 
# MAGIC for the expensive drug if it prevents hospitalizations?). For McKesson, this positions 
# MAGIC their data assets as essential inputs to value-based care models.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary: Product Opportunity Matrix
# MAGIC 
# MAGIC | # | Analysis | Pattern | Product Concept | McKesson Value |
# MAGIC |---|----------|---------|-----------------|----------------|
# MAGIC | 1 | Generic vs. Brand Spread | Margin compression in specific drug categories | Margin Optimization Recommender | Value-add for pharmacy customers |
# MAGIC | 2 | Chronic Condition Adherence | Condition-specific adherence gaps | Adherence Risk Scoring | Strengthens pharmacy relationships |
# MAGIC | 3 | Geographic Pricing | Regional acquisition cost variance | Pricing Intelligence Dashboard | Distribution strategy input |
# MAGIC | 4 | High-Cost Downstream Utilization | Rx spend vs. acute care correlation | Total Cost of Care Analytics | Positions McKesson data in VBC models |
# MAGIC 
# MAGIC Each of these could be taken through Amy's ideate-incubate-pilot cycle.
# MAGIC The gold-layer data foundation built in Notebook 4 supports all four.
# MAGIC 
# MAGIC **Next:** Notebook 6 builds two lightweight ML models as proof-of-concept extensions.
