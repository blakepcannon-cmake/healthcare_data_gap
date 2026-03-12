# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 6: AI/ML Extension
# MAGIC
# MAGIC **Purpose:** Demonstrate how the gold-layer data foundation supports AI use cases.
# MAGIC Two lightweight models, both feasible on Free Edition serverless compute.
# MAGIC
# MAGIC **Models:**
# MAGIC 1. **Prescription Refill Prediction** -- Logistic regression predicting whether a patient 
# MAGIC    will refill a prescription within the expected window. Powers a medication adherence 
# MAGIC    alerting product.
# MAGIC 2. **Drug Cost Anomaly Detection** -- Isolation forest flagging drugs whose acquisition 
# MAGIC    cost has deviated significantly from recent trends. Powers a pricing monitoring product.
# MAGIC
# MAGIC **Important framing:** These are proofs of concept, not production models. The goal is to 
# MAGIC show that the data architecture supports ML, not to achieve state-of-the-art accuracy.
# MAGIC On Amy's team, this would be the "incubate" phase output -- enough to validate the 
# MAGIC product concept before investing in production-grade ML infrastructure.

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Model 1: Prescription Refill Prediction
# MAGIC
# MAGIC **Business context:** A patient is prescribed a 30-day supply of a medication. 
# MAGIC Do they come back for a refill within a reasonable window (e.g., 45 days)? 
# MAGIC If not, they may be abandoning therapy. Early identification of at-risk patients 
# MAGIC enables pharmacies or health plans to intervene (reminder calls, pharmacist outreach, 
# MAGIC prior authorization help).
# MAGIC
# MAGIC **Target variable:** Binary -- did the patient refill within 1.5x their days_supply?
# MAGIC
# MAGIC **Features:** Demographics, chronic conditions, prescription history patterns.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

# Load gold table
bdu = spark.table("workspace.gold.beneficiary_drug_utilization").toPandas()

print(f"Beneficiary drug utilization: {len(bdu):,} rows")
print(f"Columns: {list(bdu.columns)}")

# COMMAND ----------

# Build per-beneficiary fill sequence from silver PDE data
# We need to look at consecutive fills for the same drug by the same patient
fact_rx = spark.table("workspace.silver.fact_prescription_events")

# For each beneficiary + NDC combination, order fills by date
# and compute the gap between consecutive fills
from pyspark.sql.functions import (
    col, lag, datediff, when, lit, count, avg, sum as spark_sum,
    row_number, desc
)
from pyspark.sql.window import Window

fill_window = Window.partitionBy("beneficiary_id", "ndc").orderBy("service_date")

fill_sequences = (
    fact_rx
    .filter(col("days_supply").isNotNull() & (col("days_supply") > 0))
    .withColumn("prev_fill_date", lag("service_date").over(fill_window))
    .withColumn("prev_days_supply", lag("days_supply").over(fill_window))
    .withColumn("fill_number", row_number().over(fill_window))
    .filter(col("fill_number") > 1)  # Need a prior fill to compute gap
    .withColumn(
        "days_between_fills",
        datediff(col("service_date"), col("prev_fill_date"))
    )
    .withColumn(
        "refilled_on_time",
        when(
            col("days_between_fills") <= (col("prev_days_supply") * 1.5),
            1
        ).otherwise(0)
    )
)

# Aggregate to beneficiary level: what is their historical on-time refill rate?
bene_refill_stats = (
    fill_sequences
    .groupBy("beneficiary_id")
    .agg(
        count("*").alias("total_refill_opportunities"),
        spark_sum("refilled_on_time").alias("on_time_refills"),
        avg("days_between_fills").alias("avg_gap_days"),
        avg("days_supply").alias("avg_days_supply"),
    )
    .withColumn(
        "historical_refill_rate",
        col("on_time_refills") / col("total_refill_opportunities")
    )
    .toPandas()
)

print(f"Beneficiaries with refill history: {len(bene_refill_stats):,}")
print(f"Average on-time refill rate: {bene_refill_stats['historical_refill_rate'].mean():.3f}")
print(f"Columns: {list(bene_refill_stats.columns)}")

# COMMAND ----------

# Merge refill stats with beneficiary utilization features
model_df = bdu.merge(bene_refill_stats, on="beneficiary_id", how="inner")

# Create binary target: will this patient be a "low adherer" overall?
# Define low adherence as on-time refill rate below 0.6
model_df["low_adherence"] = (model_df["historical_refill_rate"] < 0.6).astype(int)

print(f"Model dataset: {len(model_df):,} rows")
print(f"Low adherence rate: {model_df['low_adherence'].mean():.3f}")
print(f"Columns: {list(model_df.columns)}")

# COMMAND ----------

# DBTITLE 1,Untitled
# Select features
feature_cols = [
    "chronic_condition_count",
    "cc_diabetes",
    "cc_heart_failure",
    "cc_copd",
    "cc_depression",
    "cc_cancer",
    "unique_drugs",
    "total_rx_fills",
    "total_rx_spend",
    "total_patient_pay",
    "avg_days_supply_x",  # Fixed: use actual column name
    "avg_days_supply_y",  # Fixed: use actual column name; conflict in previous cell merge created two columsn
    "total_medical_spend",
    "rx_share_of_total_spend",
]

# Drop rows with nulls in feature columns
model_clean = model_df[feature_cols + ["low_adherence"]].dropna()
print(f"Clean rows for modeling: {len(model_clean):,}")

X = model_clean[feature_cols].values
y = model_clean["low_adherence"].values

# COMMAND ----------

# MAGIC %md
# MAGIC ## ### Train and Evaluate

# COMMAND ----------

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train set: {len(X_train):,} ({y_train.mean():.3f} positive rate)")
print(f"Test set:  {len(X_test):,} ({y_test.mean():.3f} positive rate)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

# Model 1a: Logistic Regression (interpretable baseline)
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("=== Logistic Regression Results ===")
print(classification_report(y_test, y_pred_lr, target_names=["On-Time", "Low Adherence"]))

try:
    auc = roc_auc_score(y_test, y_prob_lr)
    print(f"ROC AUC: {auc:.3f}")
except:
    print("ROC AUC: Could not compute (single class in test set)")

# COMMAND ----------

# Model 1b: Gradient Boosted Trees (better performance, still lightweight)
gbt = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
)
gbt.fit(X_train_scaled, y_train)

y_pred_gbt = gbt.predict(X_test_scaled)
y_prob_gbt = gbt.predict_proba(X_test_scaled)[:, 1]

print("=== Gradient Boosted Trees Results ===")
print(classification_report(y_test, y_pred_gbt, target_names=["On-Time", "Low Adherence"]))

try:
    auc = roc_auc_score(y_test, y_prob_gbt)
    print(f"ROC AUC: {auc:.3f}")
except:
    print("ROC AUC: Could not compute (single class in test set)")

# COMMAND ----------

# Feature importance from GBT
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": gbt.feature_importances_
}).sort_values("importance", ascending=False)

print("=== Feature Importance (Gradient Boosted Trees) ===")
for _, row in feature_importance.iterrows():
    bar = "#" * int(row["importance"] * 100)
    print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")

# COMMAND ----------

# Logistic regression coefficients (for interpretability)
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr.coef_[0],
    "abs_coef": np.abs(lr.coef_[0])
}).sort_values("abs_coef", ascending=False)

print("\n=== Logistic Regression Coefficients ===")
print("(Positive = increases likelihood of low adherence)")
for _, row in coef_df.iterrows():
    direction = "+" if row["coefficient"] > 0 else "-"
    print(f"  {direction} {row['feature']:<35} {row['coefficient']:>8.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 1 Summary
# MAGIC
# MAGIC **What this demonstrates:**
# MAGIC - The gold-layer data (beneficiary_drug_utilization) directly supports ML feature engineering
# MAGIC - Both models are lightweight enough for Free Edition compute
# MAGIC - Feature importance reveals which patient characteristics drive adherence risk
# MAGIC - The logistic regression provides interpretable coefficients (important for healthcare 
# MAGIC   where "why" matters as much as "what")
# MAGIC
# MAGIC **Production path:** In a real deployment, this model would be:
# MAGIC - Retrained on actual (non-synthetic) claims data
# MAGIC - Enriched with features like SDoH (social determinants of health), prior auth history, 
# MAGIC   and pharmacy interaction data
# MAGIC - Deployed as a batch scoring job in Databricks, outputting risk tiers to a pharmacy 
# MAGIC   workflow system
# MAGIC - Monitored for drift as population characteristics change

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Model 2: Drug Cost Anomaly Detection
# MAGIC
# MAGIC **Business context:** Drug acquisition costs shift over time due to market dynamics, 
# MAGIC supply shortages, generic entries, and manufacturer pricing decisions. Sudden or 
# MAGIC unusual cost movements represent either risk (margin compression) or opportunity 
# MAGIC (new generic entry). An automated anomaly detector flags drugs that need attention.
# MAGIC
# MAGIC **Approach:** Isolation Forest on NADAC pricing features per drug.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering for Anomaly Detection

# COMMAND ----------

# Build drug-level pricing features from NADAC time series
fact_pricing = spark.table("workspace.silver.fact_drug_pricing")

# Compute pricing statistics per NDC
from pyspark.sql.functions import stddev, percentile_approx, round as spark_round, min as spark_min, max as spark_max

drug_pricing_features = (
    fact_pricing
    .filter(col("nadac_per_unit").isNotNull() & (col("nadac_per_unit") > 0))
    .groupBy("ndc")
    .agg(
        count("*").alias("price_observations"),
        avg("nadac_per_unit").alias("mean_price"),
        stddev("nadac_per_unit").alias("price_stddev"),
        spark_min("nadac_per_unit").alias("min_price"),
        spark_max("nadac_per_unit").alias("max_price"),
        percentile_approx("nadac_per_unit", 0.5).alias("median_price"),
    )
    .filter(col("price_observations") >= 4)  # Need enough data points
    .withColumn(
        "price_range",
        col("max_price") - col("min_price")
    )
    .withColumn(
        "coefficient_of_variation",
        when(col("mean_price") > 0, col("price_stddev") / col("mean_price"))
    )
    .withColumn(
        "range_to_mean_ratio",
        when(col("mean_price") > 0, col("price_range") / col("mean_price"))
    )
    .toPandas()
)

# Also get the most recent price and compute its deviation from mean
from pyspark.sql.functions import first as spark_first

latest_prices = (
    fact_pricing
    .filter(col("nadac_per_unit").isNotNull())
    .withColumn(
        "_rank",
        row_number().over(
            Window.partitionBy("ndc").orderBy(desc("effective_date"))
        )
    )
    .filter(col("_rank") == 1)
    .select(
        col("ndc"),
        col("nadac_per_unit").alias("latest_price"),
        col("effective_date").alias("latest_date"),
        col("rate_classification"),
    )
    .toPandas()
)

# Merge
anomaly_df = drug_pricing_features.merge(latest_prices, on="ndc", how="inner")

# Compute deviation of latest price from historical mean
anomaly_df["latest_deviation"] = (
    (anomaly_df["latest_price"] - anomaly_df["mean_price"]) / 
    anomaly_df["mean_price"].replace(0, np.nan)
)

anomaly_df["latest_z_score"] = (
    (anomaly_df["latest_price"] - anomaly_df["mean_price"]) / 
    anomaly_df["price_stddev"].replace(0, np.nan)
)

print(f"Drugs with sufficient pricing history: {len(anomaly_df):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Isolation Forest

# COMMAND ----------

# Features for anomaly detection
anomaly_features = [
    "mean_price",
    "price_stddev",
    "coefficient_of_variation",
    "range_to_mean_ratio",
    "latest_deviation",
]

# Clean
anomaly_clean = anomaly_df.dropna(subset=anomaly_features)
X_anomaly = anomaly_clean[anomaly_features].values

print(f"Drugs for anomaly detection: {len(anomaly_clean):,}")

# Scale
scaler_a = StandardScaler()
X_anomaly_scaled = scaler_a.fit_transform(X_anomaly)

# Fit Isolation Forest
# contamination=0.05 means we expect ~5% of drugs to be anomalous
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
)
anomaly_clean["anomaly_score"] = iso_forest.fit_predict(X_anomaly_scaled)
anomaly_clean["anomaly_raw_score"] = iso_forest.decision_function(X_anomaly_scaled)

# -1 = anomaly, 1 = normal in sklearn's convention
anomaly_clean["is_anomaly"] = (anomaly_clean["anomaly_score"] == -1).astype(int)

n_anomalies = anomaly_clean["is_anomaly"].sum()
print(f"Anomalies detected: {n_anomalies:,} ({n_anomalies/len(anomaly_clean)*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Anomaly Results

# COMMAND ----------

# Get drug names for the anomalies
dim_drug_pd = spark.table("workspace.silver.dim_drug").select(
    "ndc", "brand_name", "generic_name", "labeler_name"
).toPandas()

anomaly_results = anomaly_clean.merge(dim_drug_pd, on="ndc", how="left")

# Show the most anomalous drugs
anomalies = (
    anomaly_results[anomaly_results["is_anomaly"] == 1]
    .sort_values("anomaly_raw_score")
    .head(20)
)

print("=== Top 20 Most Anomalous Drug Pricing Patterns ===")
print(f"{'NDC':<15} {'Drug Name':<35} {'Mean $':>10} {'Latest $':>10} {'Deviation':>10} {'CV':>8}")
print("-" * 90)

for _, row in anomalies.iterrows():
    name = str(row.get("generic_name") or row.get("brand_name") or "Unknown")[:33]
    print(
        f"  {row['ndc']:<13} {name:<35} "
        f"${row['mean_price']:>8.2f} ${row['latest_price']:>8.2f} "
        f"{row['latest_deviation']:>+9.1%} {row['coefficient_of_variation']:>7.3f}"
    )

# COMMAND ----------

# Compare anomalous vs. normal drugs
comparison = anomaly_results.groupby("is_anomaly").agg({
    "mean_price": "mean",
    "price_stddev": "mean",
    "coefficient_of_variation": "mean",
    "latest_deviation": "mean",
    "range_to_mean_ratio": "mean",
    "ndc": "count",
}).round(4)

comparison.columns = ["avg_mean_price", "avg_stddev", "avg_cv", "avg_deviation", "avg_range_ratio", "count"]
print("\n=== Anomalous vs. Normal Drug Pricing Characteristics ===")
print(comparison.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Z-Score Based Detection (Complementary Approach)
# MAGIC
# MAGIC As a simpler, more interpretable alternative to isolation forest, flag drugs 
# MAGIC where the latest price deviates by more than 2 standard deviations from the historical mean.

# COMMAND ----------

z_anomalies = anomaly_results[
    (anomaly_results["latest_z_score"].abs() > 2) &
    (anomaly_results["price_observations"] >= 8)
].sort_values("latest_z_score", key=abs, ascending=False)

print(f"=== Drugs with |z-score| > 2 (price_obs >= 8) ===")
print(f"Count: {len(z_anomalies):,}")
print()
print(f"{'NDC':<15} {'Drug Name':<30} {'Z-Score':>10} {'Direction':>12} {'Observations':>14}")
print("-" * 85)

for _, row in z_anomalies.head(20).iterrows():
    name = str(row.get("generic_name") or row.get("brand_name") or "Unknown")[:28]
    direction = "PRICE UP" if row["latest_z_score"] > 0 else "PRICE DOWN"
    print(
        f"  {row['ndc']:<13} {name:<30} "
        f"{row['latest_z_score']:>+9.2f} {direction:>12} "
        f"{int(row['price_observations']):>14}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 2 Summary
# MAGIC
# MAGIC **What this demonstrates:**
# MAGIC - Time series pricing data (NADAC) supports automated monitoring use cases
# MAGIC - Both statistical (z-score) and ML (isolation forest) approaches work on this data
# MAGIC - The z-score approach is more interpretable and easier to explain to stakeholders
# MAGIC - The isolation forest captures multivariate anomalies (e.g., a drug that is not just 
# MAGIC   expensive but also unusually volatile)
# MAGIC
# MAGIC **Production path:** In a real deployment, this would be:
# MAGIC - Run as a scheduled Databricks job (weekly, aligned with NADAC update cadence)
# MAGIC - Output anomaly flags to a dashboard or alert system
# MAGIC - Enriched with supply chain signals (shortage alerts, generic entry dates, 
# MAGIC   manufacturer announcements)
# MAGIC - For McKesson, this could feed into procurement recommendations or customer 
# MAGIC   pricing communications

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Overall Summary: AI/ML on the Healthcare Data Foundation
# MAGIC
# MAGIC | Model | Input Data | Business Question | Technique | Product Concept |
# MAGIC |-------|-----------|-------------------|-----------|-----------------|
# MAGIC | Refill Prediction | Gold: beneficiary_drug_utilization + Silver: PDE sequences | Which patients will stop filling? | Logistic Regression / GBT | Adherence alerting |
# MAGIC | Cost Anomaly | Silver: fact_drug_pricing time series | Which drug prices are behaving unusually? | Isolation Forest / Z-Score | Pricing monitor |
# MAGIC
# MAGIC **The architecture point:** Both models were built directly on the medallion data 
# MAGIC foundation with minimal additional feature engineering. The silver and gold tables 
# MAGIC were designed with analytical consumption in mind, which is why ML features can be 
# MAGIC derived from them without extensive preprocessing. This is the difference between 
# MAGIC a data warehouse and a data product platform.
# MAGIC
# MAGIC **What this says about the role:** The Sr. Data Product Manager does not need to 
# MAGIC build production ML pipelines. But they do need to know what the data can support, 
# MAGIC how to frame the problem, and how to evaluate whether a proof of concept is worth 
# MAGIC taking to the next stage. That is exactly what this notebook demonstrates.
