# =============================================================================
# CDC Diabetes Health Indicators — Exploratory Data Analysis
# Author: Rachel Ratajczak
# Dataset: CDC Diabetes Health Indicators (Kaggle)
# https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
# =============================================================================
# This script performs an exploratory data analysis on the CDC Behavioral Risk
# Factor Surveillance System (BRFSS) dataset to identify key clinical and
# behavioral indicators associated with diabetes diagnosis.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings as wr
wr.filterwarnings('ignore')
import matplotlib.ticker as mtick 

# =============================================================================
# 1. LOAD DATA
# =============================================================================
# Download from Kaggle:
# https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
# File: diabetes_012_health_indicators_BRFSS2015.csv
# Get full filepath to load into a pandas dataframe

df = pd.read_csv("/Users/rachelratajczak/Desktop/Career/GitHub/EDA_Diabetes_Project/archive/diabetes_012_health_indicators_BRFSS2015.csv")
print(df.head())

# =============================================================================
# 2. DATA OVERVIEW
# =============================================================================

print("\n Number of rows (observations) and columns (features)")
print("-------------------------------------------------------------------------")
print(df.shape)

print("\n Number of records, data type, missing values,  and memory usage ")
print("-------------------------------------------------------------------------")
df.info()

print("\n Statistical summary")
print("-------------------------------------------------------------------------")
print(df.describe().transpose())

# Convert column names into Python list 
df.columns.tolist()

# Check for missing values 
print("\n Missing values")
print("-------------------------------------------------------------------------")
print(df.isnull().sum())

# Check for duplicates - looking for unique values
print("\n Duplicate values")
print("-------------------------------------------------------------------------")
print(df.nunique())

# =============================================================================
# 3. DATA ANALYSIS
# =============================================================================

# --- 3.1 Target variable distribution ---------------------------------------
print("\n Diabetes_012 value counts")
print("-------------------------------------------------------------------------")
print(df["Diabetes_012"].value_counts())

labels = ["No Diabetes (0)", "Prediabetes (1)", "Diabetes (2)"]
counts = df["Diabetes_012"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
sns.barplot(x=labels, y=counts.values, palette=["#5DCAA5", "#FAC775", "#D85A30"])
plt.title("Diabetes Diagnosis Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("01_target_distribution.png")
plt.show()
print("Saved: 01_target_distribution.png")

# --- 3.2 BMI by diabetes status ---------------------------------------------
print("\n Mean BMI by Diabetes_012")
print("-------------------------------------------------------------------------")
print(df.groupby("Diabetes_012")["BMI"].mean().round(2))

plt.figure(figsize=(8, 5))
sns.boxplot(x="Diabetes_012", y="BMI", data=df,
            palette=["#5DCAA5", "#FAC775", "#D85A30"])
plt.title("BMI Distribution by Diabetes Status")
plt.xlabel("Diabetes Status (0=None, 1=Prediabetes, 2=Diabetes)")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig("02_bmi_by_diabetes.png")
plt.show()
print("Saved: 02_bmi_by_diabetes.png")

# --- 3.3 Risk factor prevalence ---------------------------------------------
risk_factors = ["HighBP", "HighChol", "Smoker", "Stroke",
                "HeartDiseaseorAttack", "PhysActivity", "DiffWalk"]

prevalence = df.groupby("Diabetes_012")[risk_factors].mean() * 100
prevalence.index = ["No Diabetes", "Prediabetes", "Diabetes"]

print("\n Risk factor prevalence (%) by diabetes status")
print("-------------------------------------------------------------------------")
print(prevalence.round(1).transpose())

prevalence.transpose().plot(kind="bar", figsize=(12, 6),
                            color=["#5DCAA5", "#FAC775", "#D85A30"],
                            alpha=0.85, edgecolor="white")
plt.title("Risk Factor Prevalence by Diabetes Status")
plt.xlabel("Risk Factor")
plt.ylabel("Prevalence (%)")
plt.xticks(rotation=30, ha="right")
plt.legend(title="Diagnosis")
plt.tight_layout()
plt.savefig("03_risk_factor_prevalence.png")
plt.show()
print("Saved: 03_risk_factor_prevalence.png")

# --- 3.4 Age group analysis -------------------------------------------------
# Age coded 1-13 (1 = 18-24 ... 13 = 80+)
age_labels = {1:"18-24", 2:"25-29", 3:"30-34", 4:"35-39", 5:"40-44",
              6:"45-49", 7:"50-54", 8:"55-59", 9:"60-64", 10:"65-69",
              11:"70-74", 12:"75-79", 13:"80+"}

age_diab  = df[df["Diabetes_012"] == 2].groupby("Age").size()
age_total = df.groupby("Age").size()
age_pct   = (age_diab / age_total * 100).round(1)

print("\n Diabetes prevalence (%) by age group")
print("-------------------------------------------------------------------------")
print(age_pct)

plt.figure(figsize=(11, 5))
plt.plot(list(age_labels.values()), age_pct.values,
         marker="o", color="#D85A30", linewidth=2.5, markersize=7)
plt.fill_between(range(len(age_pct)), age_pct.values, alpha=0.12, color="#D85A30")
plt.title("Diabetes Prevalence by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Diabetes Prevalence (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("04_diabetes_by_age.png")
plt.show()
print("Saved: 04_diabetes_by_age.png")

# --- 3.5 Correlation heatmap ------------------------------------------------
key_cols = ["Diabetes_012", "BMI", "Age", "HighBP", "HighChol",
            "HeartDiseaseorAttack", "PhysActivity", "GenHlth",
            "MentHlth", "PhysHlth", "DiffWalk", "Stroke"]

corr = df[key_cols].corr()

print("\n Top correlates with Diabetes_012")
print("-------------------------------------------------------------------------")
print(corr["Diabetes_012"].drop("Diabetes_012").abs()
      .sort_values(ascending=False).round(3))

plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, annot_kws={"size": 8})
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("05_correlation_heatmap.png")
plt.show()
print("Saved: 05_correlation_heatmap.png")

# --- 3.6 General health rating by diabetes status ---------------------------
health_map = {1:"Excellent", 2:"Very Good", 3:"Good", 4:"Fair", 5:"Poor"}
df["GenHlth_label"] = df["GenHlth"].map(health_map)

genhlth_pct = (df.groupby(["GenHlth_label", "Diabetes_012"])
               .size().unstack(fill_value=0))
genhlth_pct.columns = ["No Diabetes", "Prediabetes", "Diabetes"]
genhlth_pct = genhlth_pct.div(genhlth_pct.sum(axis=1), axis=0) * 100
genhlth_pct = genhlth_pct.loc[["Excellent","Very Good","Good","Fair","Poor"]]

print("\n General health rating distribution by diabetes status (%)")
print("-------------------------------------------------------------------------")
print(genhlth_pct.round(1))

genhlth_pct.plot(kind="bar", figsize=(9, 5),
                 color=["#5DCAA5", "#FAC775", "#D85A30"],
                 alpha=0.85, edgecolor="white", width=0.6)
plt.title("Self-Reported General Health by Diabetes Status")
plt.xlabel("General Health Rating")
plt.ylabel("Proportion (%)")
plt.xticks(rotation=0)
plt.legend(title="Diagnosis")
plt.tight_layout()
plt.savefig("06_general_health.png")
plt.show()
print("Saved: 06_general_health.png")

# =============================================================================
# 4. KEY FINDINGS
# =============================================================================

print("\n")
print("=" * 60)
print("KEY FINDINGS")
print("=" * 60)

total  = len(df)
n_diab = int((df["Diabetes_012"] == 2).sum())
n_pre  = int((df["Diabetes_012"] == 1).sum())
n_none = int((df["Diabetes_012"] == 0).sum())

print(f"\n1. Class distribution ({total:,} total respondents):")
print(f"   No diabetes:  {n_none:,} ({n_none/total*100:.1f}%)")
print(f"   Prediabetes:  {n_pre:,}  ({n_pre/total*100:.1f}%)")
print(f"   Diabetes:     {n_diab:,} ({n_diab/total*100:.1f}%)")

bmi_means = df.groupby("Diabetes_012")["BMI"].mean()
print(f"\n2. Mean BMI — No Diabetes: {bmi_means[0]:.1f} | "
      f"Prediabetes: {bmi_means[1]:.1f} | Diabetes: {bmi_means[2]:.1f}")

top5 = (corr["Diabetes_012"].drop("Diabetes_012").abs()
        .sort_values(ascending=False).head(5))
print(f"\n3. Top 5 correlates with diabetes diagnosis:")
for feat, val in top5.items():
    print(f"   {feat}: r={val:.3f}")

print(f"\n4. High blood pressure prevalence:")
print(f"   No Diabetes: {prevalence.loc['No Diabetes','HighBP']:.1f}% | "
      f"Prediabetes: {prevalence.loc['Prediabetes','HighBP']:.1f}% | "
      f"Diabetes: {prevalence.loc['Diabetes','HighBP']:.1f}%")

print(f"\n5. Diabetes prevalence rises with age, "
      f"peaking at {age_pct.max():.1f}% in the "
      f"{age_labels[int(age_pct.idxmax())]} age group.")

print("\n" + "=" * 60)
print("All plots saved. EDA complete.")
print("=" * 60)
