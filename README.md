# CDC Diabetes Health Indicators — EDA
Exploratory data analysis of clinical and behavioral risk factors associated with diabetes diagnosis using CDC BRFSS 2015 survey data

![Python](https://img.shields.io/badge/Language-Python-blue) ![pandas](https://img.shields.io/badge/pandas-seaborn-green) ![CDC](https://img.shields.io/badge/Data-CDC%20BRFSS-orange)

## Overview
This project analyzes the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset to identify clinical and behavioral indicators associated with diabetes diagnosis. The dataset contains 253,680 survey respondents classified into three groups: no diabetes, prediabetes, and diabetes. The analysis explores risk factor prevalence, BMI distributions, age-based trends, and feature correlations to build a data-driven picture of who is at risk and what comorbidities cluster with diabetes.

## Clinical context
Diabetes affects over 37 million Americans and is one of the leading causes of preventable death and disability. Prediabetes — affecting an estimated 96 million adults — often goes undiagnosed. Early identification of at-risk populations through behavioral and clinical screening data is a core use case for healthcare data analytics, enabling targeted interventions before progression to full diabetes.

## Dataset
CDC Diabetes Health Indicators Dataset via Kaggle — derived from the 2015 BRFSS telephone survey. 253,680 respondents, 21 features.

- Target: `Diabetes_012` — 0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes
- Features: HighBP, HighChol, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, Age, Education, Income
- Source: [Kaggle — CDC Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

## Visualizations
| Plot | Description |
|------|-------------|
| 01 — Target distribution | Class balance across no diabetes, prediabetes, and diabetes groups |
| 02 — BMI by diabetes status | Boxplot showing BMI distribution rises with diagnosis severity |
| 03 — Risk factor prevalence | HighBP, HighChol, Stroke, HeartDisease, PhysActivity by group |
| 04 — Age group trend | Diabetes prevalence (%) across 13 age groups |
| 05 — Correlation heatmap | Pairwise correlations between 12 clinical and behavioral features |
| 06 — General health rating | Self-reported health (Excellent to Poor) by diabetes status |

## Key findings
- High blood pressure is present in ~70% of diabetic respondents vs ~39% of non-diabetic respondents
- Mean BMI rises progressively: No Diabetes → Prediabetes → Diabetes
- Diabetes prevalence increases steadily with age, with the sharpest rise after age 45
- General health (GenHlth) and BMI are the strongest correlates with diabetes diagnosis
- Physical activity shows an inverse relationship — diabetic respondents report lower activity rates
- Respondents rating their health as "Poor" or "Fair" are disproportionately represented in the diabetes group

## How to run
```bash
pip install pandas numpy matplotlib seaborn
python cdc_diabetes_eda.py
```
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and place `diabetes_012_health_indicators_BRFSS2015.csv` in the same folder as the script. Running it generates 6 PNG plots and prints a key findings summary.

## Project structure
```
cdc-diabetes-eda/
├── cdc_diabetes_eda.py
├── 01_target_distribution.png
├── 02_bmi_by_diabetes.png
├── 03_risk_factor_prevalence.png
├── 04_diabetes_by_age.png
├── 05_correlation_heatmap.png
├── 06_general_health.png
└── README.md
```

## Tools
Python 3 — pandas, numpy, matplotlib, seaborn · CDC BRFSS 2015 public domain data
