Machine Learning Model for Predicting Compressive Strength of Portland Cement under Different Cooling Conditions
Overview

This repository contains the dataset and Python code used to predict the compressive strength of Portland cement under various cooling regimes using a Random Forest regression model.
The study aims to evaluate how cooling rate, clinker phase composition, and age affect mechanical performance and to optimize cooling conditions for sustainable cement production.

Files
run.py — Main Python script for data preprocessing, model training, evaluation, and visualization.

clinker_rf_data.csv — Dataset containing clinker composition, grain sizes, cooling rate, and measured strength (not included here).
Input Features
Feature	Description
C3S_wt	Alite content (wt%)
C2S_wt	Belite content (wt%)
C3A_wt	Tricalcium aluminate (wt%)
C4AF_wt	Tetracalcium aluminoferrite (wt%)
alite_grain_um	Average grain size of alite (µm)
belite_grain_um	Average grain size of belite (µm)
belite_alite_ratio	Belite-to-alite ratio
C2S_polymorphic_ratio	Ratio of α', β, and γ-C2S polymorphs
cooling_rate_C_per_min	Cooling rate (°C/min)
age_days	Curing age (days)
Target:
strength_MPa — Measured compressive strength (MPa)

Model Workflow

Load and clean dataset

Encode cooling rate categories

Split data into training and test sets (80:20)

Train a Random Forest Regressor

Perform 10-fold cross-validation

Evaluate model with R² and RMSE

Generate plots:

Feature importance

Measured vs Predicted strength

Strength distribution by cooling regime

Predicted strength vs age

Example Output
=== 10-fold Cross Validation ===
R2  mean ± std : 0.923 ± 0.028
RMSE mean ± std: 2.85 ± 0.41

=== Hold-out Test Set Performance ===
R2   (test) : 0.937
RMSE (test) : 2.62 MPa
Dependencies
Install required Python libraries:
pip install numpy pandas matplotlib scikit-learn

Install required Python libraries:

pip install numpy pandas matplotlib scikit-learn
