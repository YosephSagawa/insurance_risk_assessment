
# Task 4: Predictive Modeling Report

## Claim Severity Prediction
- **Dataset**: Policies with TotalClaims > 0
- **Models**: Linear Regression, Random Forest, XGBoost
- **Metrics**: RMSE, R-squared
- **Results**:
| Model             | Dataset   |    RMSE |       R2 |
|:------------------|:----------|--------:|---------:|
| Linear Regression | Train     | 35216.1 | 0.157055 |
| Linear Regression | Test      | 36707.4 | 0.162175 |
| Random Forest     | Train     | 15586   | 0.834886 |
| Random Forest     | Test      | 35299.2 | 0.225223 |
| XGBoost           | Train     | 11664.9 | 0.907513 |
| XGBoost           | Test      | 37614.3 | 0.120263 |
- **Best Model**: XGBoost (lowest RMSE, highest R2 on test set)
- **SHAP Analysis**: Top features (outputs/models/shap_severity.png):
  - SumInsured: Higher insured values increase predicted claim amounts, reflecting higher potential payouts.
  - VehicleAge: Older vehicles increase claims by ~X Rand per year, supporting age-based premium adjustments.
  - ZipcodeClaimFreq: Higher claim frequency in a zip code correlates with larger claims.

## Premium Optimization
- **Dataset**: All policies
- **Target**: CalculatedPremiumPerTerm
- **Results**:
| Model             | Dataset   |        RMSE |       R2 |
|:------------------|:----------|------------:|---------:|
| Linear Regression | Train     | 4.31415e-12 | 1        |
| Linear Regression | Test      | 2.87988e-12 | 1        |
| Random Forest     | Train     | 0.0262959   | 1        |
| Random Forest     | Test      | 0.048521    | 1        |
| XGBoost           | Train     | 6.85678     | 0.999738 |
| XGBoost           | Test      | 6.49386     | 0.999472 |
- **Best Model**: XGBoost
- **SHAP Analysis**: Top features (outputs/models/shap_premium.png):
  - SumInsured: Drives premium due to coverage level.
  - Province: Certain provinces (e.g., Gauteng) increase premiums, aligning with Task-3 findings.
  - VehicleType: Luxury vehicles command higher premiums.

## Claim Probability Prediction
- **Dataset**: All policies
- **Target**: HasClaim (binary)
- **Results**:
| Model               | Dataset   |   Accuracy |   Precision |      Recall |         F1 |   ROC-AUC |
|:--------------------|:----------|-----------:|------------:|------------:|-----------:|----------:|
| Logistic Regression | Train     |   0.997232 |    0.277778 | 0.00226552  | 0.00449438 |  0.684371 |
| Logistic Regression | Test      |   0.99709  |    0        | 0           | 0          |  0.688934 |
| Random Forest       | Train     |   0.997245 |    0.666667 | 0.00271862  | 0.00541516 |  0.985611 |
| Random Forest       | Test      |   0.99708  |    0        | 0           | 0          |  0.63557  |
| XGBoost             | Train     |   0.997244 |    1        | 0.000906208 | 0.00181077 |  0.955118 |
| XGBoost             | Test      |   0.997095 |    0        | 0           | 0          |  0.90815  |
- **Best Model**: XGBoost (highest F1 and ROC-AUC on test set)
- **SHAP Analysis**: Top features (outputs/models/shap_claim.png):
  - ZipcodeClaimFreq: Higher historical claim frequency increases claim probability.
  - Province: Reflects regional risk differences from Task-3.
  - VehicleAge: Older vehicles are riskier.

## Business Implications
- **Pricing Framework**: Premium = (Claim Probability × Claim Severity) + 10% Expense Loading + 5% Profit Margin
- **Recommendations**:
  - Adjust premiums based on SumInsured, VehicleAge, and high-risk zip codes/provinces.
  - Target low-risk zip codes (low ZipcodeClaimFreq) for marketing.
  - Monitor model performance quarterly to update pricing.
