
# Task 4: Predictive Modeling Report

## Claim Severity Prediction
- **Dataset**: Policies with TotalClaims > 0
- **Models**: Linear Regression, Random Forest, XGBoost
- **Metrics**: RMSE, R-squared
- **Results**:
| Model             | Dataset   |     RMSE |         R2 |
|:------------------|:----------|---------:|-----------:|
| Linear Regression | Train     | 234867   | -36.4938   |
| Linear Regression | Test      | 370584   | -84.3928   |
| Random Forest     | Train     |  23372.8 |   0.62869  |
| Random Forest     | Test      |  36807.5 |   0.157598 |
| XGBoost           | Train     |  33767.5 |   0.224978 |
| XGBoost           | Test      |  36657.4 |   0.164455 |
- **Best Model**: XGBoost
- **SHAP Analysis**: Top features (outputs/models/shap_severity.png):
  - SumInsured: Higher insured values increase claim amounts.
  - VehicleAge: Older vehicles increase claims (see logs for impact).
  - ZipcodeClaimFreq: High-risk zip codes correlate with larger claims.

## Premium Optimization
- **Dataset**: All policies
- **Target**: CalculatedPremiumPerTerm
- **Results**:
| Model             | Dataset   |     RMSE |         R2 |
|:------------------|:----------|---------:|-----------:|
| Linear Regression | Train     | 422.607  | 0.00638345 |
| Linear Regression | Test      | 280.313  | 0.0157044  |
| Random Forest     | Train     |  57.0029 | 0.981922   |
| Random Forest     | Test      |  60.8673 | 0.95359    |
| XGBoost           | Train     | 258.972  | 0.626879   |
| XGBoost           | Test      | 185.074  | 0.570928   |
- **Best Model**: XGBoost
- **SHAP Analysis**: Top features (outputs/models/shap_premium.png):
  - SumInsured: Drives premium due to coverage.
  - Province: High-risk provinces increase premiums.
  - VehicleType: Luxury vehicles command higher premiums.

## Claim Probability Prediction
- **Dataset**: All policies
- **Target**: HasClaim (binary)
- **Results**:
| Model               | Dataset   |   Accuracy |   Precision |   Recall |         F1 |   ROC-AUC |
|:--------------------|:----------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression | Train     |   0.243181 |  0.00347963 | 0.957861 | 0.00693408 |  0.689117 |
| Logistic Regression | Test      |   0.243221 |  0.00356875 | 0.932874 | 0.00711029 |  0.684264 |
| Random Forest       | Train     |   0.957572 |  0.0251922  | 0.381513 | 0.0472635  |  0.962377 |
| Random Forest       | Test      |   0.956394 |  0.0140862  | 0.203098 | 0.0263452  |  0.645107 |
| XGBoost             | Train     |   0.77192  |  0.0113572  | 0.949252 | 0.0224458  |  0.901517 |
| XGBoost             | Test      |   0.771583 |  0.0117762  | 0.936317 | 0.0232598  |  0.888852 |
- **Best Model**: XGBoost
- **SHAP Analysis**: Top features (outputs/models/shap_claim.png):
  - ZipcodeClaimFreq: Higher claim frequency increases probability.
  - Province: Reflects regional risk (Task-3).
  - VehicleAge: Older vehicles are riskier.

## Business Implications
- **Pricing Framework**: Premium = (Claim Probability × Claim Severity) + 10% Expense + 5% Profit
- **Recommendations**:
  - Adjust premiums based on SumInsured, VehicleAge, high-risk zip codes/provinces.
  - Target low-risk zip codes for marketing.
  - Monitor models quarterly.
