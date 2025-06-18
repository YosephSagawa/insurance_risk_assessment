import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory
Path("outputs/results").mkdir(parents=True, exist_ok=True)

# Load dataset
try:
    df = pd.read_csv('../../data/processed/insurance_data_cleaned.csv', low_memory=False)
    logger.info("Dataset loaded successfully")
except FileNotFoundError:
    logger.error("Dataset file not found. Please check the file path.")
    raise
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise

# Log column names
logger.info(f"Columns in dataset: {df.columns.tolist()}")

# Define metrics
df['HasClaim'] = (df['TotalClaims'] > 0).astype(bool)  # Explicitly compute from TotalClaims
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

# Helper function for Cramer's V
def cramers_v(chi2, n, k1, k2):
    return np.sqrt(chi2 / (n * (min(k1, k2) - 1)))

# Helper function for hypothesis testing
def test_hypothesis(group_col, metric, test_type='numerical', group_a=None, group_b=None):
    results = {}
    try:
        if test_type == 'frequency':
            # Chi-squared test for Claim Frequency
            contingency_table = pd.crosstab(df[group_col], df['HasClaim'])
            chi2, p_value, _, expected = chi2_contingency(contingency_table)
            # Validate expected frequencies
            if (expected < 5).any():
                logger.warning(f"Low expected frequencies (< 5) in {group_col} contingency table. Results may be unreliable.")
                results['warning'] = 'Low expected frequencies detected'
            n = contingency_table.sum().sum()
            k1, k2 = contingency_table.shape
            results['p_value'] = p_value
            results['test_statistic'] = chi2
            results['cramers_v'] = cramers_v(chi2, n, k1, k2)
            results['interpretation'] = ('Reject H0: Significant difference in claim frequency'
                                         if p_value < 0.05 else 'Fail to reject H0: No significant difference')
            # Log contingency table summary
            logger.info(f"Contingency table for {group_col}:\n{contingency_table}")
        else:
            # Mann-Whitney U for numerical metrics (Claim Severity or Margin)
            if group_a and group_b:
                group_a_data = df[df[group_col] == group_a][metric].dropna()
                group_b_data = df[df[group_col] == group_b][metric].dropna()
            else:
                groups = df[group_col].dropna().unique()
                if len(groups) < 2:
                    raise ValueError(f"Less than two groups found for {group_col}")
                group_a_data = df[df[group_col] == groups[0]][metric].dropna()
                group_b_data = df[df[group_col] == groups[1]][metric].dropna()
                logger.info(f"Comparing {group_col}: {groups[0]} vs {groups[1]}")
            stat, p_value = mannwhitneyu(group_a_data, group_b_data, alternative='two-sided')
            results['p_value'] = p_value
            results['test_statistic'] = stat
            results['interpretation'] = ('Reject H0: Significant difference in ' + metric
                                         if p_value < 0.05 else 'Fail to reject H0: No significant difference')
    except Exception as e:
        logger.error(f"Error in hypothesis testing for {group_col}, {metric}: {str(e)}")
        results = {'p_value': np.nan, 'test_statistic': np.nan, 'interpretation': f'Error: {str(e)}'}
    return results

# Test hypotheses
hypotheses = [
    {'col': 'Province', 'metric': 'HasClaim', 'test_type': 'frequency', 'name': 'Claim Frequency by Province'},
    {'col': 'MainCrestaZone', 'metric': 'HasClaim', 'test_type': 'frequency', 'name': 'Claim Frequency by Cresta Zone'},
    {'col': 'PostalCode', 'metric': 'Margin', 'test_type': 'numerical', 'name': 'Margin by Zipcode'},
    {'col': 'Gender', 'metric': 'HasClaim', 'test_type': 'frequency', 'name': 'Claim Frequency by Gender'},
]

# Run tests and save results
results_list = []
for hyp in hypotheses:
    logger.info(f"Testing hypothesis: {hyp['name']}")
    result = test_hypothesis(hyp['col'], hyp['metric'], hyp['test_type'])
    result['hypothesis'] = hyp['name']
    results_list.append(result)

# Save results to CSV
try:
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('outputs/results/hypothesis_test_results.csv', index=False, encoding='utf-8')
    logger.info("Hypothesis test results saved to CSV")
except Exception as e:
    logger.error(f"Error saving results to CSV: {str(e)}")
    raise

# Business interpretations
try:
    with open('outputs/results/hypothesis_interpretations.txt', 'w', encoding='utf-8') as f:
        for result in results_list:
            f.write(f"Hypothesis: {result['hypothesis']}\n")
            f.write(f"P-value: {result['p_value']:.4f}\n")
            f.write(f"Interpretation: {result['interpretation']}\n")
            if 'cramers_v' in result:
                f.write(f"Cramer's V: {result['cramers_v']:.4f}\n")
            if 'warning' in result:
                f.write(f"Warning: {result['warning']}\n")
            if 'Province' in result['hypothesis'] and result['p_value'] < 0.05:
                f.write("Business Insight: Adjust premiums by province due to varying claim frequencies.\n")
            elif 'Cresta Zone' in result['hypothesis'] and result['p_value'] < 0.05:
                f.write("Business Insight: Adjust premiums by Cresta Zone due to varying claim frequencies.\n")
            elif 'PostalCode' in result['hypothesis'] and 'Margin' in result['hypothesis'] and result['p_value'] < 0.05:
                f.write("Business Insight: Optimize pricing in high-margin zip codes to attract low-risk clients.\n")
            f.write("\n")
    logger.info("Business interpretations saved to text file")
except Exception as e:
    logger.error(f"Error writing interpretations: {str(e)}")
    raise