# Customer Churn Prediction - Final Analysis Report

**Date:** 2026-08-29  
**Project:** Customer Churn Prediction - End-to-End Workflow  
**Status:** ✅ Complete

---

## 1. Project Scope

This project covers the full churn prediction workflow from data preparation to model comparison and SQL analysis.

Completed stages:
- Data cleaning and validation
- Decision Tree model training
- Random Forest model training
- Model comparison and visual analysis
- SQL-based churn queries using SQLite
- PDF export of all SQL results for review

---

## 2. Configuration and Project Files

The centralized configuration is stored in [config.py](config.py). It now covers:
- base directory and output paths
- original and cleaned dataset paths
- machine-learning output locations
- Random Forest and Decision Tree file references
- SQL report export settings
- model training constants such as test size and random state

Key configuration values:
- Target variable: Churn
- Train-test split: 80/20
- Random state: 42
- Output folder: output

---

## 3. Data Cleaning Summary

The cleaned dataset is saved at [output/customer_churn_cleaned.csv](output/customer_churn_cleaned.csv).

### Data quality checks performed
- Duplicate check
- Missing value check
- Outlier capping for critical numerical fields
- Validation of allowed category values
- Removal of non-predictive identifier column

### Final cleaned dataset characteristics
- Total customers: 64,374
- Target column: Churn
- Class distribution: churn and non-churn retained in usable form for model training
- Output audit report: [output/customer_churn_cleaning_audit.csv](output/customer_churn_cleaning_audit.csv)

---

## 4. Machine Learning Models

### 4.1 Decision Tree Model
The Decision Tree workflow is implemented in [03_decision_tree.py](03_decision_tree.py).

Outputs saved in [output/machine_learning](output/machine_learning):
- decision_tree_model.joblib
- decision_tree_metrics.csv
- decision_tree_classification_report.txt
- decision_tree_feature_importance.csv
- decision_tree_rules.txt

### 4.2 Random Forest Model
The Random Forest workflow is implemented in [04_Random Forest Classifier.py](04_Random%20Forest%20Classifier.py).

Outputs saved in [output/machine_learning](output/machine_learning):
- random_forest_model.joblib
- random_forest_metrics.csv
- random_forest_classification_report.txt
- random_forest_feature_importance.csv
- random_forest_confusion_matrix.png

### 4.3 Model Comparison
The comparison script is implemented in [05_model_Evaluation.py](05_model_Evaluation.py).

Saved comparison artifacts:
- [output/model_comparison.csv](output/model_comparison.csv)
- [output/model_comparison_metrics.png](output/model_comparison_metrics.png)
- [output/model_f1_score_comparison.png](output/model_f1_score_comparison.png)

This comparison helps identify which model performs better on the churn dataset using F1-score and ROC-AUC.

---

## 5. SQL Analysis Workflow

The SQL analysis script is implemented in [06_sql_queries.py](06_sql_queries.py).

It uses SQLite to load the cleaned CSV into an in-memory database and run churn questions without requiring a database server.

The project includes 15 SQL queries covering churn patterns such as:
- total customers
- churn vs non-churn counts
- age and tenure comparisons
- subscription and contract analysis
- spend and payment-delay analysis
- support calls and usage trends
- high-risk customer segment detection

All SQL results can be viewed in the terminal or exported as a PDF report.

### SQL PDF report
The generated PDF report is saved at [output/sql_queries_report.pdf](output/sql_queries_report.pdf).

---

## 6. Key Findings

From the completed analysis workflow:
- Contract Length and Subscription Type are major churn drivers.
- Monthly contracts show the highest churn rate.
- Customers with high support calls and short tenure are more likely to churn.
- Longer payment delays are associated with churn risk.
- High-risk segments can be isolated through SQL filters for proactive retention actions.

---

## 7. Final Output Summary

Main project outputs:
- [output/customer_churn_cleaned.csv](output/customer_churn_cleaned.csv)
- [output/customer_churn_cleaning_audit.csv](output/customer_churn_cleaning_audit.csv)
- [output/model_comparison.csv](output/model_comparison.csv)
- [output/model_comparison_metrics.png](output/model_comparison_metrics.png)
- [output/sql_queries_report.pdf](output/sql_queries_report.pdf)

Project status: complete and ready for presentation or further model refinement.

- ✅ No data loss during cleaning
- ✅ All 64,374 records preserved
- ✅ No duplicate records introduced
- ✅ All features in correct numeric format
- ✅ Target variable (Churn) intact and balanced

---

## 7. Recommendations

### 6.1 Immediate Actions

1. **Update pandas deprecation warning** in `ashish_data_cleaning.py`:
   ```python
   # Line 24 - Change from:
   cat_cols = df.select_dtypes(include="object").columns.tolist()
   
   # To:
   cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
   ```

2. **Data validation script** - Consider creating a validation step to verify:
   - Expected column presence
   - Data type correctness
   - Missing value checks
   - Outlier bounds

### 6.2 Next Steps for ML Pipeline

1. **Feature Scaling** - Normalize/standardize numeric features
2. **Train-Test Split** - Divide cleaned data (e.g., 80-20 split)
3. **Class Imbalance Handling** - Address ~47% churn rate with:
   - SMOTE (Synthetic Minority Over-sampling)
   - Class weights in model
   - Stratified sampling
4. **Model Selection** - Consider:
   - Logistic Regression (baseline)
   - Random Forest (interpretability)
   - Gradient Boosting (performance)
   - Neural Networks (complex patterns)

### 6.3 Documentation

- ✅ `config.py` includes self-documentation
- ✅ All cleaning steps logged in script output
- ✅ Configuration summary printable via `python config.py`

---

## 8. File Summary

| File | Status | Purpose |
|------|--------|---------|
| `config.py` | ✅ Created | Configuration & path management |
| `ashish_data_cleaning.py` | ✅ Running | Data cleaning pipeline |
| `01_data_exploration.py` | Ready | Data exploration & reporting |
| `02_data_cleaning.py` | Ready | Alternative cleaning approach |
| `customer_churn_cleaned.csv` | ✅ Generated | Output cleaned dataset |
| `03_decision_tree.py` | ✅ Complete | Decision Tree training and evaluation |
| `output/machine_learning/` | ✅ Generated | Decision Tree model and evaluation outputs |

---

## 9. Conclusion

✅ **All errors resolved.**  
✅ **Configuration file successfully created and tested.**  
✅ **Data cleaning pipeline executing without errors.**  
✅ **Cleaned dataset ready for model training.**  
✅ **Decision Tree classifier trained and evaluated successfully.**

The project is now ready for the next phase: model development and training.

---

**End of Report**
