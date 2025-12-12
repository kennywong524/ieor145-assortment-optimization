# Featurized Choice Models for "Top-K" Restaurant Assortment Optimization

> **⚠️ Note: This repository is actively under development. Changes are still ongoing.**

A comparative study of Featurized Multinomial Logit (MNL) with Traditional Machine Learning (ML) algorithms for assortment optimization in online marketplaces.

**Authors:** Rishika Gorai, Ryan Gu, Ashlee Liu, Kenny Wongchamcharoen  
**Course:** IEOR 145 - Fall 2025  
**Report:** [IEOR_145_Final_Project.pdf](IEOR_145_Final_Project.pdf)

---

## Project Overview

This project investigates the trade-off between predictive accuracy and economic efficiency in assortment optimization for on-demand food delivery platforms. We compare:

1. **Structural Revenue Management Approach**: Featurized MNL with Greedy Optimization
2. **Reduced-Form ML Baselines**: Random Forest and XGBoost

Our simulation environment models a capacity-constrained marketplace with 100 restaurants and 10 latent user profiles, evaluating performance through counterfactual analysis.

## Repository Structure

```
.
├── README.md
├── 145-groundtruth-generation.ipynb  # Ground truth transaction data generation
├── 145-featurized-MNL-rev.ipynb      # MNL pipeline with greedy optimization
├── 145-ml-pipeline.ipynb              # ML pipeline (Random Forest & XGBoost)
├── randomforest.ipynb                 # Weighted Random Forest with price-based penalties
├── xgb2.ipynb                          # Weighted XGBoost with price-based penalties
├── data/
│   ├── berkeley_real_restaurants_100.csv
│   └── groundtruth_transaction_data.csv
└── IEOR_145_Final_Project.pdf        # Full project report
```

## Methodology

### Data Generation
- **Restaurant Universe**: 100 Berkeley restaurants with features (cuisine, price level, rating, location)
- **User Profiles**: 10 behavioral profiles (Budget Shopper, Speed-Obsessed, Rating Snob, etc.)
- **Transaction Simulation**: Deterministic Ranked Choice Model (RCM) with 5-restaurant offer sets

### Models

#### 1. Featurized MNL
- Estimates β parameters via maximum likelihood
- Features: Cuisine Match, Price Gap, Rating, ETA, Intercept
- Optimization: Greedy revenue maximization (Top-K = 5)

#### 2. Machine Learning Baselines

**Standard ML Models** (`145-ml-pipeline.ipynb`):
- **Random Forest**: Binary classification (purchase probability)
- **XGBoost**: Gradient boosting with hyperparameter tuning
- Optimization: Independent ranking heuristic (Score = P(buy) × Price)

**Weighted ML Models** (Revenue-Optimized):
- **Weighted Random Forest** (`randomforest.ipynb`): Random Forest with price-based sample weighting
  - Uses sample weights proportional to `PriceBase^penalty_exponent` to bias learning toward higher-revenue items
  - Tunes penalty exponent via grid search to maximize revenue on validation set
  - Achieves improved revenue extraction compared to standard RF
  
- **Weighted XGBoost** (`xgb2.ipynb`): XGBoost with price-based sample weighting
  - Similar price-based weighting approach with hyperparameter tuning
  - Combines AUC optimization with revenue-focused sample weighting
  - Balances predictive accuracy and revenue maximization

### Evaluation Metrics
- **Average Revenue per User**: Primary monetization metric
- **Global Hit Rate**: % of users where true favorite is in top-K
- **Average True Utility**: User satisfaction measure
- **Revenue Gap**: Upsell analysis (Actual Revenue - Favorite Item Revenue)

## Getting Started

### Prerequisites
```python
numpy
pandas
scipy
scikit-learn
xgboost
folium  # For visualization
matplotlib
```

### Running the Notebooks

**Note**: The datasets in `data/` are already generated. You can run the notebooks below directly. If you need to regenerate the data, run the ground truth generation notebook first.

1. **Ground Truth Generation** (`145-groundtruth-generation.ipynb`):
   - Generates synthetic transaction data using 10 customer profiles
   - Creates deterministic Ranked Choice Model (RCM) transactions
   - Outputs `groundtruth_transaction_data.csv` (already provided)

2. **MNL Pipeline** (`145-featurized-MNL-rev.ipynb`):
   - Loads transaction and restaurant data
   - Estimates MNL parameters
   - Optimizes assortments using greedy algorithm
   - Evaluates against ground truth oracle

3. **ML Pipeline** (`145-ml-pipeline.ipynb`):
   - Trains standard Random Forest/XGBoost classifiers
   - Generates purchase probability predictions
   - Optimizes assortments using independent ranking
   - Compares performance with MNL baseline

4. **Weighted Random Forest** (`randomforest.ipynb`):
   - Implements price-based sample weighting for revenue optimization
   - Trains baseline RF (no penalty) and penalized RF models with varying penalty exponents
   - Selects best model based on revenue performance on tuning set
   - Evaluates hit rate, utility, and revenue gap metrics

5. **Weighted XGBoost** (`xgb2.ipynb`):
   - Implements price-based sample weighting with XGBoost
   - Performs hyperparameter tuning (max_depth, n_estimators, learning_rate, etc.)
   - Tunes penalty exponent via grid search to optimize validation AUC
   - Evaluates comprehensive performance metrics including profile-level breakdowns

### Data Requirements
- Ensure `data/` folder is in the same directory as notebooks
- CSV files should be in the `data/` subdirectory

## References

- Talluri, K. T., & van Ryzin, G. J. (2004). *The Theory and Practice of Revenue Management*
- Rusmevichientong, P., et al. (2010). *A Nonparametric Approach to Multiproduct Pricing*

## License

This project is part of an undergraduate course (IEOR 145: Fundamentals of Revenue Management) at UC Berkeley.

---

**Note**: This repository contains all implementation notebooks and datasets referenced in our final project report. The ground truth generation notebook is included for reproducibility, though the final datasets are already provided in `data/`.

