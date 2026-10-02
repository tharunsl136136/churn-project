# Literature Review Reading List
## Telco Customer Churn Prediction — Search Guide

> You will read and write the review yourself. This list gives you the exact
> search terms, paper titles, and the specific argument each paper supports.
> Organised by the section of your report where each source belongs.

---

## HOW TO FIND THESE PAPERS

**Free sources (no paywall):**
- Google Scholar: scholar.google.com
- Semantic Scholar: semanticscholar.org
- arXiv: arxiv.org (for ML papers)
- ResearchGate: researchgate.net
- PubMed (for statistics papers)

**Search tip:** Use the exact title in quotes for a direct hit.

---

## SECTION 1 — Problem Statement & Business Motivation

### 1.1 Customer Lifetime Value & Churn Cost
**Search:** "customer churn telecom cost retention"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Gallo, A. (2014). "The value of keeping the right customers." *Harvard Business Review*. | hbr.org (free) | "Acquiring a new customer is 5–25× more expensive than retaining an existing one" |
| Reichheld, F. F., & Sasser, W. E. (1990). "Zero defections: Quality comes to services." *HBR*, 68(5), 105–111. | Google Scholar | 5% increase in retention → 25–95% profit increase |
| Verbeke, W., et al. (2012). "New insights into churn prediction in the telecommunication sector." *European Journal of Operational Research*, 218(1), 211–229. | Sci-Hub / ResearchGate | Telecom-specific churn modelling; cost-sensitive evaluation |

---

## SECTION 2 — Dataset & EDA

### 2.1 IBM Telco Dataset
**Search:** "telco customer churn IBM Watson dataset"

| Paper/Source | Where to find | What to quote |
|-------------|--------------|---------------|
| IBM Watson Analytics (2017). Telco Customer Churn dataset. Kaggle. | kaggle.com/datasets/blastchar/telco-customer-churn | Dataset description, 7,043 rows, 20 features |
| Amin, A., et al. (2019). "Customer churn prediction in the telecommunication sector using a rough set approach." *Neurocomputing*, 237, 242–254. | ScienceDirect | Feature importance in telecom churn; contract type dominates |

### 2.2 Class Imbalance in Churn
**Search:** "class imbalance churn prediction machine learning"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| He, H., & Garcia, E. A. (2009). "Learning from imbalanced data." *IEEE TKDE*, 21(9), 1263–1284. | IEEE Xplore | Comprehensive survey of imbalance strategies |
| Chawla, N. V., et al. (2002). "SMOTE: Synthetic minority over-sampling technique." *JAIR*, 16, 321–357. | jair.org (free) | SMOTE algorithm; why oversampling helps |

---

## SECTION 3 — Feature Engineering

### 3.1 Domain-Driven Feature Engineering
**Search:** "feature engineering customer churn telecom"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Huang, B., et al. (2012). "Customer churn prediction in telecommunications." *Expert Systems with Applications*, 39(1), 1414–1425. | ScienceDirect | Tenure, contract type, and charges as top predictors |
| Coussement, K., & Van den Poel, D. (2008). "Churn prediction in subscription services." *European Journal of Operational Research*, 188(2), 472–486. | ScienceDirect | Behavioural features outperform demographic features |

### 3.2 Interaction Features
**Search:** "interaction features gradient boosting tabular"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Natekin, A., & Knoll, A. (2013). "Gradient boosting machines, a tutorial." *Frontiers in Neurorobotics*, 7, 21. | frontiersin.org (free) | How boosting captures feature interactions automatically |

---

## SECTION 4 — Model Selection

### 4.1 Gradient Boosting (Primary Algorithm)
**Search:** "XGBoost gradient boosting tabular classification"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *KDD*, 785–794. | arxiv.org/abs/1603.02754 (free) | XGBoost wins on 10/29 Kaggle tabular benchmarks |
| Ke, G., et al. (2017). "LightGBM: A highly efficient gradient boosting decision tree." *NeurIPS*, 30. | papers.nips.cc (free) | Leaf-wise growth; 20× faster than XGBoost on large data |
| Prokhorenkova, L., et al. (2018). "CatBoost: Unbiased boosting with categorical features." *NeurIPS*, 31. | arxiv.org/abs/1706.09516 (free) | Ordered boosting; native categorical handling |

### 4.2 Ensemble Methods
**Search:** "stacking ensemble classification survey"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Breiman, L. (1996). "Stacked regressions." *Machine Learning*, 24(1), 49–64. | Springer | Original stacking paper |
| Breiman, L. (2001). "Random forests." *Machine Learning*, 45(1), 5–32. | Springer | Bagging ensemble; feature importance via impurity |

### 4.3 Why Not Neural Networks for Small Tabular Data
**Search:** "deep learning tabular data comparison gradient boosting"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Grinsztajn, L., et al. (2022). "Why tree-based models still outperform deep learning on tabular data." *NeurIPS*, 35. | arxiv.org/abs/2207.08815 (free) | Tree models beat neural nets on 45 tabular benchmarks |
| Shwartz-Ziv, R., & Armon, A. (2022). "Tabular data: Deep learning is not all you need." *Information Fusion*, 81, 84–90. | arxiv.org/abs/2106.03253 (free) | XGBoost competitive with or better than deep learning |

---

## SECTION 5 — Evaluation Metrics

### 5.1 PR-AUC vs ROC-AUC for Imbalanced Data
**Search:** "precision recall AUC imbalanced classification"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Davis, J., & Goadrich, M. (2006). "The relationship between Precision-Recall and ROC curves." *ICML*, 233–240. | dl.acm.org | "PR curves give a more informative picture of an algorithm's performance on imbalanced datasets" |
| Saito, T., & Rehmsmeier, M. (2015). "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets." *PLOS ONE*, 10(3). | plosone.org (free) | Empirical proof across 100 datasets |

### 5.2 F-beta Score & Cost-Sensitive Evaluation
**Search:** "F-beta score cost-sensitive learning churn"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Elkan, C. (2001). "The foundations of cost-sensitive learning." *IJCAI*, 973–978. | ijcai.org (free) | Theoretical basis for asymmetric cost classification |
| Verbeke, W., et al. (2012). (same as Section 1) | — | Maximum profit criterion for churn; FN >> FP cost |

### 5.3 MCC (Matthews Correlation Coefficient)
**Search:** "Matthews correlation coefficient imbalanced classification"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Chicco, D., & Jurman, G. (2020). "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation." *BMC Genomics*, 21(1), 1–13. | biomedcentral.com (free) | MCC is the most informative single metric for binary classification |

### 5.4 Statistical Comparison of Classifiers
**Search:** "McNemar test classifier comparison machine learning"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions." *Psychometrika*, 12(2), 153–157. | Springer | Original McNemar test |
| Dietterich, T. G. (1998). "Approximate statistical tests for comparing supervised classification learning algorithms." *Neural Computation*, 10(7), 1895–1923. | MIT Press | McNemar recommended for comparing two classifiers on one test set |

---

## SECTION 6 — Hyperparameter Optimisation

### 6.1 Bayesian Optimisation / Optuna
**Search:** "Bayesian hyperparameter optimisation TPE"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Bergstra, J., et al. (2011). "Algorithms for hyper-parameter optimization." *NeurIPS*, 24. | papers.nips.cc (free) | TPE (Tree-structured Parzen Estimator) algorithm |
| Akiba, T., et al. (2019). "Optuna: A next-generation hyperparameter optimization framework." *KDD*, 2623–2631. | arxiv.org/abs/1907.10902 (free) | Optuna framework; define-by-run API |

---

## SECTION 7 — Explainability (SHAP)

### 7.1 SHAP Theory
**Search:** "SHAP Shapley values machine learning interpretability"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *NeurIPS*, 30. | arxiv.org/abs/1705.07874 (free) | SHAP is the UNIQUE additive attribution satisfying local accuracy, missingness, consistency |
| Lundberg, S. M., et al. (2020). "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence*, 2(1), 56–67. | nature.com | TreeExplainer: exact SHAP in O(TLD²); SHAP interaction values |

### 7.2 Interpretability vs Accuracy Trade-off
**Search:** "interpretable machine learning churn prediction"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Ribeiro, M. T., et al. (2016). "Why should I trust you? Explaining the predictions of any classifier." *KDD*, 1135–1144. | arxiv.org/abs/1602.04938 (free) | LIME — local approximation (compare/contrast with SHAP) |
| Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.). | christophm.github.io/interpretable-ml-book/ (free online) | Comprehensive textbook; cite for SHAP, PDP, permutation importance |

---

## SECTION 8 — Error Analysis & Fairness

### 8.1 Fairness in ML
**Search:** "algorithmic fairness machine learning classification"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning*. MIT Press. | fairmlbook.org (free online) | Definitions: demographic parity, equalised odds, calibration |
| Hardt, M., et al. (2016). "Equality of opportunity in supervised learning." *NeurIPS*, 29. | arxiv.org/abs/1610.02413 (free) | Equalised odds: equal TPR and FPR across groups |

### 8.2 Learning Curves & Model Capacity
**Search:** "learning curve bias variance machine learning"

| Paper | Where to find | What to quote |
|-------|--------------|---------------|
| Ng, A. (2012). "Advice for applying machine learning." *CS229 Lecture Notes*. | cs229.stanford.edu (free) | Learning curves diagnose bias vs variance |

---

## QUICK REFERENCE — Minimum Citations Per Section

For a strong literature review, aim for:

| Report Section | Minimum Citations | Key Papers |
|----------------|------------------|------------|
| Problem statement | 2 | Gallo (2014), Verbeke (2012) |
| EDA / dataset | 2 | IBM Kaggle, Amin (2019) |
| Feature engineering | 2 | Huang (2012), Coussement (2008) |
| Model selection | 4 | Chen (2016), Ke (2017), Breiman (2001), Grinsztajn (2022) |
| Metrics | 3 | Davis (2006), Saito (2015), Chicco (2020) |
| Tuning | 2 | Bergstra (2011), Akiba (2019) |
| Explainability | 2 | Lundberg (2017), Lundberg (2020) |
| Error/fairness | 2 | Barocas (2023), Hardt (2016) |
| **Total** | **~17** | |

---

## WRITING TIPS

1. **Don't summarise papers** — explain why each choice is justified by the literature.
   - Bad: "Chen & Guestrin (2016) proposed XGBoost."
   - Good: "XGBoost was selected because Chen & Guestrin (2016) demonstrated it wins on 10/29 Kaggle tabular benchmarks, and its `scale_pos_weight` parameter directly addresses the 26.54% class imbalance in our dataset."

2. **Connect to your results** — every citation should link to a number you measured.
   - "Davis & Goadrich (2006) showed PR-AUC is more informative than ROC-AUC under imbalance; our results confirm this — the Dummy classifier achieves ROC-AUC=0.50 but PR-AUC=0.265, correctly reflecting its uselessness."

3. **Cite your own cost model** — the FN=$500/FP=$50 assumption is yours; justify it with domain knowledge and cite Elkan (2001) for the theoretical framework.

4. **SHAP vs LIME** — mention LIME (Ribeiro, 2016) to show you know alternatives, then justify SHAP: "Unlike LIME's local linear approximation, SHAP provides theoretically grounded exact values satisfying three desirable axioms (Lundberg & Lee, 2017)."
