# Market Basket Analysis with Machine Learning

[![Open Notebook 1 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogitaalone04-web/Market-Basket-Analysis-ML/blob/main/Market_Basket_Analysis_FINAL.ipynb)
[![Open Notebook 2 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogitaalone04-web/Market-Basket-Analysis-ML/blob/main/Market_Basket_Analysis_LR_DT_RF.ipynb)

Market Basket Analysis is a data mining technique used to discover purchasing patterns in transactional data. This project applies **Apriori association rule mining** alongside **six classification models** on a real-world groceries dataset to predict the next item a customer is likely to purchase.

---

## Table of Contents

- [Dataset](#dataset)
- [Notebooks](#notebooks)
- [Project Pipeline](#project-pipeline)
- [Models](#models)
- [Results](#results)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Authors](#authors)

---

## Dataset

| Property | Details |
|---|---|
| Name | Groceries Dataset for Machine Learning |
| Format | CSV — comma-separated items per transaction row |
| Drive Path | '/content/drive/MyDrive/ML_Dataset/groceries.csv' |

---

## Notebooks

| Notebook | Models |
|---|---|
| `Market_Basket_Analysis_FINAL.ipynb` | KNN · Naive Bayes · SVM |
| `Market_Basket_Analysis_LR_DT_RF.ipynb` | Logistic Regression · Decision Tree · Random Forest |

Both notebooks share the same EDA and Apriori pipeline and can be run independently.

---

## Project Pipeline

```
Raw CSV
   │
   ▼
Parse & Build Transactions
   │
   ▼
Exploratory Data Analysis
   ├── Top 20 most-purchased items
   └── Transaction size distribution
   │
   ▼
One-Hot Encode (TransactionEncoder)
   │
   ├──► Apriori → Association Rules (support, confidence, lift)
   │
   └──► Classification (last item as label)
           ├── Notebook 1: KNN | Naive Bayes | SVM
           └── Notebook 2: Logistic Regression | Decision Tree | Random Forest
                   │
                   ▼
        Both splits: 80:20 and 70:30
        + GridSearchCV tuning (Notebook 2)
                   │
                   ▼
        Comparison Table & Bar Charts
                   │
                   ▼
        Next-Item Prediction + Rule-Based Recommendations
```

---

## Models

### Apriori — Association Rules

| Parameter | Value |
|---|---|
| `min_support` | 0.01 |
| `metric` | lift |
| `min_threshold` | 1.0 |

### Notebook 1

| Model | Configuration |
|---|---|
| K-Nearest Neighbours | k=5, Euclidean distance |
| Naive Bayes | Gaussian NB |
| Support Vector Machine | RBF kernel, C=1.0, gamma='scale' |

### Notebook 2 (with GridSearchCV)

| Model | Configuration | Tuned Parameters |
|---|---|---|
| Logistic Regression | max_iter=1000 | C · penalty · solver |
| Decision Tree | max_leaf_nodes=10 | max_depth · min_samples_leaf · criterion |
| Random Forest | n_estimators=100, max_depth=5, oob_score=True | max_depth · min_samples_leaf · n_estimators |

---

## Results

### 70/30 Train-Test Split

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 0.13 | 0.04 | 0.13 | 0.03 |
| KNN | 0.06 | 0.10 | 0.06 | 0.06 |
| SVM | 0.16 | 0.08 | 0.16 | 0.08 |
| Decision Tree | 0.12 | 0.05 | 0.12 | 0.06 |
| Logistic Regression | 0.16 | 0.10 | 0.16 | 0.11 |
| Naive Bayes | 0.16 | 0.08 | 0.16 | 0.08 |

### 80/20 Train-Test Split

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 0.13 | 0.03 | 0.13 | 0.03 |
| KNN | 0.05 | 0.08 | 0.05 | 0.05 |
| SVM | 0.14 | 0.06 | 0.14 | 0.07 |
| Decision Tree | 0.13 | 0.03 | 0.13 | 0.03 |
| Logistic Regression | 0.13 | 0.03 | 0.13 | 0.03 |
| Naive Bayes | 0.02 | 0.18 | 0.02 | 0.03 |

**Key Observations:**
- SVM and Logistic Regression achieved the highest accuracy (0.16) on the 70/30 split.
- Naive Bayes showed the highest precision (0.18) on the 80/20 split despite low accuracy.
- The multi-class nature of the groceries dataset (100+ unique items as labels) leads to low overall accuracy across all models — this is expected and consistent with literature on high-cardinality item prediction tasks.

---

## Getting Started

### Option 1 — Google Colab (Recommended)

1. Upload `groceries.csv` to Google Drive at `MyDrive/ML_Dataset/groceries.csv`
2. Open either notebook via the Colab badges at the top of this README
3. Run all cells — **Runtime → Run all**

### Option 2 — Run Locally

```bash
# Clone the repository
git clone https://github.com/yogitaalone04-web/Market-Basket-Analysis-ML.git
cd Market-Basket-Analysis-ML

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend

# Launch Jupyter
jupyter notebook
```

> Update `FILE_PATH` in each notebook to your local CSV path before running.

---

## Project Structure

```
Market-Basket-Analysis-ML/
│
├── Market_Basket_Analysis_FINAL.ipynb        # KNN, Naive Bayes, SVM
├── Market_Basket_Analysis_LR_DT_RF.ipynb     # Logistic Regression, Decision Tree, Random Forest
├── README.md
│
└── outputs/
    ├── top20_items.png
    ├── transaction_size.png
    ├── rules_scatter.png
    ├── cm_*.png
    ├── model_comparison.png
    ├── model_results.csv
    └── association_rules.csv
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data Manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Association Mining | mlxtend — Apriori, TransactionEncoder, association_rules |
| Machine Learning | scikit-learn — KNN, GaussianNB, SVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GridSearchCV |
| Environment | Google Colab / Jupyter Notebook |

---

## Authors

**Yogita Alone** — [@yogitaalone04-web](https://github.com/yogitaalone04-web)

**Shinam Sheikh** — USN: CS24D010

Repository: [Market-Basket-Analysis-ML](https://github.com/yogitaalone04-web/Market-Basket-Analysis-ML)
