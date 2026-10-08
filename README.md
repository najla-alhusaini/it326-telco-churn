# Telecom Customer Churn Prediction

Predicting which telecom customers will leave (churn) with an interpretable Decision Tree, and segmenting customers with K-Means.
Course project for **IT326 – Data Mining**, King Saud University.

## Highlights

- **Imbalance-aware evaluation.** Only 26.5% of customers churn, so a model that always says "No churn" is already 73.5% accurate. Models are therefore selected on **F1 for the churn class**, with recall, precision, balanced accuracy, ROC-AUC and PR-AUC, and compared against a majority-class baseline.
- **Class-weighted, cross-validated Decision Tree.** `class_weight="balanced"` plus depth/leaf-size tuning with 5-fold stratified CV on the training data only.
- **Interpretable results.** A depth-4 tree whose rules point to concrete retention actions.

## Results

| Model | Accuracy | Recall (Churn) | Precision (Churn) | F1 (Churn) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Always predict "No churn" | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | 0.265 |
| Unpruned Decision Tree (first version) | 0.730 | 0.506 | 0.492 | 0.499 | 0.659 | 0.381 |
| **Class-weighted, tuned Decision Tree** | **0.760** | **0.739** | **0.534** | **0.620** | **0.820** | **0.574** |

*Averages over three train/test partitions (90/10, 80/20, 70/30) × two split criteria (Gini, Entropy). The best single model (Entropy, `max_depth=4`) catches 136 of 187 churners in its test set (F1 = 0.63, ROC-AUC = 0.83).*

**What changed and why.** Our first version ranked models by accuracy. Its trees scored ~73% accuracy, which is slightly *below* the trivial baseline, and missed about half of all churners. Re-evaluating on the churn class exposed this, and class weighting plus pruning raised churn recall from 51% to 74%.

### What drives churn

| Factor | Finding |
|---|---|
| Contract | Month-to-month customers churn at 42.7%, vs 11.3% (one-year) and 2.8% (two-year). The most important feature in the tree (~66% of importance). |
| Online security | Month-to-month customers without online security form the highest-risk branch. |
| Tenure | Risk is concentrated in the first ~10 months. |
| Internet service | Fiber-optic customers churn at 41.9%, vs 19.0% for DSL. |
| Clustering (K = 2) | Isolates customers without internet service, a low-risk segment (7.4% churn vs 31.8%). |

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample data via Kaggle): 7,043 customers, 20 features (demographics, services, account and billing) and the `Churn` label (No: 5,174 / Yes: 1,869).

## Approach

1. **Exploration (Phase 1–2):** distributions, outlier check (IQR), class balance.
2. **Preprocessing (Phase 2):** convert `TotalCharges` to numeric (11 blank values for brand-new customers set to 0), label-encode categorical features, standardise numeric features, drop `customerID`.
3. **Classification (Phase 3):** Decision Tree (Gini vs Entropy) on three partitions; majority-class baseline; class weighting; `GridSearchCV` over `max_depth` and `min_samples_leaf`; evaluation with confusion matrices, ROC and precision–recall curves.
4. **Clustering (Phase 3):** K-Means (K = 2, 3, 4) on one-hot encoded, scaled features; elbow method, silhouette analysis, PCA plots, churn rate per cluster.

## Repository structure

```
├── Dataset/
│   ├── Raw_dataset.csv            # original Kaggle data
│   ├── Preprocessed_dataset.csv   # output of Phase 2, input of Phase 3
│   └── Research paper.pdf.pdf     # reference paper (Ahmad et al., 2019)
├── Reports/
│   ├── Phase1.ipynb               # dataset overview
│   ├── Phase2.ipynb               # summarisation and preprocessing
│   ├── Phase3.ipynb               # full report: classification, clustering, findings
│   └── IT326.pdf                  # course report
└── requirements.txt
```

## Run it

Open any notebook in Colab with the badge at the top, or locally:

```bash
pip install -r requirements.txt
jupyter notebook Reports/Phase3.ipynb
```

The notebooks read the data from `Dataset/` when the repository is cloned and fall back to the GitHub copy otherwise.

## Team Members & Contributions

This project was collaboratively developed by four Information Technology students at King Saud University as part of the IT326 Data Mining course.

All team members contributed throughout the project, including data exploration, preprocessing, classification using Decision Trees, clustering using K-Means, model evaluation, and documentation.

### Team Members
- **Yara Zakzouk**
- **Najla Alhusaini**
- **Latifah Alsaif**
- **Tala Alqahtani**
## Reference

A. K. Ahmad, A. Jafar, and K. Aljoumaa, "Customer churn prediction in telecom using machine learning in big data platform," *Journal of Big Data*, vol. 6, no. 28, 2019.
