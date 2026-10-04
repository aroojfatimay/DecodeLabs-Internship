# Credit Card Fraud Detection

## Description

This project is completed as part of the **DecodeLabs Data Science Internship – Project 2**.

The project builds a supervised machine learning pipeline for detecting fraudulent credit card transactions using a highly imbalanced dataset containing **284,807 transactions**.

The project includes class imbalance analysis, SMOTE oversampling, Logistic Regression, Random Forest classification, 5-Fold Cross-Validation, hyperparameter tuning using GridSearchCV, and model evaluation using Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/aroojfatimay/DecodeLabs-Internship.git
```

### 2. Open the project folder

```bash
cd DecodeLabs-Internship/Project-2-Fraud-Detection
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 4. Dataset

Place the `creditcard.csv` dataset inside:

```text
data/creditcard.csv
```

### 5. Open the Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/Project_2_Fraud_Detection.ipynb
```

Run the notebook cells from top to bottom.

### Dataset

The dataset contains:

* **284,807 total transactions**
* **284,315 legitimate transactions**
* **492 fraudulent transactions**

The target column is `Class`, where `0` represents legitimate transactions and `1` represents fraudulent transactions.
