# Breast Mass Classification with Logistic Regression
Written by: Lucky (Nhi) Nguyen
## Project Overview

This project investigates binary classification of breast masses using logistic regression. The analysis uses two features from the Wisconsin Diagnostic Breast Cancer dataset:

- `mean radius`
- `mean texture`

The target is coded as:

- `0` = benign (negative)
- `1` = malignant (positive)

The project implements logistic regression in NumPy using full-batch gradient descent and compares the result with a scikit-learn reference implementation. It also evaluates a majority-class baseline and investigates how different classification thresholds affect precision, recall, F1 score, and false-positive/false-negative (FN/FP)trade-offs.

This project is for educational purposes only. It is not a clinical diagnostic system and should not be used to make medical decisions.

---

## Repository Structure

```text
breast-mass-logistic-regression/
│
├── logistic_regression.py
├── memo.md
├── README.md
├── requirements.txt
├── .gitignore
│
└── reports/
    ├── threshold_results.csv
    ├── assignment_results.csv
    └── training_loss.png
```

The .venv directory used locally is a Python virtual environment. So it will not be committed to the repository.

---

## Dataset

The project uses the Wisconsin Diagnostic Breast Cancer dataset provided by scikit-learn through:

```python
from sklearn.datasets import load_breast_cancer
```

The dataset contains 569 observations. Only the following two features are used:

```text
mean radius
mean texture
```

The original target labels are recoded so that:

```text
0 = benign
1 = malignant
```

The overall class counts are:

```text
Benign:     357
Malignant:  212
```

There are no missing values in both 2 selected feature.

---

## Experimental Design

The data are split using row indices in two stages.

First, 20% of the observations are held out as the test set. The remaining 80% form the training set.

The training set is then split into:

- 75% training (60%/80% of the observations)
- 25% validation (20%/80% of the observations)

The resulting sizes are:

| Split | Rows |
|---|---:|
| Training | 341 |
| Validation | 114 |
| Test | 114 |

Both splits use stratification and:

```text
random_state = 42
```

The three index sets are checked to ensure that they are disjoint and together cover all 569 observations.

### Feature Scaling

`StandardScaler` is fitted **only on the training data**.

The same fitted scaler is then used to transform:

- training data
- validation data
- test data

This prevents information from the validation or test sets from leaking into the training procedure.

---

## Models

### 1. NumPy Logistic Regression

The main model is implemented from scratch using NumPy.

The model uses:

- full-batch gradient descent
- learning rate = `0.1`
- epochs = `20,000`
- zero initialization
- no regularization
- no class weights
- a numerically stable sigmoid
- numerically stable binary cross-entropy

The loss is recorded before training and after every gradient update.

The training loss decreased from:

```text
Initial loss: 0.693147
Final loss:   0.246834
```

The resulting coefficients in standardized feature units are:

| Coefficient | Value |
|---|---:|
| Intercept | -0.727172 |
| Mean radius | 3.656794 |
| Mean texture | 1.009872 |

### 2. scikit-learn Reference Model

A scikit-learn logistic regression model is used as a reference implementation with:

```python
LogisticRegression(
    C=np.inf,
    solver="lbfgs",
    max_iter=10000,
    tol=1e-9
)
```

The scikit-learn model receives the two scaled features directly and fits its own intercept.

The NumPy and scikit-learn coefficients match to six decimal places. The mean absolute difference between their validation probabilities is approximately:

```text
1.19 × 10^-10
```

This provides a check that the custom NumPy implementation converged to essentially the same solution as the reference implementation.

### 3. Majority-Class Baseline

A majority/prior baseline is also evaluated.

The baseline predicts the majority class using the class distribution observed in the training data. It provides a simple reference point for determining whether the logistic regression model provides useful predictive performance beyond a trivial classifier.

---

## Threshold Selection

Logistic regression produces probabilities for class 1 (malignant). A classification threshold is then used to convert probabilities into predicted classes.

The following thresholds are evaluated:

```text
0.20
0.30
0.40
0.50
0.60
0.70
0.80
```

The threshold is selected using **validation F1 score only**.

The validation results identify:

```text
Selected threshold = 0.60
Validation F1 = 0.8312
```

The selected threshold is then frozen and applied to the test set. The test set is not used to select or tune the threshold.

---

## Test Results

At the default threshold of `0.50`, the NumPy model achieves:

| Metric | Test result |
|---|---:|
| Accuracy | 0.8684 |
| Precision | 0.8857 |
| Recall | 0.7381 |
| F1 | 0.8052 |
| Log loss | 0.2557 |

The corresponding confusion matrix is:

```text
TN = 68
FP = 4
FN = 11
TP = 31
```

At the selected threshold of `0.60`:

| Metric | Test result |
|---|---:|
| Accuracy | 0.8509 |
| Precision | 0.9032 |
| Recall | 0.6667 |
| F1 | 0.7671 |
| Log loss | 0.2557 |

The confusion matrix becomes:

```text
TN = 69
FP = 3
FN = 14
TP = 28
```

The higher threshold reduces false positives from 4 to 3, but increases false negatives from 11 to 14. This illustrates the trade-off between precision and recall when the classification threshold changes.

The selected threshold was chosen using validation data as required. It is not changed after observing the test results.

---

## Baseline Comparison

The majority baseline achieves approximately:

```text
Test accuracy = 0.6316
```

However, it predicts all test observations as benign. Therefore:

```text
TP = 0
FN = 42
Recall = 0
F1 = 0
```

This demonstrates why accuracy alone is not sufficient for evaluating this classification problem. A model can obtain moderate accuracy by predicting the more common class while completely failing to identify malignant cases.

---

## Output Files

Running the Python script creates the following files inside `reports/`.

### `threshold_results.csv`

Contains the validation results for all seven candidate thresholds.

The file includes:

```text
threshold
tn
fp
fn
tp
accuracy
precision
recall
f1
```

### `assignment_results.csv`

Contains results for five model/policy configurations across validation and test splits:

```text
validation  baseline  default
validation  numpy     default
validation  numpy     selected
validation  sklearn   default
validation  sklearn   selected

test        baseline  default
test        numpy     default
test        numpy     selected
test        sklearn   default
test        sklearn   selected
```

The file contains confusion-matrix counts and:

- accuracy
- precision
- recall
- F1
- log loss

### `training_loss.png`

Shows the NumPy model's training loss over the gradient-descent epochs.

---

# Installation and Running the Project

## Requirements

The project requires:

- Python 3
- NumPy
- Matplotlib
- scikit-learn

The dependency file should be named:

```text
requirements.txt
```

with the required packages listed inside it.

---

## Windows PowerShell

### 1. Clone the repository

```powershell
git clone <https://github.com/luckynnmn/breast-mass-logistic-regression>
```

### 2. Create a virtual environment

In Visual Studio > click "Terminal" > click "New Terminal"

```Visual Studio 
cd breast-mass-logistic-regression
py -m venv logistic
```
logistic (name of venv, can customize on your own)
### 3. Activate the virtual environment

```Visual Studio 
.\logistic\Scripts\Activate.ps1
```

### 4. Install dependencies

```Visual Studio 
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Run the analysis

```
python logistic_regression.py
```

The script should print the Part 1 calculations, data preparation checks, model training results, coefficient comparison, threshold selection results, and evaluation metrics.

It will also generate:

```text
reports/threshold_results.csv
reports/assignment_results.csv
reports/training_loss.png
```

---

## macOS / Linux Terminal

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd breast-mass-logistic-regression
```

### 2. Create a virtual environment

```bash
python3 -m venv logistic
```
logistic (name of venv, can customize on your own)

### 3. Activate the virtual environment

```bash
source logistic/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Run the analysis

```bash
python logistic_regression.py
```

The generated files will appear in:

```text
reports/
```

---

## Reproducibility

The experiment uses fixed random seeds and prescribed model settings so that the results can be reproduced.

The main reproducibility settings are:

```text
random_state = 42
learning_rate = 0.1
epochs = 20000
thresholds = 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80
```

The threshold is selected from validation F1 and then frozen before test evaluation.

The scaler is fitted only on the training partition.

---

## Interpreting the Results

The logistic regression coefficient for `mean radius` is positive:

```text
3.656794
```

Because the features are standardized, this means that, holding `mean texture` constant, a one training-standard-deviation increase in mean radius is associated with an increase of approximately 3.657 in the model's log-odds of malignancy.

This coefficient should not be interpreted as a constant increase in probability or as a causal effect.

The threshold results also show that a higher classification threshold can increase precision while reducing recall. In this experiment, moving from 0.50 to the validation-selected threshold of 0.60 reduced test false positives but increased test false negatives.

---

## Limitations and Scope

This project is an educational investigation rather than a clinical prediction system.

Important limitations include:

- Only two features are used
- The dataset is relatively small and historical;
- External validation is not performed;
- Clinical calibration is not established;
- The threshold was selected using an assignment-defined F1 criterion rather than a clinical cost function.

In a medical classification setting, false negatives (FN) can be particularly consequential because a malignant case classified as benign could delay appropriate follow-up.

Therefore, the model should not be used for clinical diagnosis or treatment decisions.

---

## Source Acknowledgement

The dataset is the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn and accessed through:

```python
sklearn.datasets.load_breast_cancer()
```

The dataset's `DESCR` information was used as the source context for the dataset description.

NumPy was used for the custom logistic-regression training implementation. Scikit-learn was used for dataset loading, train/validation/test splitting, feature scaling, the reference logistic-regression model, the baseline, and evaluation metrics.

The project also documents the experimental process, including leakage prevention, comparison with the reference implementation, threshold selection, and evaluation.

---

## Academic Integrity and Code Reuse

The custom logistic-regression training procedure is implemented using NumPy. Standard library/model utilities from NumPy and scikit-learn are used where appropriate for data loading, splitting, scaling, reference comparison, and evaluation.

Any code reused or adapted from earlier coursework should be acknowledged in the project submission according to the course requirements.