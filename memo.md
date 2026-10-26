# Memo: Breast Mass Classification with Logistic Regression and Decision Thresholds

Educational use only. This is a class project, not a diagnostic tool.

## 1. Part 1: From score to prediction

I used z = beta_0 + beta_1 * x and p = 1 / (1 + exp(-z)). The predicted class is 1 when p >= threshold.

**Case A calculation** (beta_0 = -1.0, beta_1 = 0.8, x = 2.0, threshold = 0.50):

- z = -1.0 + 0.8 * 2.0 = 0.60
- p = 1 / (1 + exp(-0.60)) = 1 / (1 + 0.5488) = 0.6457
- 0.6457 >= 0.50, so the predicted class is 1

| Case | beta_0 | beta_1 | x | Threshold | z | p | Predicted class |
|------|--------|--------|-----|-----------|-------|--------|-----------------|
| A | -1.0 | 0.8 | 2.0 | 0.50 | 0.60 | 0.6457 | 1 |
| B | -1.0 | 0.8 | 2.0 | 0.70 | 0.60 | 0.6457 | 0 |
| C | -1.0 | -0.8 | 2.0 | 0.50 | -2.60 | 0.0691 | 0 |
| D | 1.0 | 0.0 | 2.0 | 0.50 | 1.00 | 0.7311 | 1 |

From A to B, z and p stay the same. Only the threshold changes, so the class changes from 1 to 0. In C, beta_1 is negative, so z goes down and p drops to 0.0691. In D, beta_1 = 0, so z = 1.0 for every x and p is always 0.7311. In Part 1, I set the coefficients by hand. In Part 3, gradient descent learns them from the data.

## 2. Data and leakage

I used the Wisconsin Diagnostic Breast Cancer dataset from `load_breast_cancer()`. The DESCR credits Wolberg, Street, and Mangasarian (UCI Machine Learning Repository). I used only `mean radius` and `mean texture`. I recoded the target so **1 = malignant (positive)** and **0 = benign (negative)**.

There are no missing values. The 569 rows have 357 benign and 212 malignant. The split is 341 train (214 / 127), 114 validation (71 / 43), and 114 test (72 / 42), shown as benign / malignant. I split the row indices first. The script checks that the sets do not overlap and cover all 569 rows. I fit the StandardScaler on the training rows only, so validation and test did not affect training. I only used the test set in Part 5.

## 3. Training and coefficients

Gradient descent used learning rate 0.1 and 20000 epochs, with all coefficients starting at zero. The training loss went from 0.693147 to 0.246834. It dropped fast first (0.3104 at epoch 100), then became flat (0.246834 from epoch 5000).

Cross-entropy is the average negative log of the probability the model gives to the real label. The likelihood is how likely the real labels are under the model. A lower loss means the real labels are more likely, so minimizing it learns good coefficients.

| Coefficient | NumPy | scikit-learn |
|-------------|-----------|--------------|
| Intercept (beta_0) | -0.727172 | -0.727172 |
| Mean radius (beta_1) | 3.656794 | 3.656794 |
| Mean texture (beta_2) | 1.009872 | 1.009872 |

The mean absolute difference of validation probabilities is about 1.19e-10, so the two models are almost the same.

If mean radius goes up by one training standard deviation (about 3.59), the log-odds of malignant goes up by about 3.66, holding mean texture fixed. This is a change in log-odds. It is not a constant change in probability, and it does not mean radius causes cancer.

## 4. Baseline vs logistic regression

The baseline learned P(malignant) = 0.372, so at 0.50 it always predicts benign. Its test accuracy is 0.6316, but TP = 0 and FN = 42, so it finds no malignant samples. Precision, recall, and F1 are 0 because I used zero_division = 0 (the baseline never predicts class 1, so precision divides by zero). Accuracy alone is not enough. The logistic models have test accuracy 0.8684 and they find malignant samples.

**NumPy at 0.50, test set** (rows = actual, columns = predicted):

| | Predicted benign (0) | Predicted malignant (1) |
|---|---|---|
| **Actual benign (0)** | TN = 68 | FP = 4 |
| **Actual malignant (1)** | FN = 11 | TP = 31 |

Of 72 benign samples, 68 were correct and 4 were called malignant. Of 42 malignant samples, 31 were found and 11 were missed.

## 5. Threshold selection

On validation, the highest F1 was 0.8312 at **0.60**, so I selected it and used it for both models.

| Test result (NumPy) | FP | FN | Precision | Recall | F1 |
|---------------------|----|----|-----------|--------|-------|
| Default (0.50) | 4 | 11 | 0.8857 | 0.7381 | 0.8052 |
| Selected (0.60) | 3 | 14 | 0.9032 | 0.6667 | 0.7671 |

On the test set, 0.60 had one fewer FP but three more FN. Precision went up a little, but recall and F1 went down. I did not change the threshold after seeing this. The threshold only turns p into a class. It does not change the probabilities, so log loss stays the same (0.2557 on test).

## 6. Not ready for clinical use

A false negative (FN) means a malignant sample is predicted as benign. This could delay treatment, which is serious. F1 treats FP and FN the same, but in real life they are not equal. Other limitations: only two features, a small dataset (the test set has 114 rows), no check of calibration, and no external validation.

## 7. Research process

**Design decision: how to calculate the sigmoid and the loss.** The assignment suggested `exp(-logaddexp(0, -z))` for the sigmoid and `mean(logaddexp(0, z) - y*z)` for the loss. I used both. This way the code never takes the log of zero, even if p gets very close to 0 or 1. I used the same loss function for the whole training, so the 20001 loss values are all calculated the same way.
 
**How I checked it.**
 
- Before training, all coefficients are 0, so p = 0.5 for every single row. The initial loss was 0.693147, which is log 2. This is what I expected.
- The loss went down to 0.246834 and stayed flat after about epoch 5000, so I did not need to change the learning rate or the number of epochs.
- I did not copy the scikit-learn coefficients. I trained my NumPy model first, then compared. The three coefficients match to 6 decimals (-0.727172, 3.656794, 1.009872). The mean difference of validation probabilities was about 1.19e-10.
- The script checks that the train, validation, and test index sets do not overlap and cover all 569 rows. It printed "Index sets are disjoint and cover all 569 rows."
**Choosing the threshold.** I used the validation set only. The best F1 was 0.8312 at 0.60. The next best was 0.8276 at 0.40, so there was no tie. After that, I only used the test set once, in the final evaluation. I did not change the threshold after seeing that 0.60 did worse than 0.50 on test.
 
**How I ran it.** I ran the script in Terminal inside a virtual environment `logistic` (can adjust the venv name on your own). The output printed all five parts, and the three files were saved in the `reports` folder.
 
**Challenges I really had:** I forgot to import `matplotlib` at the beginning, so the python file `python logistic_regression.py` showed the error.
 
**Sources and reused code.** Assignment instructions (MSBA265, University of the Pacific); scikit-learn and NumPy documentation. My gradient descent loop follows the pattern from Project 1. 