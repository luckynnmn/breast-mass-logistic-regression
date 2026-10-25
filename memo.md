# Breast Mass Classification: Logistic Regression and Decision Thresholds

## Part 1. From Scores to Decisions

Logistic regression converts a linear score into a probability and then into a binary decision. Class 1 is malignant (positive) and class 0 is benign (negative).

| Case | β₀ | β₁ | x | Score z | Probability p | Threshold | Prediction |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.0 | 0.8 | 2.0 | 0.60 | 0.6457 | 0.50 | 1 |
| B | -1.0 | 0.8 | 2.0 | 0.60 | 0.6457 | 0.70 | 0 |
| C | -1.0 | -0.8 | 2.0 | -2.60 | 0.0691 | 0.50 | 0 |
| D | 1.0 | 0.0 | 2.0 | 1.00 | 0.7311 | 0.50 | 1 |

For case A, `z = -1.0 + 0.8(2.0) = 0.60`, and `p = 1/(1 + exp(-0.60)) = 0.6457`. Since `0.6457 >= 0.50`, the prediction is 1 (malignant). Case B has the same probability, but its higher threshold changes the prediction to 0. Reversing the coefficient in case C reverses the direction of the score and produces a much lower probability. In case D, the input coefficient is zero, so the probability remains constant for every x. These are manually specified examples, not learned parameters.

## 2. Dataset and Experimental Design

The Wisconsin Diagnostic Breast Cancer dataset was loaded from scikit-learn using `load_breast_cancer()`. Only `mean radius` and `mean texture` were used. The original labels were recoded so that 1 = malignant and 0 = benign. There were 569 observations, with 357 benign and 212 malignant cases. Neither selected feature had missing values.

The two-stage stratified index split produced 341 training rows, 114 validation rows, and 114 test rows. The index sets were verified to be disjoint and to cover all 569 observations. `StandardScaler` was fitted only on the training features and then used to transform validation and test data. This prevents held-out information from affecting scaling parameters.

## 3. NumPy Logistic Regression

The NumPy model used full-batch gradient descent with learning rate 0.1, 20,000 epochs, zero initialization, and no regularization. Training loss decreased from 0.693147 to 0.246834.

| Coefficient | NumPy | scikit-learn |
|---|---:|---:|
| Intercept | -0.727172 | -0.727172 |
| Mean radius | 3.656794 | 3.656794 |
| Mean texture | 1.009872 | 1.009872 |

Cross-entropy is the negative average log-likelihood for the observed binary labels, so minimizing cross-entropy corresponds to maximizing the likelihood of the observed labels. The NumPy and scikit-learn coefficients match to six decimal places. Their validation probability difference was only `1.19 × 10^-10`, providing strong evidence that the custom implementation converged to the same unregularized solution.

The mean-radius coefficient means that, holding mean texture constant, a one training-standard-deviation increase in mean radius increases the model's log-odds of malignancy by about 3.657. This is not a probability change or causal effect.

## 4. Baseline and Classification Results

The majority baseline demonstrates why accuracy alone is insufficient. On the test set it achieved 0.6316 accuracy while predicting every case as benign, producing 42 false negatives and zero true positives. Its malignant recall and F1 were both 0.

At 0.50, the NumPy model achieved test accuracy 0.8684, precision 0.8857, recall 0.7381, and F1 0.8052. Its confusion matrix was TN=68, FP=4, FN=11, and TP=31. It correctly identified 31 malignant cases but missed 11, while incorrectly labeling 4 benign cases as malignant.

## 5. Threshold Selection

Threshold selection used only the NumPy model's validation probabilities. Of the required thresholds from 0.20 to 0.80, 0.60 produced the highest validation F1, 0.8312, compared with 0.8193 at 0.50. Therefore, 0.60 was frozen for final evaluation.

On the test set, 0.60 did not improve performance over 0.50. Precision increased from 0.8857 to 0.9032, but recall decreased from 0.7381 to 0.6667 and F1 decreased from 0.8052 to 0.7671. False positives decreased from 4 to 3, while false negatives increased from 11 to 14. This demonstrates the threshold trade-off between false positives and false negatives. Log loss stayed at 0.2557 because thresholding changes predicted classes, not the underlying probabilities used to calculate log loss.

## 6. Limitations and Nonclinical Scope

This analysis is educational only and is not a diagnostic tool or evidence of clinical readiness. The model uses only two features, excluding other measurements that may contain useful information. The dataset is small and historical, and this experiment does not establish external validation or clinical calibration. False negatives matter because a malignant sample classified as benign could delay follow-up. The F1 threshold rule is an assignment requirement, not a clinical cost function.

## Research Process

A key design decision was to keep the experiment leakage-free. I verified the required split sequence, confirmed that the three index sets were disjoint and covered all 569 observations, and checked that `StandardScaler` was fitted only on the training data. I also independently compared the NumPy implementation with the required scikit-learn reference model. The coefficients matched to six decimal places and the validation probability difference was approximately `1.19 × 10^-10`, confirming convergence. Finally, I checked all seven validation F1 scores before freezing the selected threshold at 0.60 and evaluating it on the test set.

## Source Acknowledgement

The dataset source is the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn and loaded with `load_breast_cancer()`. The dataset `DESCR` was used for source context. NumPy was used for custom training; scikit-learn was used for loading, splitting, scaling, the reference model, baseline, and metrics.