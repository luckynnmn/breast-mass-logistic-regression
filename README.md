# Breast Mass Classification with Logistic Regression
Written by: Lucky (Nhi) Nguyen
**Educational use only.** This is a class project for MSBA265 at the University of the Pacific. The model uses only two features and a small, historical dataset. It is **not** a diagnostic tool and it does not give advice about any patient.
 
Full report: **[memo.md](memo.md)**

## 1. Project overview
 
**Question.** Can two cell measurements (`mean radius` and `mean texture`) predict if a breast-mass sample is malignant or benign? And what happens to the mistakes (false positives and false negatives) when we change the threshold?
 
**Data.** The Wisconsin Diagnostic Breast Cancer dataset, loaded with `sklearn.datasets.load_breast_cancer()`. It comes with scikit-learn, so there is **nothing to download**. The data is from the UCI Machine Learning Repository (Wolberg, Street, and Mangasarian). It has 569 rows.
 
**Positive class.** I recoded the target so that **1 = malignant (positive)** and **0 = benign (negative)**.
 
**What I compared.**
 
1. A logistic regression that I wrote with NumPy (gradient descent, mean binary cross-entropy).
2. A scikit-learn LogisticRegression (reference).
3. A majority baseline (DummyClassifier).
**Main findings** (details in the memo):
 
- My NumPy model and scikit-learn give the same coefficients (to 6 decimals).
- The baseline has a test accuracy of 0.6316, but it finds 0 malignant samples.
- The threshold chosen on validation was 0.60. On the test set, it had fewer false positives (3 vs 4) but more false negatives (14 vs 11), so recall and F1 went down.
**Limitations.** Only two features, a small dataset, no external validation, and I did not check calibration. See the memo, section 6.
 
---
 
## 2. What you need before you start
 
| Software | Link | Note |
|----------|------|------|
| Python 3.10 or newer | https://www.python.org/downloads/ | On Windows, check **"Add Python to PATH"** in the installer. |
| Git | https://git-scm.com/downloads | Used to download (clone) this project. |
 
**Check that they are installed.** Open a terminal (see below) and type these commands.
 
- Windows PowerShell: `python --version` and `git --version`
- macOS Terminal: `python3 --version` and `git --version`
You should see a version number such as `Python 3.12.x`. If you see an error, go to the Troubleshooting section.
 
**How to open a terminal**
 
- **Windows:** Press the Windows key, type `PowerShell`, and press Enter.
- **macOS:** Press Command + Space, type `Terminal`, and press Enter.
---
 
## 3. Download the project
 
The link for this project is:
 
```
https://github.com/luckynnmn/breast-mass-logistic-regression
```
 
Run these commands (same on Windows and macOS):
 
```
git clone https://github.com/luckynnmn/breast-mass-logistic-regression
cd breast-mass-logistic-regression
```
 
The second line moves you into the project folder. If you cloned into a different folder name, use that name after `cd`. The folder should contain `logistic_regression.py`, `requirements.txt`, and `README.md`. You can check with `dir` (Windows) or `ls` (macOS).
 
All commands below must be run **from this project folder**.
 
---
 
## 4. Set up and run
 
### Windows (PowerShell) or Open 'Visual Studio Code' > Click 'Terminal' > Click 'New Terminal'
 
```powershell/Terminal
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python logistic_regression.py
```
 
**If activation is blocked** (error says "running scripts is disabled"), run this command and then try the activation line again. It only changes the current PowerShell window, not your computer permanently:
 
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
 
When the environment is active, you see `(.venv)` at the start of the line.
 
### macOS (Terminal)
 
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python logistic_regression.py
```
 
### Run details
 
- **Script name:** `logistic_regression.py`
- **Working directory:** the project folder (the one that contains the script)
- **Training settings:** learning rate = 0.1, epochs = 20000, as suggested in the assignment (no changes)
- **Dataset:** loaded from scikit-learn, no separate download needed
- The script saves outputs next to itself, so it also works if you run it from another folder (for example `python path/to/logistic_regression.py`). The `reports` folder is created automatically.
---
 
## 5. What a successful run looks like
 
- **Run time:** about 6 seconds on my test run.
- The terminal prints these parts in order: Part 1 (four cases), Part 2 (data preparation), Part 3 (NumPy training), Part 4 (scikit-learn and baseline), Part 5 (thresholds and the final table).
- Key numbers you should see:
  - Case A: `z = 0.60`, `p = 0.6457`, predicted class 1
  - Class counts: benign = 357, malignant = 212
  - Initial training loss `0.693147`, final training loss `0.246834`
  - Coefficients (standardized units): intercept `-0.727172`, mean radius `3.656794`, mean texture `1.009872`
  - Selected threshold: `0.6`
- The script may print a `UserWarning` from scikit-learn about `penalty=None` and `C`. This is expected because the assignment asks for `C=np.inf`. It is not an error.
**Output files** (in the `reports` folder inside the project):
 
| File | What it is |
|------|-----------|
| `reports/threshold_results.csv` | 7 validation rows, one for each threshold |
| `reports/assignment_results.csv` | 10 rows: 5 configurations for validation and for test |
| `reports/training_loss.png` | NumPy training loss vs. epoch |
 
To open them: double-click the files, or on Windows run `start reports\training_loss.png`, and on macOS run `open reports/training_loss.png`. CSV files open in Excel or any text editor.
 
**Small differences.** The splits use `random_state=42`, so counts should be exactly the same. The metrics could differ in the last decimal places if you have different package versions. The CSV files keep 8 decimal places.
 
---
 
## 6. Troubleshooting
 
| Problem | What to try |
|---------|-------------|
| `python` is not recognized (Windows) | Python is not installed or not in PATH. Reinstall it and check "Add Python to PATH", then open a new PowerShell window. You can also try `py --version`. |
| `python: command not found` (macOS) | Use `python3` before you activate the environment. After activation, `python` works. |
| `ModuleNotFoundError: No module named 'numpy'` (or sklearn, matplotlib) | The packages are not installed, or the environment is not active. Activate the environment, then run `python -m pip install -r requirements.txt` again. |
| `can't open file ... logistic_regression.py` | You are in the wrong folder. Use `cd` to go into the project folder, then check with `dir` / `ls`. |
| PowerShell blocks `Activate.ps1` | See the "If activation is blocked" step in section 4. |
| `git` is not recognized | Install Git from the link in section 2, then open a new terminal. |
 
---
 
## 7. Files in this repository
 
| File | Description |
|------|-------------|
| `logistic_regression.py` | The full script (Parts 1 to 6) |
| `requirements.txt` | Packages needed |
| `reports/` | The two CSV files and the loss plot |
| `memo.md` | The written report, tables, and research process |
| `.gitignore` | Files that Git should ignore |
 
---
 
## 8. Sources
 
- Wisconsin Diagnostic Breast Cancer dataset, from scikit-learn (`load_breast_cancer`, see DESCR) and the UCI Machine Learning Repository.
- scikit-learn and NumPy documentation.
- MSBA265 Project Assignment 2 instructions_by Shyla Solis