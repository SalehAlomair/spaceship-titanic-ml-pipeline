# Spaceship Titanic - End-to-End ML Pipeline

This repository contains my machine learning workflow for the Kaggle **Spaceship Titanic** competition.

## Project Summary
I built an end-to-end classification pipeline to predict whether a passenger was transported to another dimension (`Transported`).

The project covers:
- Exploratory Data Analysis (EDA)
- Missing-value handling with domain-aware imputation
- Feature engineering from `PassengerId` and `Cabin`
- Categorical encoding and preprocessing
- Model training and comparison
- Ensemble modeling and Kaggle submission generation

## Dataset
Competition files used:
- `data/spaceship-titanic/train.csv`
- `data/spaceship-titanic/test.csv`
- `data/spaceship-titanic/sample_submission.csv`

## Workflow
1. **EDA**
- Checked distributions and class balance.
- Explored feature-target relations (for example, spending features and age vs `Transported`).

2. **Data Cleaning and Imputation**
- Filled categorical and numerical missing values using grouped/domain logic.
- Applied spending-related imputations and built `TotalSpends`.

3. **Feature Engineering**
- Built group-level features from `PassengerId`:
  - `Group`, `GroupSize`, `GroupMember`, `SoloPassenger`
- Split `Cabin` into:
  - `Deck`, `CabinNum`, `Side`

4. **Preprocessing and Encoding**
- One-hot encoded categorical features.
- Used `ColumnTransformer` and `Pipeline` for robust preprocessing.

5. **Modeling**
- Trained and compared:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - CatBoost
  - LightGBM
- Built a weighted soft-voting ensemble.

6. **Validation Improvement**
- Fixed leakage-prone steps.
- Switched evaluation to **group-aware splitting/CV** (`GroupShuffleSplit` + `GroupKFold`) using passenger groups to produce more realistic local estimates.

7. **Submission**
- Generated final predictions and saved:
  - `submission_final.csv`

## Results
- **Kaggle Public Leaderboard Score:** `0.80570`
- **Public Leaderboard Position (snapshot):** `433 / 2020` (**Top 21.4%**)

This score reflects the final submission generated from the notebook pipeline in this repository.

## Repository Structure
- `notebook.ipynb`: full project notebook (EDA to submission)
- `submission_final.csv`: submission file for Kaggle
- `data/spaceship-titanic/`: competition data files
- `requirements.txt`: Python dependencies

## How To Run
```bash
pip install -r requirements.txt
jupyter notebook
```
Then open `notebook.ipynb` and run cells top-to-bottom.
