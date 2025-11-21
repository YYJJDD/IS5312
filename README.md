# IS5312 Mini-Project: Speed Dating Match Prediction

This project predicts the likelihood of a match in speed dating based on various factors such as attractiveness ratings, personality traits, shared interests, and participant expectations.

## Project Structure

```
IS5312/
├── main.py                 # Main analysis script
├── SpeedDating.csv         # Dataset
├── variables.txt           # Variable definitions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Dataset Description

The dataset contains information about speed dating events, including:
- **Demographics**: Gender, age, race
- **Ratings**: Self-ratings and partner ratings on attractiveness, sincerity, intelligence, humor, ambition
- **Preferences**: What participants look for in a partner
- **Expectations**: Expected number of matches and interest
- **Outcome**: Whether a match occurred (target variable)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script:
```bash
python main.py
```

## What the Script Does

1. **Data Exploration**: 
   - Loads and examines the dataset
   - Displays basic statistics and missing values
   - Analyzes target variable distribution

2. **Data Preprocessing**:
   - Handles categorical variables (gender, samerace, match)
   - Fills missing values
   - Encodes categorical features

3. **Model Building**:
   - Trains three classification models:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
   - Performs cross-validation

4. **Model Evaluation**:
   - Compares model performance
   - Generates classification reports
   - Creates confusion matrices
   - Plots ROC curves

5. **Feature Importance**:
   - Identifies most important features (for tree-based models)
   - Creates visualization of feature importance

6. **Outputs**:
   - Console output with detailed results
   - `feature_importance.png`: Feature importance plot
   - `roc_curves.png`: ROC curve comparison

## Key Features

- **Multiple Models**: Compares different machine learning algorithms
- **Comprehensive Evaluation**: Uses multiple metrics (accuracy, AUC-ROC, cross-validation)
- **Visualizations**: Generates plots for better understanding
- **Feature Analysis**: Identifies which factors are most important for match prediction

## Expected Output

The script will output:
- Dataset statistics and information
- Preprocessing steps
- Model training progress
- Performance metrics for each model
- Best model selection
- Detailed evaluation results
- Feature importance analysis
- Summary statistics

## Notes

- The script automatically handles missing values
- Categorical variables are encoded using label encoding
- Features are standardized for Logistic Regression
- The best model is selected based on AUC-ROC score

