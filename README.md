# Groundwater Hardness Prediction with LightGBM
https://www.kaggle.com/competitions/titanic/overview

## Project Objective
This project aims to predict the hardness of groundwater in various regions of Mexico based on chemical composition. The goal is to assist in addressing water quality issues in Mexico by leveraging machine learning techniques, specifically using LightGBM.

## Methods Used
* Machine Learning
* LightGBM Regression
* Data Preprocessing & Feature Engineering

## Evaluation Metrics
* R-squared (R²) Score

## Language
* Python

## Modules Used
* Pandas
* NumPy
* Sklearn
* LightGBM

## Step-by-Step Process

1. **Dataset Analysis & Preprocessing**
    * Load and inspect the dataset
    * Check missing values and handle them accordingly

2. **Feature Engineering**
    * Create new features based on chemical composition
    * Standardize numerical features for better model performance

3. **Data Preparation**
    * Remove irrelevant columns
    * Split data into training and testing sets (80:20 split)

4. **Model Training**
    * Train a LightGBM regressor with hyperparameter tuning
    * Implement cross-validation to improve generalization

5. **Hyperparameter Tuning**
    * Use GridSearchCV to find optimal parameters
    * Optimize learning rate, max depth, and boosting rounds

6. **Model Evaluation**
    * **Metric** | **Training Data** | **Test Data**
        * R² Score | 0.9211 | 0.7662

7. **Results**
    * Achieved an R² score of 0.9211 on the training set and 0.7662 on the validation set.
    * Scored **0.91952** in the Data Science Academy COMPFEST 2024 competition.

## Titanic Classification with Neural Network

Additionally, a classification model was built for Titanic survival prediction using a neural network.

### Model Performance

* **Classification Report**
  
    | Class | Precision | Recall | F1-score | Support |
    |-------|------------|------------|------------|------------|
    | Not Survived (0) | 0.88 | 0.87 | 0.88 | 105 |
    | Survived (1) | 0.82 | 0.84 | 0.83 | 74 |

* **Overall Metrics**
  * Accuracy: **0.85**
  * Macro Avg: Precision = **0.85**, Recall = **0.85**, F1-score = **0.85**
  * Weighted Avg: Precision = **0.86**, Recall = **0.85**, F1-score = **0.86**

* **Neural Network Training Performance**
  * Final Validation Accuracy: **0.8156**
  * Test Loss: **0.4391**
  * Test Accuracy: **0.8156**

## Conclusion
 * Scored **0.78229** on Kaggle competition: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview)
