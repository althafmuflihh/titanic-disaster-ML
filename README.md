# Titanic Survival Prediction with Neural Networks
https://www.kaggle.com/competitions/titanic/overview
## Project Objective
This project aims to develop a neural network model to predict Titanic passenger survival based on various features. The model is optimized using feature engineering and class balancing techniques, with performance evaluated through accuracy, precision, recall, and F1-score.

## Methods Used
* Machine Learning
* Neural Networks (Deep Learning)
* Data Preprocessing & Feature Engineering

## Evaluation Metrics
* Accuracy
* Precision
* Recall
* F1-score
* Loss Function

## Language
* Python

## Modules Used
* Pandas
* NumPy
* TensorFlow/Keras
* Sklearn

## Step-by-Step Process

1. **Dataset Analysis & Preprocessing**
    * Check class distribution (survived vs. not survived)
    * Handle missing values

2. **Feature Engineering**
    * Extract titles from names and map them to standardized categories
    * Create new features:
        * Family Size
        * IsAlone (indicator for passengers traveling alone)
    * Convert categorical variables (e.g., Sex) into numerical values
    * Normalize numerical features

3. **Data Preparation**
    * Remove unnecessary columns (e.g., Name, Ticket, Cabin)
    * Standardize numerical features

4. **Model Training**
    * Split data into train-test (80:20 split, stratified)
    * Train a feedforward neural network with the following architecture:
        * **Layer 1:** Dense (16 neurons, ReLU activation)
        * **Layer 2:** Dense (8 neurons, ReLU activation)
        * **Output Layer:** Dense (1 neuron, Sigmoid activation)
    * Loss function: Binary Crossentropy
    * Optimizer: Adam
    * Batch Size: 8
    * Epochs: 50
    * Class Weights: {0: 0.80, 1: 1.33} to handle class imbalance

5. **Model Evaluation**
    * **Metric** | **Validation Data**
        * Accuracy | 81.56%
        * Final Kaggle Score | 0.78229
    * **Classification Report:**
      ```
                  precision    recall  f1-score   support

               0       0.88      0.87      0.88       105
               1       0.82      0.84      0.83        74

        accuracy                           0.85       179
       macro avg       0.85      0.85      0.85       179
    weighted avg       0.86      0.85      0.86       179
      ```
    * **Final Test Accuracy:** 81.56%
    * **Final Test Loss:** 0.4391

## Results
Scored **0.78229** in the leaderboard of the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/overview).
