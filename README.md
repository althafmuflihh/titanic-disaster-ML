# Titanic Survival Prediction

This project applies machine learning to predict survival outcomes for passengers on the Titanic. A neural network model was trained to classify passengers based on various features.

## Dataset
The dataset comes from the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/overview). It contains information about Titanic passengers, including their demographics, ticket details, and survival status.

## Model Performance
- **Validation Accuracy:** 81.56%
- **Final Kaggle Score:** 0.78229

## Features Used
- Passenger class
- Sex
- Age
- Family size
- Fare
- Title extracted from names
- Alone status

## Preprocessing Steps
- Extracted titles from names and mapped them to standardized categories
- Filled missing age values based on grouped mean by title
- Converted categorical variables (e.g., Sex) into numerical values
- Created new features such as Family Size and IsAlone
- Normalized numerical features

## Model Architecture
A simple feedforward neural network built using TensorFlow/Keras:
- **Layer 1:** Dense (16 neurons, ReLU activation)
- **Layer 2:** Dense (8 neurons, ReLU activation)
- **Output Layer:** Dense (1 neuron, Sigmoid activation)

## Training
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Batch Size: 8
- Epochs: 50
- Class Weights: {0: 0.80, 1: 1.33} to handle class imbalance

## Results
- **Classification Report:**
  ```
              precision    recall  f1-score   support

           0       0.88      0.87      0.88       105
           1       0.82      0.84      0.83        74

    accuracy                           0.85       179
   macro avg       0.85      0.85      0.85       179
weighted avg       0.86      0.85      0.86       179
  ```
- **Final Test Accuracy:** 81.56%
- **Final Test Loss:** 0.4391

## Submission
The final predictions were saved and submitted to Kaggle using:
```python
submission.to_csv('submission.csv', index=False, sep=',')
```

## Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/titanic-ml.git
   cd titanic-ml
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```sh
   python train.py
   ```
4. Generate predictions for test data:
   ```sh
   python predict.py
   ```

## Acknowledgments
This project was developed as part of the Kaggle Titanic Competition and achieved a score of **0.78229**.

