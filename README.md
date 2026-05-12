# Polynomial Regression - Position Salary Prediction

A machine learning project that compares Simple Linear Regression with Polynomial Regression for predicting employee salaries based on position levels.

---

# Project Overview

This project uses a dataset containing:

- Position Level
- Salary

The goal is to analyze whether a simple linear relationship is enough to model salaries or if a polynomial curve performs better.

The project demonstrates how Polynomial Regression can fit nonlinear relationships more accurately than standard Linear Regression.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# Machine Learning Concepts Used

This project demonstrates understanding of:

- Simple Linear Regression
- Polynomial Regression
- Feature Engineering
- Polynomial Features
- Model Prediction
- Overfitting Basics
- Mean Squared Error (MSE)
- Data Visualization
- Curve Fitting

---

# Dataset

Dataset columns:

| Column | Description |
|---|---|
| Position Level | Employee level in the company |
| Salary | Employee salary |

---

# Project Workflow

## 1. Import Libraries

The project starts by importing the required libraries for:

- Data analysis
- Machine learning
- Visualization

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

---

## 2. Load Dataset

```python
data = pd.read_csv("Position_Salaries.csv")
```

---

## 3. Feature Selection

Input feature:

```python
x = data.iloc[:,1:2].values
```

Target variable:

```python
y = data.iloc[:,-1:]
```

---

# Simple Linear Regression

## Train Linear Regression Model

```python
linearModel = LinearRegression()
linearModel.fit(x,y)
```

---

## Prediction

```python
y_pred = linearModel.predict(x)
```

---

## Visualization

The regression line is visualized using matplotlib.

---

## MSE Calculation

```python
mse = mean_squared_error(y, y_pred)
```

This measures how far predictions are from actual salary values.

---

# Polynomial Regression

The project then improves the model using Polynomial Regression.

---

## Generate Polynomial Features

```python
poly_reg = PolynomialFeatures(degree=4)
```

This creates new features such as:

- x²
- x³
- x⁴

---

## Transform Features

```python
x_poly = poly_reg.fit_transform(x)
```

---

## Train Polynomial Model

```python
linear5 = LinearRegression()
linear5.fit(x_poly, y)
```

Although the model is still Linear Regression internally, polynomial features allow it to fit nonlinear curves.

---

# Mathematical Formula

The polynomial model learns an equation similar to:

:contentReference[oaicite:0]{index=0}

---

# Smooth Curve Visualization

A smooth prediction curve is generated using:

```python
x_grid = np.arange(min(x), max(x), 0.1)
```

This improves visualization quality and creates a continuous polynomial curve.

---

# Final Evaluation

## Polynomial MSE

The project compares the Mean Squared Error between:

- Linear Regression
- Polynomial Regression

to determine which model performs better.

---

# Key Learning Outcomes

Through this project I learned:

- The difference between Linear and Polynomial Regression
- How nonlinear relationships work
- How PolynomialFeatures transforms data
- How overfitting can happen with high polynomial degrees
- How to calculate MSE
- How to visualize regression curves properly
- How feature engineering changes model behavior

---

# Concepts Practiced

- Regression
- Curve fitting
- Feature transformation
- Prediction
- Visualization
- Error metrics

---

# Future Improvements

Possible future improvements:

- Add Train/Test split
- Compare different polynomial degrees
- Add R² Score evaluation
- Detect overfitting and underfitting
- Add Ridge and Lasso Regularization
- Visualize Bias-Variance Tradeoff

---

# Author

Saeed Hafez

Software Engineering Student interested in:
- Artificial Intelligence
- Machine Learning
- Data Science
- NLP
