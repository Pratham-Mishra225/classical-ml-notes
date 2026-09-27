# Linear Regression

> Complete theory, intuition, mathematics, practical understanding, implementation, and interview preparation.

---

## 0. Learning Objectives

By the end of this chapter, I should be able to:

- Explain Linear Regression intuitively.
- Give a formal, interview-ready definition.
- Explain what problem Linear Regression solves.
- Explain simple and multiple Linear Regression.
- Explain the mathematical model and the meaning of coefficients.
- Derive the Ordinary Least Squares (OLS) objective.
- Explain Mean Squared Error, Residual Sum of Squares, and $R^2$.
- Derive the Normal Equation.
- Explain Gradient Descent and when it is useful.
- Explain the assumptions of Linear Regression.
- Understand bias, variance, overfitting, and underfitting.
- Explain the effect of outliers and multicollinearity.
- Understand why feature scaling is usually not required for ordinary least squares, but can still be useful for numerical stability.
- Tune the important model settings and understand related regularized models such as Ridge and Lasso.
- Compare Linear Regression with Polynomial Regression, Ridge, Lasso, Decision Trees, and Random Forest.
- Implement Linear Regression using Scikit-Learn.
- Understand a basic from-scratch implementation.
- Diagnose common Linear Regression problems using residuals.
- Answer placement and interview follow-up questions.

---

# 1. Prerequisites

## Concepts I Should Know First

- Basic algebra
- Functions and graphs
- Vectors and matrices
- Mean and variance
- Derivatives and partial derivatives
- Basic probability
- Train/validation/test split
- Basic machine learning concepts
- Regression evaluation metrics

## Connection With Previous Algorithms

Linear Regression is one of the most important starting points for supervised machine learning.

It is a **parametric** model: instead of learning a large collection of rules, it learns a fixed number of coefficients.

The basic idea is:

```text
Features
   ↓
Weighted combination
   ↓
Predicted continuous value
```

For one feature, the model learns a line:

```text
y = β₀ + β₁x
```

For multiple features, the model learns a hyperplane:

```text
y = β₀ + β₁x₁ + β₂x₂ + ... + β_dx_d
```

Linear Regression is also the foundation for understanding:

- Ridge Regression
- Lasso Regression
- Elastic Net
- Polynomial Regression
- Logistic Regression
- Gradient-based optimization
- Bias-variance trade-offs

---

# 2. Why Do We Need This Algorithm?

## 2.1 The Problem

Suppose we have historical data such as:

| Hours Studied | Exam Score |
|---:|---:|
| 2 | 45 |
| 4 | 55 |
| 6 | 65 |
| 8 | 78 |
| 10 | 88 |

We want to learn a relationship between the input and the continuous target.

For example:

> Given the number of hours a student studies, can we estimate the student's exam score?

This is a **regression problem** because the target is numerical and continuous.

The challenge is that real data rarely follows a perfect equation.

We therefore need a model that can find a line or hyperplane that describes the relationship as well as possible.

---

## 2.2 Limitations of Previous Approaches

A basic rule such as:

```text
Score = Hours × 8
```

is manually chosen.

It does not learn from data.

A very flexible model, such as a deep Decision Tree, can model complicated relationships but may have high variance and can be harder to interpret.

Linear Regression gives us a simple, interpretable mathematical relationship:

```text
Target
  ↑
  |
  |       •
  |    •
  |  •
  | •
  +----------------→ Feature
          Best-fit line
```

Its simplicity is useful when the relationship between features and target is reasonably approximated by a linear function.

---

## 2.3 What Should a Better Approach Do?

A useful regression model should:

- Learn its parameters from data.
- Predict continuous values.
- Capture the main trend in the data.
- Make prediction errors as small as possible according to a chosen loss.
- Be interpretable.
- Be computationally efficient.
- Generalize to unseen data.
- Provide a strong baseline for more complex models.

Linear Regression satisfies these requirements when its assumptions are reasonably appropriate.

---

## 2.4 Core Idea

> **Linear Regression learns coefficients for a linear function by choosing the line or hyperplane that minimizes the difference between observed target values and predicted values, usually by minimizing the sum of squared residuals.**

The central idea is:

```text
Training Data
    ↓
Assume a linear relationship
    ↓
Choose coefficients
    ↓
Predict target values
    ↓
Measure prediction errors
    ↓
Minimize squared errors
    ↓
Best-fit linear model
```

The phrase **best-fit line** means the line whose parameters minimize the selected loss, usually squared error in Ordinary Least Squares.

---

# 3. Formal Definition

## Definition

> **Linear Regression is a supervised parametric regression algorithm that models the relationship between one or more input features and a continuous target as a linear function, with its coefficients estimated by minimizing a loss such as the residual sum of squares.**

## Interview-ready one-line definition

> **Linear Regression fits a linear relationship between input features and a continuous target by learning coefficients that minimize the sum of squared prediction errors.**

## Algorithm Classification

| Property | Description |
|---|---|
| Learning Type | Supervised |
| Task | Regression |
| Parametricity | Parametric |
| Model Type | Linear model |
| Learning Approach | Ordinary Least Squares by default |
| Output | Continuous value |
| Main Objective | Minimize squared prediction error |
| Interpretability | High when the feature set is well specified |
| Non-linearity | Not represented directly by the basic linear model |

---

# 4. Intuition

## 4.1 Core Intuition

Imagine plotting training data on a graph.

There may be many possible lines:

```text
Line A  ─────────────
Line B  ───────────╱
Line C  ────────╱
```

Linear Regression tries to find the line that fits the observed data according to the chosen objective.

For Ordinary Least Squares, it looks at the vertical difference between each actual point and the predicted point on the line.

That difference is called a **residual**.

```text
Actual point
     •
     |\
     | \  Residual
     |  \
     •---\----------- Best-fit line
 Predicted point
```

The model chooses the coefficients so that the total squared residual error is as small as possible.

---

## 4.2 Real-World Analogy

Suppose you want to estimate the selling price of a house.

A simple model might use only area:

```text
Price ≈ β₀ + β₁ × Area
```

A multiple Linear Regression model might use:

```text
Price ≈ β₀
       + β₁ × Area
       + β₂ × Bedrooms
       + β₃ × Age
       + β₄ × DistanceToStation
```

The model learns the coefficients from historical house data.

The coefficients tell us how the predicted price changes when a feature changes, **holding the other features in the model fixed**, under the linear model interpretation.

---

## 4.3 Simple Example

Suppose we have:

| Hours Studied ($x$) | Score ($y$) |
|---:|---:|
| 1 | 50 |
| 2 | 60 |
| 3 | 70 |
| 4 | 80 |

The points follow the relationship:

```text
Score = 40 + 10 × Hours
```

So:

- Intercept = 40
- Slope = 10

For $x=5$:

```math
\hat{y} = 40 + 10(5) = 90
```

The model therefore predicts a score of 90.

Real datasets will usually not fit a perfect line.

---

## 4.4 Mental Model

> **Find the coefficients of the line or hyperplane that make the squared prediction errors as small as possible.**

---

# 5. Problem Formulation

## Given

Training dataset:

```math
D = \{(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)\}
```

For multiple Linear Regression, each observation has $d$ features:

```math
x_i = (x_{i1},x_{i2},\ldots,x_{id})
```

and the target is $y_i$.

Where:

- $x_i$ = feature vector for sample $i$
- $y_i$ = observed target for sample $i$
- $n$ = number of training observations
- $d$ = number of input features
- $\beta_0$ = intercept
- $\beta_j$ = coefficient for feature $j$

## Goal

Learn coefficients:

```math
\beta_0,\beta_1,\ldots,\beta_d
```

such that predictions are as close as possible to the observed targets according to the training objective.

## Input

The model receives:

- Training feature matrix $X$
- Training target vector $y$

During prediction, the model receives a new feature vector:

```math
x = (x_1,x_2,\ldots,x_d)
```

## Output

A continuous prediction:

```math
\hat{y}
```

---

# 6. Mathematical Foundation

## 6.1 Model Representation

### Simple Linear Regression

With one feature:

```math
\hat{y}_i = \beta_0 + \beta_1x_i
```

where:

- $\beta_0$ = intercept
- $\beta_1$ = slope
- $x_i$ = input feature
- $\hat{y}_i$ = predicted target

### Multiple Linear Regression

With $d$ features:

```math
\hat{y}_i =
\beta_0
+
\beta_1x_{i1}
+
\beta_2x_{i2}
+
\cdots
+
\beta_dx_{id}
```

This can be written in matrix form.

Let the design matrix be:

```math
X =
\begin{bmatrix}
1 & x_{11} & x_{12} & \cdots & x_{1d}\\
1 & x_{21} & x_{22} & \cdots & x_{2d}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & x_{n1} & x_{n2} & \cdots & x_{nd}
\end{bmatrix}
```

Let:

```math
\beta =
\begin{bmatrix}
\beta_0\\
\beta_1\\
\beta_2\\
\vdots\\
\beta_d
\end{bmatrix}
```

and:

```math
y =
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{bmatrix}
```

Then:

```math
\hat{y} = X\beta
```

---

## 6.2 Objective

The objective is to choose $\beta$ so that predicted values are as close as possible to actual values.

For Ordinary Least Squares:

```math
\hat{\beta}
=
\arg\min_{\beta}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
```

Since:

```math
\hat{y}=X\beta
```

we can write:

```math
\hat{\beta}
=
\arg\min_{\beta}
\|y-X\beta\|_2^2
```

The quantity being minimized is the **Residual Sum of Squares (RSS)**.

---

## 6.3 Loss / Cost / Objective Function

### Residual

For observation $i$:

```math
e_i = y_i-\hat{y}_i
```

The residual is the difference between the observed value and the predicted value.

### Residual Sum of Squares

```math
RSS
=
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
```

In matrix form:

```math
RSS
=
(y-X\beta)^T(y-X\beta)
```

### Mean Squared Error

```math
MSE
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
```

### Training objective

Because multiplying RSS by the positive constant $\frac{1}{n}$ does not change the minimizer:

```math
\arg\min_{\beta} RSS
=
\arg\min_{\beta} MSE
```

Therefore, minimizing RSS and minimizing MSE give the same OLS coefficients when the same training data and model are used.

---

## 6.4 Why This Objective Function?

Squared error is useful because:

1. Positive and negative residuals do not cancel.
2. Large errors are penalized more strongly than small errors.
3. The resulting OLS objective is a smooth convex quadratic function.
4. The objective has a convenient mathematical solution.
5. It allows efficient optimization and useful statistical analysis.

For example:

```text
Error = 2
Squared error = 4

Error = 5
Squared error = 25
```

A large error therefore receives much more penalty.

This is also a limitation:

> **OLS can be sensitive to outliers because large residuals are squared.**

---

## 6.5 Optimization

There are two important ways to solve Ordinary Least Squares.

### 1. Closed-form solution

The coefficients can be obtained using the Normal Equation when the required matrix solution exists:

```math
\hat{\beta}
=
(X^TX)^{-1}X^Ty
```

In practice, numerical linear algebra methods are preferred over explicitly computing a matrix inverse.

### 2. Gradient Descent

Gradient Descent starts with initial coefficients and repeatedly updates them in the direction that reduces the objective.

A generic update is:

```math
\beta
\leftarrow
\beta-\alpha\nabla J(\beta)
```

where:

- $\beta$ = parameter vector
- $\alpha$ = learning rate
- $J(\beta)$ = cost function
- $\nabla J(\beta)$ = gradient of the cost function

For a quadratic Linear Regression objective, the optimization landscape is convex.

---

## 6.6 Derivation

### Step 1 — Write the model

```math
\hat{y}=X\beta
```

### Step 2 — Write the residual vector

```math
e=y-\hat{y}
```

Therefore:

```math
e=y-X\beta
```

### Step 3 — Write the RSS objective

```math
J(\beta)
=
(y-X\beta)^T(y-X\beta)
```

Expand it:

```math
J(\beta)
=
y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta
```

Since $y^TX\beta$ is a scalar:

```math
y^TX\beta=\beta^TX^Ty
```

Therefore:

```math
J(\beta)
=
y^Ty
-
2\beta^TX^Ty
+
\beta^TX^TX\beta
```

### Step 4 — Differentiate with respect to $\beta$

```math
\nabla_{\beta}J(\beta)
=
-2X^Ty+2X^TX\beta
```

### Step 5 — Set the gradient to zero

At the minimum:

```math
-2X^Ty+2X^TX\beta=0
```

Divide by 2:

```math
X^TX\beta=X^Ty
```

These are called the **Normal Equations**.

### Step 6 — Solve for $\beta$

If $X^TX$ is invertible:

```math
\hat{\beta}
=
(X^TX)^{-1}X^Ty
```

This is the Normal Equation solution.

### Important Numerical Note

You generally should not implement the solution by explicitly calculating:

```math
(X^TX)^{-1}
```

because explicit inversion can be numerically less stable and less efficient.

Numerical linear algebra libraries typically use stable matrix factorization or least-squares solvers.

---

## 6.7 Important Mathematical Properties

- **Objective:** Ordinary Least Squares minimizes a convex quadratic objective.
- **Convex / Non-convex:** Convex.
- **Differentiable / Non-differentiable:** Differentiable.
- **Closed-form / Iterative:** OLS has a closed-form characterization through the Normal Equations; iterative methods such as Gradient Descent can also be used.
- **Local vs Global Optimum:** For the convex OLS objective, every local minimum is a global minimum.
- **Unique solution:** The OLS coefficient vector is unique when the relevant design matrix has full column rank.
- **Singular case:** If features are perfectly linearly dependent, $X^TX$ is singular; a pseudoinverse or another solver can be used to obtain a least-squares solution.
- **Differentiability:** The squared-error objective is smooth.
- **Gradient:** The gradient is linear in $\beta$.

---

## 6.8 Coefficient Interpretation

For the model:

```math
\hat{y}
=
\beta_0
+
\beta_1x_1
+
\cdots
+
\beta_dx_d
```

### Intercept

$\beta_0$ is the predicted target when all input features are zero, assuming that such a point is meaningful and within the relevant data range.

### Coefficient

$\beta_j$ represents the change in the model's predicted target for a one-unit increase in $x_j$, **holding the other included features fixed**.

For example:

```math
\hat{Price}
=
20
+
0.5(Area)
-
2(Age)
```

Then:

- $\beta_{\text{Area}}=0.5$
- $\beta_{\text{Age}}=-2$

The interpretation of the coefficients depends on the units of the variables.

> **Important interview point:** A coefficient describes association within the fitted model. It does not automatically imply that changing a feature causes the target to change.

---

## 6.9 $R^2$

The coefficient of determination is:

```math
R^2
=
1-
\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
```

where:

```math
\bar{y}
=
\frac{1}{n}
\sum_{i=1}^{n}y_i
```

Interpretation:

- Numerator = unexplained squared error from the model.
- Denominator = total squared variation around the mean.
- $R^2$ compares the model with a mean-only baseline.

For a standard OLS model evaluated on the same training data with an intercept, $R^2$ is commonly between 0 and 1.

On unseen data, $R^2$ can be negative.

---

## 6.10 Statistical Meaning vs Machine Learning Meaning

Linear Regression is used in two related but distinct ways.

### Predictive use

The goal is to predict unseen target values accurately.

The focus is usually:

- Generalization
- Validation error
- Test error
- Cross-validation

### Statistical inference

The goal may be to understand relationships and quantify uncertainty.

The focus may include:

- Coefficient estimates
- Standard errors
- Confidence intervals
- Hypothesis tests
- Residual assumptions

These are related but not identical goals.

> **A model can be useful for prediction without being suitable for causal interpretation.**

---

# 7. Geometric / Visual Understanding

## 7.1 What Does the Data Look Like?

### One Feature

With one feature, the data can be plotted in two dimensions:

```text
Target
  ↑
90|                 •
80|              •
70|           •
60|        •
50|     •
  +--------------------------→ Feature
```

Linear Regression learns a line through the data.

### Two Features

With two input features and one target, the model learns a plane.

### More Than Two Features

With $d$ features, the model learns a hyperplane in $d$-dimensional feature space.

---

## 7.2 What Does the Model Learn?

The model learns:

- An intercept $\beta_0$
- One coefficient for each feature
- A linear prediction function

For example:

```math
\hat{y}
=
\beta_0+\beta_1x_1+\beta_2x_2
```

The model does not learn arbitrary decision regions like a Decision Tree.

It learns one global linear relationship in the original feature space.

---

## 7.3 Effect of Model Complexity

### Simple Linear Model

```text
y = β₀ + β₁x
```

The model can represent only straight-line relationships.

### Multiple Linear Model

```text
y = β₀ + β₁x₁ + β₂x₂ + ... + β_dx_d
```

The model can use more features but remains linear in the coefficients.

### Polynomial Features

We can create additional features such as:

```math
x^2,\;x^3,\;x_1x_2
```

and then fit a linear model to those transformed features.

For example:

```math
\hat{y}
=
\beta_0
+
\beta_1x
+
\beta_2x^2
```

This produces a curved relationship in the original feature space even though the model is still linear in the parameters.

### Important distinction

> **Linear Regression means linear in the parameters, not necessarily that the graph must be a straight line after arbitrary feature transformations.**

---

### Diagram

```text
Simple Linear Regression:

y
↑
|       •
|     •
|   •     •
| •
|________________→ x
      /
     /
    /
   Best-fit line


Multiple Linear Regression:

y
↑
|        _________
|      /          /
|    /          /
|  /__________/
|
+----------------→ x₁
  with x₂ forming the second feature dimension
```

---

# 8. How the Algorithm Works

## Step-by-Step

### Step 1 — Start With the Training Dataset

Suppose:

```text
X_train → input features
y_train → continuous target
```

The dataset contains $n$ observations and $d$ features.

---

### Step 2 — Specify the Model

For multiple Linear Regression:

```math
\hat{y}
=
\beta_0
+
\beta_1x_1
+
\cdots
+
\beta_dx_d
```

The coefficients are initially unknown.

---

### Step 3 — Define the Objective

For OLS, define:

```math
RSS
=
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
```

The goal is to find the coefficients that minimize this quantity.

---

### Step 4 — Estimate the Coefficients

The coefficients can be obtained through:

- A least-squares solver / closed-form characterization.
- Gradient Descent or related iterative optimization methods.

For the Normal Equation:

```math
\hat{\beta}
=
(X^TX)^{-1}X^Ty
```

when the required inverse exists.

---

### Step 5 — Use the Learned Model for Prediction

For a new feature vector $x$:

```math
\hat{y}
=
\beta_0
+
\beta_1x_1
+
\cdots
+
\beta_dx_d
```

The result is the predicted continuous value.

---

## Algorithm Flow

```text
Input Data
    ↓
Select features and target
    ↓
Define linear model
    ↓
Choose OLS objective
    ↓
Estimate coefficients
    ↓
Store β₀, β₁, ..., β_d
    ↓
Receive new sample
    ↓
Apply learned linear equation
    ↓
Prediction
```

## Pseudocode

```text
Input:
    Training data X, y

1. Add an intercept term when required.
2. Represent the prediction function as Xβ.
3. Define the squared-error objective.
4. Estimate β using a least-squares solver or an iterative optimizer.
5. Store the learned coefficients.
6. For a new sample x:
       Calculate y_hat = xβ
7. Return y_hat.
```

---

# 9. Training vs Prediction

## 9.1 Training Phase

When calling:

```python
model.fit(X_train, y_train)
```

the model estimates the coefficients that best fit the training targets according to least squares.

Conceptually:

```text
X_train + y_train
        ↓
    Linear model
        ↓
   Least-squares fitting
        ↓
Learn β₀, β₁, ..., β_d
        ↓
   Trained Model
```

---

## 9.2 Prediction Phase

When calling:

```python
y_pred = model.predict(X_test)
```

the model uses the learned coefficients:

```text
X_new
   ↓
Learned coefficients
   ↓
β₀ + β₁x₁ + ... + β_dx_d
   ↓
Prediction
```

No new coefficients are learned during `predict()`.

---

## 9.3 What Does the Model Actually Learn?

The model learns:

```text
β₀ → intercept
β₁ → coefficient of feature 1
β₂ → coefficient of feature 2
...
β_d → coefficient of feature d
```

These coefficients define the prediction function.

Example:

```math
\hat{y}=5+2x_1-0.7x_2
```

The trained model has learned:

```text
Intercept = 5
Coefficient of x₁ = 2
Coefficient of x₂ = -0.7
```

---

## 9.4 What Is Stored After Training?

In Scikit-Learn, a fitted `LinearRegression` model exposes learned coefficients and intercept through attributes such as:

```python
model.coef_
model.intercept_
```

Conceptually:

```text
Trained Linear Regression
│
├── intercept
│
├── coefficient for feature 1
├── coefficient for feature 2
├── ...
└── coefficient for feature d
```

Unlike a Decision Tree, the model does not store a tree structure.

---

# 10. Worked Example

## Dataset

Suppose:

| Hours Studied ($x$) | Score ($y$) |
|---:|---:|
| 1 | 50 |
| 2 | 60 |
| 3 | 70 |
| 4 | 80 |

We will fit:

```math
\hat{y}=\beta_0+\beta_1x
```

---

## Step 1 — Compute the Means

Mean of $x$:

```math
\bar{x}
=
\frac{1+2+3+4}{4}
=
2.5
```

Mean of $y$:

```math
\bar{y}
=
\frac{50+60+70+80}{4}
=
65
```

---

## Step 2 — Compute the Slope

For simple Linear Regression, the OLS slope is:

```math
\hat{\beta}_1
=
\frac{
\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})
}{
\sum_{i=1}^{n}(x_i-\bar{x})^2
}
```

For this dataset:

```math
\sum (x_i-\bar{x})(y_i-\bar{y})=50
```

and:

```math
\sum (x_i-\bar{x})^2=5
```

Therefore:

```math
\hat{\beta}_1
=
\frac{50}{5}
=
10
```

---

## Step 3 — Compute the Intercept

The intercept is:

```math
\hat{\beta}_0
=
\bar{y}
-
\hat{\beta}_1\bar{x}
```

Substitute:

```math
\hat{\beta}_0
=
65-(10)(2.5)
=
40
```

---

## Step 4 — Write the Final Model

```math
\hat{y}=40+10x
```

---

## Step 5 — Make a Prediction

For $x=5$:

```math
\hat{y}=40+10(5)=90
```

Predicted score:

```text
90
```

---

## Step 6 — Calculate a Residual

Suppose the actual score for a student with 5 hours studied is 85.

Then:

```math
e
=
y-\hat{y}
=
85-90
=
-5
```

The negative residual means the model predicted a value higher than the actual value.

---

## Step 7 — Calculate Squared Error

```math
e^2=(-5)^2=25
```

---

## Final Result

The fitted model is:

```math
\boxed{\hat{y}=40+10x}
```

and for 5 hours of study:

```math
\boxed{\hat{y}=90}
```

> Goal: I should be able to manually calculate the basic slope, intercept, prediction, residual, and squared error for a tiny dataset.

---

# 11. Important Concepts & Terminology

## 11.1 Dependent Variable / Target

**Definition:** The target is the variable that the model tries to predict.

**Why it matters:** Linear Regression predicts a continuous target.

---

## 11.2 Independent Variable / Feature

**Definition:** A feature is an input variable used to predict the target.

**Why it matters:** Linear Regression estimates a coefficient for each included feature.

---

## 11.3 Coefficient

**Definition:** A coefficient is a learned numerical parameter that determines how strongly a feature contributes to the model's predicted value, conditional on the other included features in the linear model.

**Why it matters:** Coefficients make Linear Regression relatively interpretable.

---

## 11.4 Intercept

**Definition:** The intercept is the model's predicted target when all features are zero.

**Why it matters:** It shifts the fitted hyperplane vertically.

---

## 11.5 Residual

**Definition:** A residual is the difference between an observed target and its fitted prediction.

```math
e_i=y_i-\hat{y}_i
```

**Why it matters:** Residuals are central to OLS fitting and model diagnostics.

---

## 11.6 Ordinary Least Squares

**Definition:** Ordinary Least Squares is a method that estimates regression coefficients by minimizing the sum of squared residuals.

```math
\min_{\beta}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
```

**Why it matters:** It is the standard fitting objective behind Scikit-Learn's `LinearRegression`.

---

## 11.7 Residual Sum of Squares

**Definition:** RSS is the sum of squared residuals.

```math
RSS=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
```

**Why it matters:** It is the main objective minimized by OLS.

---

## 11.8 Mean Squared Error

**Definition:** MSE is the average of the squared prediction errors.

```math
MSE
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
```

**Why it matters:** MSE is widely used for regression training and evaluation.

---

## 11.9 Root Mean Squared Error

**Definition:** RMSE is the square root of MSE.

```math
RMSE=\sqrt{MSE}
```

**Why it matters:** RMSE is expressed in the same unit as the target.

---

## 11.10 $R^2$

**Definition:** $R^2$ measures how much the model improves squared-error fit relative to a mean-only baseline.

```math
R^2
=
1-\frac{RSS}{TSS}
```

where:

```math
TSS=\sum_{i=1}^{n}(y_i-\bar{y})^2
```

**Why it matters:** It is a common measure of regression fit, but it should not be used alone.

---

## 11.11 Normal Equation

**Definition:** The Normal Equation gives the OLS coefficient solution by solving:

```math
X^TX\hat{\beta}=X^Ty
```

and, when $X^TX$ is invertible:

```math
\hat{\beta}=(X^TX)^{-1}X^Ty
```

**Why it matters:** It gives a direct mathematical solution to the OLS problem.

---

## 11.12 Gradient Descent

**Definition:** Gradient Descent is an iterative optimization method that updates model parameters in the direction opposite to the gradient of the objective.

```math
\beta
\leftarrow
\beta-\alpha\nabla J(\beta)
```

**Why it matters:** It becomes useful for large-scale problems and is fundamental to understanding optimization in machine learning.

---

## 11.13 Multicollinearity

**Definition:** Multicollinearity occurs when two or more predictor variables are strongly linearly related.

**Why it matters:** It can make coefficient estimates unstable and difficult to interpret.

---

## 11.14 Homoscedasticity

**Definition:** Homoscedasticity means that the variance of the regression errors is approximately constant across the relevant range of predictions or features.

**Why it matters:** It is important for standard OLS statistical inference.

---

## 11.15 Heteroscedasticity

**Definition:** Heteroscedasticity occurs when the error variance changes across observations or levels of the predictors.

**Why it matters:** It can make usual standard errors unreliable even when coefficient estimates remain useful for prediction under suitable conditions.

---

# 12. Parameters vs Hyperparameters

## Parameters

**Definition:** Parameters are values learned from the training data.

For Linear Regression, the main learned parameters are:

- $\beta_0$ — intercept
- $\beta_1,\ldots,\beta_d$ — feature coefficients

These values are estimated by fitting the model.

## Hyperparameters

**Definition:** Hyperparameters are settings chosen by the practitioner rather than learned as ordinary OLS coefficients.

For `LinearRegression`, there are relatively few important hyperparameters because OLS itself is largely determined by the data and objective.

Examples of estimator settings include:

- `fit_intercept`
- `copy_X`
- `positive`
- `tol`
- `n_jobs`

Regularized models introduce more important model-selection hyperparameters, such as the regularization strength $\alpha$ in Ridge and Lasso.

## Key Difference

| Parameters | Hyperparameters |
|---|---|
| Learned from data | Chosen by the practitioner |
| Define the fitted function | Control how fitting is performed |
| Example: $\beta_1$ | Example: regularization strength $\alpha$ |
| Updated/estimated during fitting | Selected before or during model selection |

---

# 13. Important Hyperparameters

Unlike many tree-based models, `LinearRegression` has very few predictive hyperparameters.

| Setting | Meaning | Effect | Main Use |
|---|---|---|---|
| `fit_intercept` | Whether to fit an intercept | Changes the model specification | Important |
| `positive` | Constrains coefficients to be non-negative | Changes feasible coefficient set | Domain-specific |
| `tol` | Solver tolerance in supported settings | Controls numerical convergence/precision | Numerical |
| `n_jobs` | Number of parallel jobs where supported | Can affect computation time | Computational |
| `copy_X` | Whether to copy the input matrix | Can affect memory/input mutation behavior | Implementation |

## Most Important Hyperparameters

### `fit_intercept`

Default:

```python
fit_intercept=True
```

Use an intercept unless there is a clear reason not to.

If:

```python
fit_intercept=False
```

the model assumes the intercept is zero.

That means the fitted relationship is forced through the origin.

> **Interview point:** Do not remove the intercept simply because the feature values look large or small. The decision should come from the model specification and data-generating context.

### `positive`

When:

```python
positive=True
```

the coefficients are constrained to be non-negative.

This can make sense when domain knowledge says that increasing a feature should not decrease the prediction under the chosen model.

It changes the optimization problem and is not the default OLS solution.

### `tol`

`tol` controls numerical solver tolerance in the implementation settings where it applies.

It is primarily a computational setting rather than a typical bias-variance hyperparameter.

### `n_jobs`

`n_jobs` controls parallel computation where the implementation can benefit from it.

It does not change the mathematical definition of OLS.

## Hyperparameter Interactions

The most important practical interaction is between Linear Regression and **feature transformations or regularization**.

For example:

```text
Linear Regression
      ↓
Model too sensitive / unstable
      ↓
Add regularization
      ↓
Ridge / Lasso
```

Another interaction is:

```text
Linear relationship is too simple
      ↓
Add polynomial or interaction features
      ↓
Fit a linear model on transformed features
```

---

# 14. Bias-Variance & Generalization

## Bias

Bias is error caused by a model being systematically too simple for the true relationship.

Basic Linear Regression has relatively high bias when the true relationship is strongly non-linear and the model uses only raw features.

For example, if:

```math
y=x^2
```

a straight-line model on raw $x$ cannot represent the curve exactly.

---

## Variance

Variance measures how much a model's predictions change when the training data changes.

Ordinary Linear Regression often has lower variance than highly flexible models when the feature set is modest.

However, variance can become large when:

- There are many features relative to the number of observations.
- Predictors are highly correlated.
- The dataset is small.
- The data contains strong noise.

Regularization such as Ridge can reduce coefficient variance.

---

## Overfitting

### Why Can Linear Regression Overfit?

Linear Regression is less flexible than many non-linear models, but it can still overfit when:

- There are many features.
- Polynomial features create a very large feature space.
- Interaction features are added aggressively.
- The dataset is small.
- Noise features are included.
- Model selection leaks validation information.
- Multicollinearity makes coefficients unstable.

### Signs of Overfitting

```text
Training error    → very low
Validation error  → significantly higher
Test error        → significantly higher
```

---

## Underfitting

### Why Can Linear Regression Underfit?

It can underfit when:

- The true relationship is strongly non-linear.
- Important interactions are missing.
- Important features are omitted.
- The model is forced to be too simple.
- Strong regularization is applied.

### Signs of Underfitting

```text
Training error    → high
Validation error  → also high
```

---

## Bias-Variance Trade-off

```text
Model Flexibility
       ↓
High Bias → Good Generalization → High Variance
   ↓               ↓                  ↓
Underfit        Balanced           Overfit
```

For Linear Regression:

```text
Too simple
   ↓
High bias

Too many / poorly constructed features
   ↓
Higher variance
```

---

## Controlling Overfitting

- Use Ridge, Lasso, or Elastic Net when appropriate.
- Remove irrelevant features.
- Use cross-validation.
- Use feature selection based on validation performance.
- Avoid unnecessary high-degree polynomial expansion.
- Check for data leakage.
- Collect more representative data.

## Controlling Underfitting

- Add informative features.
- Add meaningful interaction or polynomial features.
- Reduce excessive regularization.
- Consider a more flexible model such as a tree-based method.

---

# 15. Assumptions

| Assumption | Required? | Why? | What If Violated? |
|---|---|---|---|
| Linearity in parameters | Yes for the basic linear model | Defines the model form | Predictions can be systematically biased if important structure is omitted |
| Independent observations/errors | Important for standard inference | Supports valid uncertainty calculations under common setups | Standard errors and validation can be misleading |
| Zero conditional mean / exogeneity | Important for unbiased coefficient interpretation | Requires errors to have mean zero given predictors | Coefficients can be biased |
| No perfect multicollinearity | Yes for a unique ordinary coefficient solution | Features must not be exact linear combinations | Coefficients may not be uniquely identifiable |
| Constant error variance | Important mainly for standard inference | Gives homoscedastic errors | Standard errors may be unreliable |
| Normally distributed residuals | Not required for OLS point estimates | Mainly useful for small-sample classical inference | Exact small-sample tests/confidence intervals may not be valid |
| Independent and representative sampling | Practically important | Supports meaningful generalization | Test performance may not represent deployment |

## Important Interview Distinction

Do not say:

> "Linear Regression requires the features to be normally distributed."

That is incorrect.

A better answer is:

> **Ordinary Least Squares does not require the input features themselves to be normally distributed. Classical statistical inference often uses assumptions about the errors, and those assumptions should be separated from the conditions needed to estimate the regression coefficients.**

Also distinguish:

```text
Model assumption
    ≠
Data preprocessing recommendation
    ≠
Causal identification assumption
    ≠
Inference assumption
```

The strongest practical assumption for coefficient unbiasedness is related to:

```math
E[\epsilon\mid X]=0
```

which means the expected error is zero conditional on the predictors.

---

# 16. Data Preprocessing

## 16.1 Missing Values

Basic Scikit-Learn `LinearRegression` workflows require missing values to be handled appropriately before fitting.

Common approaches include:

- Imputation
- Removing observations when justified
- Domain-specific missing-value handling

When using imputation:

> **Fit the imputer on the training data only and apply the learned transformation to validation and test data.**

This prevents data leakage.

---

## 16.2 Feature Scaling

**Required?** No for ordinary least-squares prediction itself.

**Why?**

For standard OLS, multiplying a feature by a positive constant changes the numerical value of its coefficient but does not fundamentally change the fitted predictions when the model is re-fitted appropriately.

Suppose:

```math
x'=ax,\qquad a>0
```

Then a model:

```math
\hat{y}=\beta_0+\beta_1x
```

can be represented as:

```math
\hat{y}
=
\beta_0
+
\frac{\beta_1}{a}x'
```

The coefficient changes scale, while the fitted relationship in terms of the original variable can remain equivalent.

### Why scaling can still be useful

Scaling may help with:

- Numerical conditioning.
- Optimization speed for gradient-based training.
- Comparing coefficient magnitudes carefully.
- Regularized models such as Ridge and Lasso.

### Interview answer

> **Feature scaling is not required for basic OLS Linear Regression, but it can improve numerical conditioning and becomes especially important when using regularization or gradient-based optimization.**

---

## 16.3 Categorical Variables

Standard Linear Regression expects numerical features.

Categorical variables therefore usually need encoding.

Common approach:

```text
Category
   ↓
One-Hot Encoding
   ↓
Numerical indicator columns
```

Example:

```text
City = Mumbai / Pune / Delhi
```

can become:

```text
City_Mumbai
City_Pune
City_Delhi
```

Avoid treating nominal categories as continuous numbers merely because an encoder can assign integers.

### Dummy Variable Trap

When an intercept is included, using all one-hot columns for a categorical variable can create perfect linear dependence.

A common strategy is to drop one reference category.

For example:

```text
City_Mumbai
City_Pune
```

with Delhi as the reference category.

Modern preprocessing pipelines can handle this carefully.

---

## 16.4 Outliers

Linear Regression can be highly sensitive to outliers.

Why?

Because the OLS objective squares residuals:

```math
(y_i-\hat{y}_i)^2
```

A large residual therefore receives a disproportionately large penalty.

A single influential observation can change:

- The slope
- The intercept
- The residual pattern
- Predictions

### Important distinction

An observation can be:

- An ordinary outlier in $y$
- An outlier in feature space
- High-leverage
- Influential

These are related but not identical concepts.

---

## 16.5 Multicollinearity

Multicollinearity matters significantly for Linear Regression.

Suppose:

```text
Feature A ─────┐
               ├── strongly related
Feature B ─────┘
```

The model may have difficulty determining how much contribution should be assigned to each feature separately.

Possible effects:

- Unstable coefficients
- Large standard errors
- Unexpected coefficient signs
- Reduced interpretability
- Numerical issues in severe cases

Prediction can still be acceptable.

> **Multicollinearity is usually more damaging to coefficient stability and interpretation than to raw predictive accuracy.**

Useful diagnostics include correlation analysis, VIF, and validation-based checks.

---

## 16.6 Feature Engineering

Linear Regression can benefit significantly from well-designed features.

Useful transformations include:

### Polynomial features

```math
x^2,\;x^3
```

### Interaction terms

```math
x_1x_2
```

### Log transforms

For strongly skewed variables, a transformation such as:

```math
x'=\log(x)
```

may make the relationship more suitable for a linear model.

### Domain-based features

Example:

```text
Price per square foot
Age at purchase
Distance from station
```

Feature engineering can allow a linear model to represent important structure without switching immediately to a non-linear model.

---

# 17. Model Complexity

## What Controls Complexity?

For basic Linear Regression, complexity is mainly influenced by:

- Number of features
- Feature transformations
- Polynomial degree
- Interaction terms
- Regularization strength in related models

The number of coefficients is roughly:

```text
d features → d + 1 parameters
```

when an intercept is included.

---

## Simple Model

Example:

```math
\hat{y}=\beta_0+\beta_1x
```

Behavior:

- Easy to interpret
- Low representational flexibility
- Can have high bias for non-linear problems

---

## Complex Model

Example:

```math
\hat{y}
=
\beta_0
+
\beta_1x
+
\beta_2x^2
+
\beta_3x^3
+
\beta_4x_1x_2
+\cdots
```

Behavior:

- More expressive
- Can capture more structure
- More vulnerable to overfitting
- Coefficient interpretation becomes more complicated

---

## Effect on Generalization

```text
Too few / weak features
       ↓
High bias

Useful features
       ↓
Better fit

Too many noisy / highly transformed features
       ↓
Higher variance
```

Regularization can control coefficient magnitude in complex feature spaces.

---

# 18. Computational Complexity

## Training Complexity

For dense OLS, a useful high-level view is that fitting depends on both the number of samples $n$ and features $d$.

A common complexity characterization for dense least-squares computation is approximately:

```math
O(nd^2+d^3)
```

The exact cost depends on:

- Solver
- Matrix shape
- Sparsity
- Numerical method
- Whether the problem is overdetermined or underdetermined

The $nd^2$ term is associated with forming or processing feature cross-products in common approaches, while the $d^3$ term reflects solving the resulting system through dense linear algebra.

### Gradient Descent

With $k$ iterations, a rough dense batch Gradient Descent cost is:

```math
O(knd)
```

per model when each iteration computes a gradient over all samples and features.

---

## Prediction Complexity

For one new observation with $d$ features:

```math
O(d)
```

because prediction requires a weighted sum of the features.

For $m$ new observations:

```math
O(md)
```

---

## Space Complexity

The basic storage for the input data is:

```math
O(nd)
```

The fitted coefficient vector requires:

```math
O(d)
```

Additional memory depends on the solver and whether intermediate matrices are formed.

If a dense $X^TX$ matrix is explicitly represented, it uses approximately:

```math
O(d^2)
```

memory.

---

## Scalability

### More Samples

More samples:

- Increase training cost.
- Increase data-storage requirements.
- Can improve generalization if the observations are representative.

OLS can scale well when the number of features is moderate.

### More Features

More features:

- Increase computational cost.
- Increase the number of parameters.
- Increase the risk of multicollinearity.
- Increase the risk of overfitting when $d$ becomes large relative to $n$.

### High-Dimensional Data

When the number of features is very large, or $d$ is comparable to or larger than $n$:

- Coefficient estimates can become unstable.
- The design matrix may be rank deficient.
- Regularization can become important.
- Sparse and specialized solvers may be preferable.

---

# 19. Regularization / Optimization Improvements

## Why Is It Needed?

Ordinary Least Squares does not penalize large coefficients.

With many features, noisy predictors, or multicollinearity, coefficients can become unstable.

Regularization adds a penalty to discourage unnecessarily large coefficients.

```text
OLS
 ↓
Fit data as closely as possible

Regularized Regression
 ↓
Fit data
+
penalize large coefficients
```

---

## Method 1 — Ridge Regression

Ridge adds an L2 penalty:

```math
J(\beta)
=
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
+
\lambda
\sum_{j=1}^{d}\beta_j^2
```

The intercept is usually not included in the penalty in standard formulations.

### Effect

Increasing $\lambda$:

```text
λ ↑
 ↓
Stronger coefficient shrinkage
 ↓
Lower variance
 ↓
Potentially higher bias
```

Ridge is especially useful when predictors are correlated.

### Interview definition

> **Ridge Regression is Linear Regression with an L2 penalty on the coefficients that discourages large coefficient values and can reduce variance.**

---

## Method 2 — Lasso Regression

Lasso adds an L1 penalty:

```math
J(\beta)
=
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
+
\lambda
\sum_{j=1}^{d}|\beta_j|
```

### Effect

Lasso can push some coefficients exactly to zero.

Therefore it can perform a form of feature selection.

### Interview definition

> **Lasso Regression is Linear Regression with an L1 penalty that can shrink some coefficients to exactly zero, providing both regularization and feature selection.**

---

## Method 3 — Elastic Net

Elastic Net combines L1 and L2 penalties.

```math
J(\beta)
=
RSS
+
\lambda
\left(
\alpha\sum_{j=1}^{d}|\beta_j|
+
(1-\alpha)\sum_{j=1}^{d}\beta_j^2
\right)
```

The exact scaling convention can differ across formulations and libraries.

Elastic Net is useful when we want a combination of:

- Lasso-like sparsity
- Ridge-like coefficient shrinkage

---

## Effect on Model

```text
Regularization strength ↑
        ↓
Coefficient magnitude ↓
        ↓
Variance often ↓
        ↓
Bias can ↑
        ↓
Generalization may improve
```

---

# 20. Evaluation

## Relevant Metrics

For regression:

### Mean Absolute Error

```math
MAE
=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
```

MAE is easy to interpret and is less sensitive to large errors than MSE.

### Mean Squared Error

```math
MSE
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
```

MSE penalizes large errors more heavily.

### Root Mean Squared Error

```math
RMSE=\sqrt{MSE}
```

RMSE is in the same units as the target.

### $R^2$

```math
R^2
=
1-
\frac{RSS}{TSS}
```

Useful for describing fit relative to a mean baseline, but not sufficient by itself.

### Adjusted $R^2$

A commonly used adjusted form is:

```math
\bar{R}^2
=
1-
(1-R^2)
\frac{n-1}{n-d-1}
```

where $d$ is the number of predictors.

Adjusted $R^2$ introduces a penalty for adding predictors.

---

## Which Metrics Should I Use?

### General regression

Start with:

- MAE
- RMSE
- $R^2$

### When large errors are particularly costly

Use RMSE or MSE.

### When interpretability in the target's original unit matters

Use MAE or RMSE.

### Comparing explanatory fit

Use $R^2$ together with error metrics.

> **Do not select a metric only because it gives the highest numerical score. Select it based on the business or modeling objective.**

---

## Residual Diagnostics

Residual analysis is especially important for Linear Regression.

Useful plots include:

```text
Residuals vs Predicted
Residuals vs Feature
Q-Q plot
Scale-location plot
```

A useful residual pattern is approximately:

```text
Residual
   ↑
  + |   •   •
    | •   •   •
  0 |--------------------→ Prediction
    |   •   •   •
  - | •     •
```

Warning patterns include:

- Curved pattern → possible non-linearity
- Funnel shape → possible heteroscedasticity
- Clusters → possible missing structure
- Extreme points → possible outliers or influential observations

---

## Cross-Validation

For ordinary i.i.d. regression data, K-fold cross-validation is commonly used.

Typical process:

```text
Dataset
   ↓
Split into K folds
   ↓
Train on K-1 folds
   ↓
Validate on remaining fold
   ↓
Repeat K times
   ↓
Average validation performance
```

For time-dependent data, use a time-aware validation strategy instead of randomly mixing past and future observations.

Preprocessing steps such as imputation, encoding, and scaling should be inside the cross-validation workflow.

---

# 21. Advantages

## 1. Simple and Interpretable

**Why?**

The prediction is expressed directly through coefficients:

```math
\hat{y}=\beta_0+\beta_1x_1+\cdots+\beta_dx_d
```

This makes it easier to explain than many complex models.

---

## 2. Fast to Train and Predict

**Why?**

The model has a simple parametric form and can be solved efficiently using numerical linear algebra.

---

## 3. Strong Baseline

**Why?**

It provides a simple reference model against which more complex algorithms can be compared.

---

## 4. Useful for Understanding Relationships

**Why?**

Coefficients provide a direct description of the fitted linear relationship, subject to the assumptions and feature specification.

---

## 5. Works Well When the Linear Assumption Is Reasonable

**Why?**

If the target can be adequately approximated by a linear function of the selected features, Linear Regression can perform very well.

---

# 22. Disadvantages

## 1. Limited Representation of Non-Linear Relationships

**Why?**

A basic linear model cannot directly represent arbitrary curves or complex interactions.

---

## 2. Sensitive to Outliers

**Why?**

OLS squares residuals, so large errors receive disproportionately large weight.

---

## 3. Sensitive to Multicollinearity

**Why?**

Strongly correlated predictors can make individual coefficients unstable and harder to interpret.

---

## 4. Extrapolation Can Be Dangerous

**Why?**

A linear model can continue the learned line outside the training range even when the true relationship changes.

Example:

```text
Observed range: x = 10 to 50
```

A fitted line may be mathematically evaluated at:

```text
x = 100
```

but the prediction may be unreliable because it is outside the region supported by the training data.

---

# 23. Failure Modes

## When Does It Perform Poorly?

Linear Regression can perform poorly when:

- The true relationship is strongly non-linear.
- Important interaction effects are missing.
- Important variables are omitted.
- The data contains severe outliers.
- Multicollinearity is severe.
- Errors are strongly heteroscedastic and the evaluation relies on classical inference.
- The problem requires reliable extrapolation far beyond the training range.
- There is strong distribution shift.
- The training data is not representative of deployment data.

---

## Why Does It Fail?

### Non-Linearity

If:

```math
y\approx x^2
```

a model that only uses $x$ as a raw feature cannot represent the curved relationship.

### Omitted Variable Bias

If an important variable is missing and is related both to included predictors and the target, coefficient estimates can be distorted.

### Outliers

Large residuals receive high squared-error weight.

### Multicollinearity

The model cannot easily separate the contributions of highly correlated predictors.

### Distribution Shift

A relationship learned on one population may not hold on another.

---

## Warning Signs

```text
Train error → low
Validation error → high
```

or:

```text
Residual plot → systematic curve
```

or:

```text
Residual variance → increases with prediction
```

or:

```text
Coefficient values → unstable across folds
```

---

## How Can We Detect the Problem?

Use:

- Residual plots
- Cross-validation
- Train/validation/test comparison
- Feature distribution checks
- Multicollinearity diagnostics
- Outlier and leverage diagnostics
- Error analysis
- Time-based validation when appropriate
- Leakage checks

---

## Possible Solutions

Depending on the problem:

- Add meaningful features.
- Add polynomial or interaction terms.
- Use Ridge or Lasso.
- Investigate and handle outliers.
- Remove redundant predictors when justified.
- Collect more representative data.
- Use a non-linear model such as a Decision Tree or Random Forest.
- Improve the validation design.
- Address data leakage.

---

# 24. Algorithm-Specific Edge Cases

## Case 1 — Perfect Linear Fit

If every training point lies exactly on a fitted linear relationship:

```math
RSS=0
```

and:

```math
MSE=0
```

For an appropriate OLS training setup with an intercept and non-constant target:

```math
R^2=1
```

This does not guarantee perfect performance on future data.

---

## Case 2 — Constant Target

If all training targets are identical:

```math
y_1=y_2=\cdots=y_n=c
```

then the problem contains no target variation to explain.

A model predicting the constant value can achieve zero training squared error.

Metrics such as $R^2$ require care when the total target variance is zero.

---

## Case 3 — Perfect Multicollinearity

Suppose:

```math
x_3=x_1+x_2
```

Then the columns are linearly dependent.

The coefficient vector may not be uniquely identifiable by the ordinary inverse formula.

Use a numerically stable least-squares solver or regularization when appropriate.

---

## Case 4 — More Features Than Samples

When:

```math
d>n
```

the design matrix cannot have full column rank.

OLS solutions may be non-unique.

Regularization becomes especially useful in high-dimensional settings.

---

## Case 5 — Strong Outlier

One extreme observation may pull the fitted line toward itself.

Always distinguish:

```text
Outlier
Leverage point
Influential observation
```

They are not synonyms.

---

## Case 6 — Heteroscedastic Errors

Example:

```text
Small predicted values → small residual spread
Large predicted values  → large residual spread
```

This can lead to a funnel-shaped residual plot.

Coefficient estimation by OLS may still be useful, but classical standard errors may be unreliable without appropriate methods.

---

## Case 7 — Time-Series Data

Do not blindly use random train/test splitting.

For temporal prediction:

```text
Past → Train
Future → Validation/Test
```

Avoid allowing future information to leak into training.

---

## Case 8 — Extrapolation

The fitted equation can produce predictions outside the training range, but the prediction may be unreliable.

```text
Training region
|-------------|
      ↓
Reliable support

Outside region
      ↓
Extrapolation uncertainty
```

---

# 25. Practical Implementation — Scikit-Learn

## Import

```python
from sklearn.linear_model import LinearRegression
```

## Create Model

```python
model = LinearRegression(
    fit_intercept=True
)
```

Scikit-Learn's `LinearRegression` implements Ordinary Least Squares. It fits coefficients by minimizing the residual sum of squares. citeturn471969search1

## Train

```python
model.fit(X_train, y_train)
```

## Inspect Parameters

```python
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

## Predict

```python
y_pred = model.predict(X_test)
```

## Evaluate

```python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)
```

## Complete Example

```python
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Example data
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([50, 60, 70, 80, 90])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("Predictions:", y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
```

## Multiple Linear Regression

No special model class is required.

If:

```python
X_train.shape
```

is:

```text
(n_samples, n_features)
```

then:

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

learns one coefficient per feature.

## Class Probability?

Linear Regression does **not** produce class probabilities.

For classification tasks, use classification algorithms such as Logistic Regression instead.

## Important Scikit-Learn Settings

Current Scikit-Learn documentation includes settings such as:

```python
LinearRegression(
    fit_intercept=True,
    copy_X=True,
    tol=1e-6,
    n_jobs=None,
    positive=False
)
```

The exact supported behavior of settings such as `tol` and `n_jobs` depends on the implementation and input type; consult the version-specific documentation when these settings matter. citeturn471969search1

### `fit_intercept`

```python
fit_intercept=True
```

is the standard choice unless the data and model specification justify a zero intercept.

### `positive`

```python
positive=True
```

constrains coefficients to be non-negative and is supported for dense arrays. citeturn471969search1

### `n_jobs`

`n_jobs` is a computational setting. It only provides speedup in particular multi-target/sparse or positive-coefficient cases rather than making ordinary single-target dense regression automatically faster. citeturn471969search1

---

# 26. From-Scratch Implementation

> Implement the core OLS algorithm to understand the mathematics. This is for learning, not for replacing a production numerical linear algebra library.

## Simple Implementation Using the Normal Equation

```python
import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        # Add a column of ones for the intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Use a pseudo-inverse for a safer least-squares solution
        beta = np.linalg.pinv(X_b) @ y

        self.intercept_ = beta[0, 0]
        self.coef_ = beta[1:, 0]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction.")

        return self.intercept_ + X @ self.coef_
```

## Code ↔ Mathematics

| Code Component | Mathematical / Algorithmic Concept |
|---|---|
| `X_b = np.c_[...]` | Add the intercept column |
| `beta = ...` | Estimate OLS coefficients |
| `beta[0]` | $\beta_0$ |
| `beta[1:]` | $\beta_1,\ldots,\beta_d$ |
| `X @ self.coef_` | Weighted sum of features |
| `intercept + prediction` | $\hat{y}=\beta_0+X\beta$ |

## Important Implementation Details

### Why use the pseudo-inverse?

The simple formula:

```math
(X^TX)^{-1}X^Ty
```

requires an invertible $X^TX$.

Using:

```python
np.linalg.pinv(X_b) @ y
```

handles singular or rank-deficient cases more safely.

It is still a learning implementation and is not intended to replace optimized least-squares solvers.

---

## Gradient Descent From Scratch

A simple Batch Gradient Descent implementation:

```python
import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights_ = None
        self.bias_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights_ + self.bias_

            error = y_pred - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights_ + self.bias_
```

### Code ↔ Mathematics

The gradient update is based on:

```math
\beta
\leftarrow
\beta-\alpha\nabla J(\beta)
```

The code:

```python
self.weights_ -= self.learning_rate * dw
```

implements the parameter update.

---

# 27. Practical Workflow

```text
Raw Dataset
    ↓
EDA
    ↓
Train / Validation / Test Split
    ↓
Missing Value Handling
    ↓
Feature Engineering
    ↓
Encoding
    ↓
Scaling (if useful / required by the chosen model)
    ↓
Baseline Linear Regression
    ↓
Evaluate MAE / RMSE / R²
    ↓
Residual Diagnostics
    ↓
Cross-Validation
    ↓
Check Multicollinearity
    ↓
Try Ridge / Lasso if needed
    ↓
Compare with non-linear models
    ↓
Final Evaluation
    ↓
Interpretation
```

## Algorithm-Specific Considerations

Pay special attention to:

- Linearity
- Residual patterns
- Outliers
- Multicollinearity
- Feature engineering
- Extrapolation
- Leakage
- Appropriate validation strategy

A strong practical baseline is:

```text
Linear Regression
      ↓
Residual analysis
      ↓
Ridge / Lasso if needed
      ↓
Compare against a non-linear model
```

---

# 28. Comparison With Related Algorithms

## Linear Regression vs Ridge Regression

| Aspect | Linear Regression | Ridge |
|---|---|---|
| Core Idea | Minimize squared error | Minimize squared error + L2 penalty |
| Regularization | No | Yes |
| Coefficients | Can become large | Shrunk toward zero |
| Multicollinearity | Can be unstable | Usually more stable |
| Feature Selection | No | No exact zeroing in the usual formulation |
| Main Use Case | Simple OLS baseline | Stabilize coefficients / reduce variance |

## Linear Regression vs Lasso Regression

| Aspect | Linear Regression | Lasso |
|---|---|---|
| Core Idea | Minimize squared error | Squared error + L1 penalty |
| Regularization | No | Yes |
| Coefficients | Unconstrained | Some can become exactly zero |
| Feature Selection | No | Yes, potentially |
| Correlated Features | Can be unstable | Selection can be unstable among highly correlated features |
| Main Use Case | Standard OLS | Sparse models / feature selection |

## Linear Regression vs Polynomial Regression

| Aspect | Linear Regression | Polynomial Regression |
|---|---|---|
| Input features | Original features | Includes polynomial features |
| Relationship in original $x$ | Linear | Can be curved |
| Model linear in coefficients | Yes | Yes |
| Complexity | Lower | Higher |
| Main Risk | Underfitting non-linearity | Overfitting at high degree |

## Linear Regression vs Decision Tree

| Aspect | Linear Regression | Decision Tree |
|---|---|---|
| Model form | Global linear equation | Recursive rules |
| Non-linear relationships | Limited directly | Yes |
| Scaling | Usually not required for OLS | Usually not required |
| Interpretability | High | High for one small tree |
| Extrapolation | Can extrapolate mathematically | Generally poor at extrapolation |
| Main Strength | Simple linear relationships | Thresholds and interactions |

## Linear Regression vs Random Forest

| Aspect | Linear Regression | Random Forest |
|---|---|---|
| Core Idea | One linear function | Ensemble of trees |
| Non-linearity | Limited | Strong |
| Interpretability | Higher | Lower |
| Scaling | Usually not required | Usually not required |
| Multicollinearity | Important | Usually less problematic for prediction |
| Extrapolation | Can extrapolate linearly | Generally poor |
| Main Strength | Simple, interpretable baseline | Complex tabular patterns |

## Key Distinction

> **Linear Regression learns one global linear relationship; tree-based models learn non-linear relationships through partitions of the feature space.**

---

# 29. When Should I Use This Algorithm?

Use Linear Regression when:

- The target is continuous.
- A linear relationship is a reasonable approximation.
- Interpretability matters.
- You want a fast baseline.
- The number of features is manageable.
- You need coefficient-based explanations.
- You want to establish whether a simple relationship is already sufficient.

A typical example is:

```text
Predict house price
from area, bedrooms, age, and location features
```

when the transformed feature relationships are reasonably approximated by a linear model.

---

# 30. When Should I Avoid This Algorithm?

Consider another approach when:

- The relationship is strongly non-linear.
- Complex interactions dominate.
- The problem has severe multicollinearity and coefficient interpretation is central without regularization.
- Extreme outliers dominate the squared-error objective.
- Reliable extrapolation is required far beyond the observed training region.
- The feature-target relationship changes substantially across populations.
- A flexible tabular model is clearly needed after a strong baseline comparison.

Possible alternatives include:

- Ridge
- Lasso
- Elastic Net
- Polynomial Regression
- Decision Trees
- Random Forest
- Gradient Boosting

---

# 31. Algorithm Selection Guide

When facing a new ML problem:

```text
What type of problem?
        ↓
Regression
        ↓
Is the target relationship approximately linear?
        ↓
      Yes
        ↓
Need high interpretability / fast baseline?
        ↓
      Yes
        ↓
Linear Regression
        │
        ├── Multicollinearity / high variance?
        │          ↓
        │       Ridge
        │
        ├── Need sparse feature selection?
        │          ↓
        │       Lasso
        │
        └── Need both?
                   ↓
               Elastic Net

      No
        ↓
Try a non-linear model
        ↓
Decision Tree / Random Forest / Gradient Boosting
```

The correct choice should ultimately be based on validation performance, problem constraints, and interpretability requirements.

---

# 32. Common Misconceptions

## Misconception 1

> "Linear Regression means the relationship must be a straight line in every possible form."

**Correction:**

The model must be linear in its parameters. With transformed features such as $x^2$ or interaction terms, the model can represent non-linear relationships in the original input variables.

---

## Misconception 2

> "Linear Regression requires normally distributed features."

**Correction:**

Normality of the input features is not required for OLS coefficient estimation.

Normality assumptions, when used, usually concern the errors and mainly affect classical small-sample inference.

---

## Misconception 3

> "A high $R^2$ means the model is good."

**Correction:**

A high $R^2$ does not guarantee good out-of-sample performance, correct specification, causality, or lack of leakage.

Always examine validation performance and residual behavior.

---

## Misconception 4

> "Correlation between a feature and target proves the feature causes the target."

**Correction:**

A regression coefficient or correlation describes an observed relationship under the model. Causal interpretation requires additional assumptions and a suitable causal design.

---

## Misconception 5

> "Feature scaling is always required for Linear Regression."

**Correction:**

Basic OLS does not require standardization for the fitted predictions, although scaling may help numerical conditioning and is especially useful for regularized models and gradient-based optimization.

---

## Misconception 6

> "OLS can never overfit because it is simple."

**Correction:**

OLS can overfit when the feature space becomes large or highly flexible, especially with polynomial or interaction features and noisy predictors.

---

# 33. Common Implementation Mistakes

## Mistake 1 — Data Leakage

Bad workflow:

```text
Impute / scale entire dataset
        ↓
Train-test split
```

Better:

```text
Train-test split
      ↓
Fit preprocessing on training data
      ↓
Transform validation/test data
```

Use a `Pipeline` when appropriate.

---

## Mistake 2 — Removing the Intercept Without Reason

Setting:

```python
fit_intercept=False
```

forces the regression surface through the origin.

Do this only when the model specification justifies it.

---

## Mistake 3 — Using the Wrong Metric

Do not rely only on $R^2$.

Also examine:

- MAE
- RMSE
- residuals
- domain-specific error costs

---

## Mistake 4 — Ignoring Outliers

Because OLS squares residuals, extreme observations can strongly affect the fitted model.

Inspect unusual observations before trusting the coefficients.

---

## Mistake 5 — Ignoring Multicollinearity

A model can have reasonable predictive accuracy while having unstable coefficients.

If interpretation matters, investigate correlated predictors.

---

## Mistake 6 — Treating Coefficients as Causal Effects

A coefficient does not automatically imply a causal relationship.

---

## Mistake 7 — Evaluating on Training Data Only

A very low training error does not establish generalization.

Use validation or cross-validation and a final held-out test set where appropriate.

---

# 34. Interview Questions

## Basic

### Q1. What is Linear Regression?

**Answer:**

> **Linear Regression is a supervised parametric regression algorithm that models a continuous target as a linear function of one or more features and learns the coefficients by minimizing a loss such as the sum of squared errors.**

---

### Q2. What is the equation of Linear Regression?

**Answer:**

For one feature:

```math
\hat{y}=\beta_0+\beta_1x
```

For multiple features:

```math
\hat{y}
=
\beta_0+\beta_1x_1+\cdots+\beta_dx_d
```

---

### Q3. What is the difference between simple and multiple Linear Regression?

**Answer:**

> Simple Linear Regression uses one predictor variable, while Multiple Linear Regression uses two or more predictor variables.

---

### Q4. What is the main objective of Linear Regression?

**Answer:**

> In Ordinary Least Squares, the objective is to minimize the sum of squared residuals between observed and predicted target values.

---

### Q5. What is a residual?

**Answer:**

> A residual is the difference between the observed value and the predicted value.

```math
e_i=y_i-\hat{y}_i
```

---

### Q6. What is the Normal Equation?

**Answer:**

> The Normal Equation is the mathematical solution to the OLS problem. When $X^TX$ is invertible, it is written as:

```math
\hat{\beta}=(X^TX)^{-1}X^Ty
```

---

### Q7. Does Linear Regression require feature scaling?

**Answer:**

> Basic OLS does not require feature scaling for the fitted predictions, although scaling can improve numerical conditioning and is particularly useful for regularized or gradient-based linear models.

---

## Intermediate

### Q8. What are the assumptions of Linear Regression?

**Answer:**

> Important assumptions include a correct linear specification in the parameters, no perfect multicollinearity, an appropriate conditional mean assumption for unbiased coefficient estimation, and independent errors for standard inference. Constant error variance and normal residuals are mainly important for classical inference rather than for obtaining the basic OLS point estimates.

---

### Q9. Why do we square the residuals?

**Answer:**

> Squaring prevents positive and negative errors from cancelling, penalizes large errors more heavily, and produces a smooth convex objective that is mathematically convenient to optimize.

---

### Q10. What does the coefficient of a feature mean?

**Answer:**

> A coefficient represents the change in the model's predicted target for a one-unit increase in that feature, holding the other included features constant.

---

### Q11. What is $R^2$?

**Answer:**

> $R^2$ measures how much the model reduces squared error relative to a mean-only baseline.

```math
R^2=1-\frac{RSS}{TSS}
```

---

### Q12. What happens if features are highly correlated?

**Answer:**

> Multicollinearity can make individual coefficients unstable, increase their uncertainty, and make interpretation difficult. Prediction can still remain reasonable.

---

### Q13. Why is Linear Regression sensitive to outliers?

**Answer:**

> Because OLS minimizes squared residuals. A large residual receives a disproportionately large penalty and can pull the fitted line toward the outlier.

---

### Q14. What is the difference between Ridge and Linear Regression?

**Answer:**

> Ridge adds an L2 penalty to Linear Regression. This shrinks coefficients toward zero and can reduce variance, especially when features are correlated.

---

## Advanced

### Q15. Why is the OLS objective convex?

**Answer:**

> The squared-error objective is a quadratic function of the coefficient vector. Its Hessian is proportional to $X^TX$, which is positive semidefinite. Therefore the objective is convex.

---

### Q16. Why does multicollinearity cause unstable coefficients?

**Answer:**

> When predictors contain very similar information, the model has difficulty separating their individual contributions. Small changes in the data can therefore cause large changes in their estimated coefficients.

---

### Q17. Why should we avoid explicitly calculating $(X^TX)^{-1}$ in production code?

**Answer:**

> Explicit matrix inversion can be less numerically stable and less efficient than directly solving the least-squares system using stable numerical linear algebra methods.

---

### Q18. Can Linear Regression model non-linear relationships?

**Answer:**

> Basic Linear Regression cannot model arbitrary non-linear relationships in the original features. However, by creating polynomial, interaction, logarithmic, or other transformed features, we can fit a model that is still linear in its parameters but non-linear in the original inputs.

---

### Q19. What is Gradient Descent doing in Linear Regression?

**Answer:**

> Gradient Descent starts from an initial parameter vector and repeatedly moves the coefficients in the direction that decreases the objective.

```math
\beta
\leftarrow
\beta-\alpha\nabla J(\beta)
```

---

### Q20. Why can $R^2$ be negative on test data?

**Answer:**

> A negative test $R^2$ means the model's squared-error performance on the test set is worse than the mean-only baseline evaluated on that test set.

---

### Q21. What is the difference between prediction and inference?

**Answer:**

> Prediction focuses on generalization to unseen data, while statistical inference focuses on estimating relationships and uncertainty around coefficients under additional assumptions.

---

### Q22. Why does a high $R^2$ not prove causality?

**Answer:**

> $R^2$ measures predictive fit relative to a baseline. It does not establish that changing a predictor causes a change in the target.

---

# 35. Interview Follow-Up Drill

The first answer is rarely the end of the interview.

### Interviewer: "Why minimize squared error?"

**Your answer:**

> Squared error prevents positive and negative residuals from cancelling, penalizes large errors more strongly, and gives a smooth convex objective that can be solved efficiently.

---

### Interviewer: "What happens if we increase the number of features?"

**Your answer:**

> The model becomes more flexible and may fit the training data better, but unnecessary or noisy features can increase variance, worsen multicollinearity, and cause overfitting.

---

### Interviewer: "Does feature scaling matter here?"

**Your answer:**

> Scaling is not required for the basic OLS fitted predictions, but it can improve numerical conditioning and is especially important when using Ridge, Lasso, or Gradient Descent.

---

### Interviewer: "What happens if two features are highly correlated?"

**Your answer:**

> Their individual coefficients can become unstable because the model has difficulty separating their effects. Prediction may still be reasonable, but interpretation becomes less reliable.

---

### Interviewer: "Why not use Random Forest?"

**Your answer:**

> If the relationship is approximately linear and interpretability, speed, and coefficient-based explanation are important, Linear Regression is a useful baseline. Random Forest is more suitable when strong non-linear relationships and interactions need to be modeled.

---

### Interviewer: "What happens if there is a strong outlier?"

**Your answer:**

> OLS can be strongly affected because the residual is squared. I would investigate whether the observation is a data error, a legitimate rare case, a leverage point, or an influential observation before deciding how to handle it.

---

### Interviewer: "Can you solve Linear Regression without Gradient Descent?"

**Your answer:**

> Yes. Ordinary Least Squares has a direct least-squares solution characterized by the Normal Equations. In practice, numerical solvers are preferred over explicitly computing a matrix inverse.

---

# 36. Explain This Algorithm in an Interview

## 30-Second Explanation

> **Linear Regression is a supervised regression algorithm used to predict continuous values. It assumes that the target can be represented as a linear combination of the input features. During training, it learns the intercept and feature coefficients by minimizing the sum of squared residuals, usually using Ordinary Least Squares. Once the coefficients are learned, prediction is simply the weighted sum of the input features plus the intercept.**

---

## 1-Minute Explanation

> **Linear Regression models a continuous target as a linear function of one or more features. For multiple features, the model is $\hat{y}=\beta_0+\beta_1x_1+\cdots+\beta_dx_d$. The parameters are learned by minimizing the residual sum of squares, which is the sum of squared differences between the actual and predicted values. The OLS objective is convex, so it has a global optimum. The coefficients can be found through a least-squares solver or, conceptually, through the Normal Equation. Linear Regression is simple and interpretable, but it can struggle with strong non-linearity, outliers, multicollinearity, and distribution shift.**

---

## 3-Minute Explanation

> **Linear Regression is a supervised parametric algorithm for predicting a continuous target. In simple Linear Regression we fit $\hat{y}=\beta_0+\beta_1x$. In multiple Linear Regression we use several features: $\hat{y}=\beta_0+\beta_1x_1+\cdots+\beta_dx_d$.**
>
> **The training objective in Ordinary Least Squares is to minimize the residual sum of squares:**
>
> ```math
> RSS=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
> ```
>
> **The residual is $e_i=y_i-\hat{y}_i$. The squared loss makes the objective smooth and convex. In matrix form, the problem is $\min_\beta\|y-X\beta\|_2^2$. Setting the gradient to zero gives the Normal Equations $X^TX\beta=X^Ty$, and when the required inverse exists, the solution is $(X^TX)^{-1}X^Ty$. Production implementations generally use stable least-squares solvers rather than explicitly calculating the inverse.**
>
> **The coefficient $\beta_j$ describes how the model's prediction changes for a one-unit change in feature $x_j$, holding other included features fixed. However, a coefficient should not automatically be interpreted as a causal effect.**
>
> **Important assumptions and practical concerns include linear specification, appropriate conditional mean assumptions, no perfect multicollinearity, and independent observations for standard inference. Constant error variance and normality of residuals are especially relevant to classical inference.**
>
> **Linear Regression is useful because it is fast, interpretable, and a strong baseline. Its main limitations are difficulty representing complex non-linear relationships, sensitivity to outliers, and instability under severe multicollinearity. Ridge and Lasso address some of these issues by adding regularization.**

---

# 37. Key Takeaways

## Core Idea

> **Learn a line or hyperplane that predicts a continuous target by minimizing squared prediction errors.**

## Mathematical Idea

> **OLS minimizes the residual sum of squares, leading to the Normal Equations and a convex quadratic optimization problem.**

## Training Idea

> **Estimate the intercept and coefficients from the training data.**

## Prediction Idea

> **Apply the learned linear equation to a new feature vector.**

## Main Strength

> **Simple, fast, and interpretable.**

## Main Limitation

> **It cannot directly represent arbitrary non-linear relationships and can be sensitive to outliers and multicollinearity.**

## Most Important Hyperparameters

> **`fit_intercept` is the main model-specification setting in basic Scikit-Learn `LinearRegression`; regularized variants introduce more important predictive hyperparameters.**

## Most Important Assumption

> **The linear specification must be appropriate for the relationship being modeled, and standard inference requires additional assumptions about the errors and sampling process.**

## Most Important Interview Concept

> **OLS minimizes squared residuals, and the Normal Equation follows from setting the gradient of this convex quadratic objective to zero.**

---

# 38. Completion Checklist

Before marking this algorithm as complete, I should be able to answer:

- [ ] What problem does Linear Regression solve?
- [ ] Why do we need it?
- [ ] What is the formal definition?
- [ ] What is the core intuition?
- [ ] What is the difference between simple and multiple Linear Regression?
- [ ] What is the mathematical model?
- [ ] What is a residual?
- [ ] What is RSS?
- [ ] What is MSE?
- [ ] Why do we square the residuals?
- [ ] What is the Normal Equation?
- [ ] Can I derive the Normal Equation?
- [ ] What is Gradient Descent?
- [ ] What does the intercept mean?
- [ ] What does a coefficient mean?
- [ ] What is $R^2$?
- [ ] What are the assumptions of Linear Regression?
- [ ] Which assumptions are needed for estimation versus classical inference?
- [ ] Does feature scaling matter?
- [ ] How does Linear Regression behave with outliers?
- [ ] What is multicollinearity?
- [ ] Why can multicollinearity make coefficients unstable?
- [ ] Why can Linear Regression overfit?
- [ ] How can overfitting be controlled?
- [ ] What is Ridge Regression?
- [ ] What is Lasso Regression?
- [ ] What is Elastic Net?
- [ ] What are the computational costs?
- [ ] What happens when $d>n$?
- [ ] What happens if $X^TX$ is singular?
- [ ] Why should explicit matrix inversion usually be avoided?
- [ ] What are the strengths and weaknesses?
- [ ] When should I use Linear Regression?
- [ ] When should I avoid it?
- [ ] How does it compare with Ridge, Lasso, Decision Tree, and Random Forest?
- [ ] Can I implement it using Scikit-Learn?
- [ ] Can I explain `fit()` and `predict()`?
- [ ] Can I explain `coef_` and `intercept_`?
- [ ] Can I implement basic OLS from scratch?
- [ ] Can I explain the code line-by-line?
- [ ] Can I diagnose residual problems?
- [ ] Can I answer "why?" follow-ups?
- [ ] Can I explain it in 30 seconds, 1 minute, and 3 minutes?

---

# 39. References

- [GitHub Docs — Writing Mathematical Expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
- [Scikit-Learn — LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Scikit-Learn — Ordinary Least Squares and Ridge Regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge.html)
- [Scikit-Learn — Linear Models User Guide](https://scikit-learn.org/stable/modules/linear_model.html)
- [ISLR — An Introduction to Statistical Learning](https://www.statlearning.com/)
- [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)
