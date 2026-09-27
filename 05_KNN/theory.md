# K-Nearest Neighbors (KNN)

> Complete theory, intuition, mathematics, practical understanding, implementation, and interview preparation.

---

## 0. Learning Objectives

By the end of this chapter, I should be able to:

- Explain K-Nearest Neighbors intuitively.
- Give a formal, interview-ready definition.
- Explain why KNN is called a lazy learner, instance-based learner, and non-parametric algorithm.
- Explain the role of distance and neighborhood structure.
- Explain how KNN performs classification and regression.
- Derive and explain common distance metrics.
- Explain majority voting, distance-weighted voting, and neighbor averaging.
- Understand how the choice of $k$ affects bias and variance.
- Explain why feature scaling is important for KNN.
- Explain the curse of dimensionality and its effect on nearest-neighbor methods.
- Tune the important KNN hyperparameters.
- Explain training and prediction complexity.
- Handle ties, duplicate points, imbalanced classes, and other edge cases.
- Compare KNN with Logistic Regression, Decision Trees, SVM, and Random Forest.
- Implement KNN using Scikit-Learn.
- Understand a basic from-scratch implementation.
- Answer placement and interview follow-up questions.

---

# 1. Prerequisites

## Concepts I Should Know First

- Supervised learning
- Classification and regression
- Euclidean distance
- Basic linear algebra and vectors
- Mean and weighted mean
- Feature scaling
- Bias and variance
- Cross-validation
- Classification and regression evaluation metrics

## Connection With Previous Algorithms

KNN is fundamentally different from algorithms such as Linear Regression and Logistic Regression.

Linear and logistic models learn a set of global parameters such as coefficients. KNN does not learn a global equation during training.

Instead, KNN stores the training examples and uses them directly when a new sample arrives.

The core idea is:

```text
Training Data
    ↓
Store the observations
    ↓
New observation arrives
    ↓
Measure distance to training observations
    ↓
Find the k nearest observations
    ↓
Combine their target values
    ↓
Prediction
```

KNN is therefore called a **lazy learner** because most of the actual predictive computation is delayed until prediction time.

---

# 2. Why Do We Need This Algorithm?

## 2.1 The Problem

Many datasets contain a useful local structure:

> Observations that are close to each other in feature space often have similar target values.

For example, suppose we want to classify a customer as likely or unlikely to churn.

A new customer may be surrounded by historical customers with similar:

- Age
- Monthly charges
- Contract length
- Usage behavior

Instead of learning one global equation, KNN asks:

> **Which training examples are most similar to this new customer, and what do those nearby examples tell us?**

This makes KNN useful when local similarity is a meaningful representation of the problem.

## 2.2 Limitations of Previous Approaches

### Linear Models

A Linear Regression or Logistic Regression model assumes a specific global functional form.

For example:

```math
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2
```

or for binary classification:

```math
p(y=1 \mid x) =
\frac{1}{1+e^{-z}}
```

where $z$ is a linear combination of the features.

These models can work well, but the relationship may not be well described by one global linear structure.

### Highly Parametric Approaches

Some models impose stronger assumptions about the form of the relationship.

KNN takes a more local approach:

> Let nearby observations provide information about the new observation.

## 2.3 What Should a Better Approach Do?

A useful local method should:

- Capture non-linear decision boundaries.
- Adapt to local patterns in the data.
- Require few assumptions about the functional form.
- Work for both classification and regression.
- Be simple to understand conceptually.
- Allow the definition of similarity to be controlled through a distance metric.

## 2.4 Core Idea

> **K-Nearest Neighbors predicts a new observation using the target values of the $k$ training observations that are closest to it according to a chosen distance metric.**

The entire method can be summarized as:

```text
New Point
    ↓
Measure distance to training points
    ↓
Select k nearest points
    ↓
Classification → Vote
Regression    → Average
    ↓
Prediction
```

---

# 3. Formal Definition

## Definition

> **K-Nearest Neighbors (KNN) is a non-parametric, instance-based supervised learning algorithm that predicts the target of a new observation using the target values of its $k$ nearest training observations according to a specified distance metric. For classification, it commonly uses majority voting; for regression, it commonly uses the mean of the neighbors' target values.**

### Interview-ready one-line definition

> **KNN predicts a new sample from the labels or target values of its $k$ nearest training samples, where closeness is defined by a distance metric.**

## Algorithm Classification

| Property | Description |
|---|---|
| Learning Type | Supervised |
| Task | Classification and Regression |
| Parametricity | Non-parametric |
| Model Type | Instance-based / distance-based |
| Learning Approach | Lazy learning |
| Core Idea | Local neighborhood prediction |
| Base Data | Stored training observations |
| Main Control | Number of neighbors $k$ |
| Distance | Euclidean by default in common KNN settings |
| Feature Scaling | Usually important |
| Explicit Training Optimization | No |

### Important interview points

KNN is:

- **Non-parametric** because it does not assume a fixed finite-dimensional parametric form for the underlying relationship.
- **Instance-based** because predictions depend directly on stored training instances.
- **Lazy** because most computation happens when making predictions rather than during `fit()`.

---

# 4. Intuition

## 4.1 Core Intuition

Imagine a new point appears on a map.

Instead of building a formula for the entire map, KNN asks:

> **Which known points are closest to this new point?**

Suppose the $k=5$ nearest points have labels:

```text
A
A
B
A
B
```

Then:

```text
A → 3 votes
B → 2 votes
```

So KNN predicts:

```text
A
```

For regression, suppose the nearest target values are:

```text
70
75
80
72
78
```

Then KNN predicts their average:

```math
\hat{y}
=
\frac{70+75+80+72+78}{5}
=
75
```

The algorithm therefore relies on a local smoothness idea:

> Similar points tend to have similar outcomes.

## 4.2 Real-World Analogy

Suppose you move to a new neighborhood and want to estimate the monthly rent of your apartment.

You could build a complex global model, but another approach is:

1. Find nearby apartments that are similar in relevant features.
2. Look at their rents.
3. Average those rents.

That is the basic intuition of KNN regression.

For classification, replace "average rent" with "majority category."

The analogy is useful only when the feature representation actually makes geographic or feature-space closeness meaningful.

## 4.3 Simple Example

Consider a binary classification problem with two features:

| $x_1$ | $x_2$ | Class |
|---:|---:|---|
| 1 | 1 | A |
| 2 | 1 | A |
| 4 | 4 | B |
| 5 | 4 | B |
| 6 | 5 | B |

For a new point:

```text
x = (2, 2)
```

Suppose $k=3$.

The nearest points are likely:

```text
(1,1) → A
(2,1) → A
(4,4) → B
```

The votes are:

```text
A → 2
B → 1
```

Therefore:

```text
Prediction → A
```

## 4.4 Mental Model

> **KNN says: "Look at the nearest examples and let their outcomes guide the prediction."**

---

# 5. Problem Formulation

## Given

Training dataset:

$$
D = \{(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)\}
$$

Where:

- $x_i \in \mathbb{R}^d$ = feature vector of training sample $i$
- $y_i$ = target associated with sample $i$
- $n$ = number of training samples
- $d$ = number of features

For a new observation:

$$
x^\ast \in \mathbb{R}^d
$$

we calculate the distance between $x^\ast$ and every relevant training observation.

Let:

$$
N_k(x^\ast)
$$

represent the set of $k$ nearest training observations to $x^\ast$.

## Goal

Use the local neighborhood $N_k(x^\ast)$ to estimate the target of $x^\ast$.

## Input

The model receives:

- Training features $X$
- Training targets $y$
- A new sample $x^\ast$
- A chosen value of $k$
- A distance metric
- Optionally, a neighbor weighting scheme

## Output

For classification:

- A predicted class
- Optionally, class probabilities or neighbor vote proportions

For regression:

- A predicted continuous value

---

# 6. Mathematical Foundation

## 6.1 Model Representation

Unlike Linear Regression, KNN does not learn a global equation such as:

$$
\hat{y} = \beta_0 + \beta_1x_1 + \cdots + \beta_dx_d
$$

Instead, the prediction is based on a neighborhood around the query point.

Define the distance:

$$
d(x^\ast,x_i)
$$

between the query point $x^\ast$ and training point $x_i$.

The algorithm selects the $k$ training observations having the smallest distances.

Therefore, the model can be viewed as:

```text
Query point
    ↓
Distance function
    ↓
Nearest-neighbor set
    ↓
Aggregation of target values
    ↓
Prediction
```

## 6.2 Objective

KNN does not optimize a single global parameterized objective during `fit()` in the same way that Linear Regression or Logistic Regression does.

Its predictive objective is local:

> Find the $k$ closest training observations and use them to estimate the target.

The exact prediction rule depends on whether the task is classification or regression.

## 6.3 Loss / Cost / Objective Function

### Distance Function

The most common distance is Euclidean distance.

For two points:

$$
x=(x_1,x_2,\ldots,x_d)
$$

and

$$
z=(z_1,z_2,\ldots,z_d)
$$

Euclidean distance is:

```math
d(x,z)
=
\sqrt{
\sum_{j=1}^{d}(x_j-z_j)^2
}
```

### Minkowski Distance

A more general family is:

```math
d_p(x,z)
=
\left(
\sum_{j=1}^{d}|x_j-z_j|^p
\right)^{1/p}
```

Special cases include:

```math
p=1
    ↓
Manhattan distance

p=2
    ↓
Euclidean distance

p \to \infty
    ↓
Chebyshev distance
```

For $p=1$:

```math
d_1(x,z)
=
\sum_{j=1}^{d}|x_j-z_j|
```

For $p=2$:

```math
d_2(x,z)
=
\sqrt{
\sum_{j=1}^{d}(x_j-z_j)^2
}
```

## 6.4 Why This Objective Function?

Distance is not the final prediction objective in KNN. It is the mechanism used to define similarity.

A good distance function should represent the type of similarity that matters for the task.

For example, if one feature is measured in thousands while another is measured between $0$ and $1$, Euclidean distance can become dominated by the large-scale feature.

That is why feature scaling is often essential for KNN.

## 6.5 Optimization

KNN does not perform gradient descent.

It also does not usually solve for coefficients using a closed-form equation.

Instead:

1. Store the training examples.
2. Receive a query point.
3. Compute distances.
4. Identify the nearest $k$ observations.
5. Aggregate their targets.

The expensive part is therefore usually **prediction**, not fitting.

## 6.6 Derivation

### Step 1 — Euclidean Distance

For query point $x^\ast$ and training point $x_i$:

```math
d_i
=
\sqrt{
\sum_{j=1}^{d}(x^\ast_j-x_{ij})^2
}
```

### Step 2 — Rank the Distances

Compute:

$$
d_1,d_2,\ldots,d_n
$$

and rank them from smallest to largest.

### Step 3 — Select the $k$ Neighbors

Let the indices of the $k$ smallest distances be:

$$
i_1,i_2,\ldots,i_k
$$

Then:

```math
N_k(x^\ast)
=
\{i_1,i_2,\ldots,i_k\}
```

### Step 4 — Classification

For class $c$, define:

```math
V_c
=
\sum_{i\in N_k(x^\ast)}
\mathbf{1}\{y_i=c\}
```

The predicted class is the class with the largest vote count:

```math
\hat{y}
=
\underset{c}{\mathrm{arg\,max}}
\;
V_c
```

### Step 5 — Regression

The standard unweighted prediction is:

```math
\hat{y}
=
\frac{1}{k}
\sum_{i\in N_k(x^\ast)} y_i
```

### Step 6 — Distance-Weighted Regression

A common weighted version is:

```math
\hat{y}
=
\frac{
\sum_{i\in N_k(x^\ast)} w_i y_i
}{
\sum_{i\in N_k(x^\ast)} w_i
}
```

where $w_i$ gives greater influence to closer neighbors.

A common idea is:

$$
w_i \propto \frac{1}{d_i}
$$

when the distance is non-zero.

For classification, the same idea can be used to give closer neighbors more voting influence.

## 6.7 Important Mathematical Properties

- Convex / Non-convex: KNN is not naturally formulated as one global convex optimization problem.
- Differentiable / Non-differentiable: the nearest-neighbor selection is discrete.
- Closed-form / Iterative: prediction is based on distance computation and neighbor selection, not coefficient optimization.
- Local vs Global Optimum: KNN is fundamentally local rather than an optimization problem with a global parameter optimum.
- Parametric / Non-parametric: non-parametric.
- Main statistical idea: local approximation using nearby observations.
- Main geometric idea: distance defines neighborhood structure.

---

# 7. Geometric / Visual Understanding

## 7.1 What Does the Data Look Like?

Each observation is a point in feature space.

For two features:

```text
Feature 2
   ↑
   |
  B     B
     B
----------------------→ Feature 1
 A   A
      A
```

A new point is placed into this feature space.

KNN looks around that point and asks:

> Which training points are closest?

## 7.2 What Does the Model Learn?

KNN does not learn a global line or a global set of coefficients.

Instead, it effectively memorizes the training data and uses:

- Training feature vectors
- Training target values
- The distance metric
- The value of $k$
- The neighbor weighting rule

The decision regions are determined by the local arrangement of training points.

For $k=1$, the feature space can become highly fragmented because each query follows the class of its nearest training point.

For larger $k$, the predictions become smoother.

## 7.3 Effect of Model Complexity

For KNN, the main complexity control is $k$.

### Small $k$

Example:

```text
k = 1
```

Behavior:

- Very local decisions
- Low bias
- High variance
- Sensitive to noise
- Highly flexible

### Large $k$

Example:

```text
k = 50
```

Behavior:

- Larger neighborhoods
- Higher bias
- Lower variance
- Smoother decision boundaries
- More stable predictions

### Diagram

```text
Small k
→ Small neighborhood
→ Highly flexible model
→ Low bias
→ High variance

Large k
→ Large neighborhood
→ Smoother model
→ Higher bias
→ Lower variance
```

---

# 8. How the Algorithm Works

## Step-by-Step

### Step 1 — Store the Training Data

Unlike many algorithms, KNN does not need to estimate a large set of model coefficients.

Conceptually, it stores:

```text
X_train
y_train
```

along with the configuration required for neighbor search.

### Step 2 — Receive a New Sample

Suppose a new observation arrives:

```text
x_new
```

The algorithm must determine which training observations are closest to this sample.

### Step 3 — Calculate Distances

For every relevant training sample, calculate:

$$
d(x_{\text{new}},x_i)
$$

using the chosen distance metric.

### Step 4 — Select the $k$ Nearest Neighbors

Sort or otherwise select the smallest distances.

The $k$ observations with the smallest distances become the neighborhood.

### Step 5 — Aggregate the Neighbor Targets

For classification:

```text
Count votes
    ↓
Majority class
```

For regression:

```text
Average target values
    ↓
Predicted value
```

### Step 6 — Return the Prediction

The final neighborhood estimate becomes the prediction for the query sample.

## Algorithm Flow

```text
Training Data
    ↓
Store training observations
    ↓
New Query Point
    ↓
Compute distances
    ↓
Find k nearest neighbors
    ↓
Classification → Majority Vote
Regression     → Mean / Weighted Mean
    ↓
Prediction
```

## Pseudocode

```text
Input:
    Training data D
    Number of neighbors k
    Distance metric
    Weighting rule

For each new sample x:

    1. Compute the distance from x to each training sample.

    2. Rank the training samples by distance.

    3. Select the k nearest samples.

    4. If classification:
           Count the class votes.
           Return the majority class.

       If regression:
           Average the target values.
           Return the average.

    5. Optionally use distance-based weights so closer neighbors
       have greater influence.
```

---

# 9. Training vs Prediction

## 9.1 Training Phase

When calling:

```python
model.fit(X_train, y_train)
```

KNN generally performs little statistical optimization.

Conceptually:

```text
X_train + y_train
        ↓
Store training observations
        ↓
Prepare neighbor-search structure if applicable
        ↓
Fitted KNN model
```

This is why KNN is called a **lazy learner**.

## 9.2 Prediction Phase

When calling:

```python
y_pred = model.predict(X_test)
```

the model performs the main computation:

```text
X_new
   ↓
Compute distances
   ↓
Find nearest neighbors
   ↓
Aggregate neighbor targets
   ↓
Prediction
```

## 9.3 What Does the Model Actually Learn?

KNN does not mainly learn coefficients.

It retains:

- Training feature vectors
- Training target values
- Number of neighbors
- Distance metric
- Distance parameters
- Weighting strategy
- Neighbor-search configuration

The predictive knowledge is primarily contained in the stored observations and their geometry.

## 9.4 What Is Stored After Training?

Conceptually:

```text
KNN Model
│
├── Training feature vectors
├── Training target values
├── Number of neighbors
├── Distance metric
├── Weighting rule
└── Neighbor-search configuration
```

Depending on the implementation and selected search algorithm, additional data structures may be built to accelerate neighbor queries.

---

# 10. Worked Example

## Dataset

Consider a binary classification problem.

| $x_1$ | $x_2$ | Class |
|---:|---:|---|
| 1 | 1 | A |
| 2 | 1 | A |
| 2 | 2 | A |
| 4 | 4 | B |
| 5 | 4 | B |
| 5 | 5 | B |

Suppose the new point is:

$$
x^\ast=(3,3)
$$

and:

$$
k=3
$$

## Step 1 — Calculate Distances

Using Euclidean distance:

```math
d(x^\ast,x)
=
\sqrt{(x_1^\ast-x_1)^2+(x_2^\ast-x_2)^2}
```

Distance to $(2,2)$:

```math
d_1
=
\sqrt{(3-2)^2+(3-2)^2}
=
\sqrt{2}
```

Distance to $(4,4)$:

```math
d_2
=
\sqrt{(3-4)^2+(3-4)^2}
=
\sqrt{2}
```

Distance to $(2,1)$:

```math
d_3
=
\sqrt{(3-2)^2+(3-1)^2}
=
\sqrt{5}
```

Distance to $(5,4)$:

```math
d_4
=
\sqrt{(3-5)^2+(3-4)^2}
=
\sqrt{5}
```

The remaining distances are larger.

## Step 2 — Select the $k=3$ Neighbors

The three nearest observations are:

```text
(2,2) → A
(4,4) → B
(2,1) → A
```

## Step 3 — Count Votes

```text
A → 2 votes
B → 1 vote
```

## Step 4 — Final Prediction

Therefore:

```text
Prediction → A
```

## Step 5 — What If k Changes?

If:

```text
k = 1
```

the prediction depends only on the nearest point.

If:

```text
k = 5
```

the prediction uses a much larger neighborhood and may change.

This demonstrates why choosing $k$ is a bias-variance trade-off.

## Final Result

```text
New Point
    ↓
Calculate distances
    ↓
Select nearest 3 points
    ↓
A, B, A
    ↓
Majority Vote
    ↓
A
```

> Goal: I should be able to manually calculate distances, identify the nearest $k$ observations, and produce the prediction.

---

# 11. Important Concepts & Terminology

## 11.1 K-Nearest Neighbors

**Definition:** KNN predicts a new sample using the target values of the $k$ nearest training samples.

**Why it matters:** It is the central mechanism of the algorithm.

## 11.2 Neighbor

**Definition:** A neighbor is a training observation that is close to a query observation according to the selected distance metric.

**Why it matters:** The prediction depends directly on the selected neighbors.

## 11.3 $k$

**Definition:** $k$ is the number of nearest training observations used to make a prediction.

**Why it matters:** It controls how local or smooth the model is.

## 11.4 Distance Metric

**Definition:** A distance metric defines how similarity or closeness between observations is measured.

**Why it matters:** A poor distance definition can produce poor neighbors even when the prediction rule itself is correct.

## 11.5 Lazy Learner

**Definition:** A lazy learner performs relatively little computation during training and delays most learning-related computation until prediction time.

**Why it matters:** KNN usually has very low fitting cost but potentially high prediction cost.

## 11.6 Instance-Based Learning

**Definition:** Instance-based learning predicts using stored examples rather than primarily constructing a global parametric model.

**Why it matters:** KNN is strongly dependent on the training instances.

## 11.7 Non-Parametric Model

**Definition:** A non-parametric model does not assume a fixed finite-dimensional parametric form for the relationship between inputs and outputs.

**Why it matters:** KNN can represent complex local relationships without assuming a linear or other fixed global equation.

## 11.8 Majority Voting

**Definition:** For classification, majority voting assigns the class that receives the largest number of votes among the $k$ nearest neighbors.

**Why it matters:** It is the standard KNN classification rule.

## 11.9 Weighted Voting

**Definition:** Weighted voting gives different influence to neighbors, often assigning greater weight to closer observations.

**Why it matters:** A very close neighbor can receive more influence than a farther neighbor.

## 11.10 Locality

**Definition:** Locality is the assumption that nearby observations in feature space tend to have similar target values.

**Why it matters:** This is one of the most important assumptions behind KNN.

## 11.11 Curse of Dimensionality

**Definition:** The curse of dimensionality describes the difficulties that arise as the number of dimensions increases, including sparse data and less informative distance relationships.

**Why it matters:** Distance-based methods often become less effective in high-dimensional spaces.

---

# 12. Parameters vs Hyperparameters

## Parameters

Traditional model parameters are values learned from the data through an optimization procedure.

Examples in Linear Regression include:

- $\beta_0$
- $\beta_1,\beta_2,\ldots,\beta_d$

KNN is different.

It does not learn a conventional set of global coefficients.

## Hyperparameters

**Definition:** Hyperparameters are settings chosen by the practitioner before or during training and are not learned as ordinary model parameters.

### KNN Examples

- `n_neighbors`
- `weights`
- `metric`
- `p`
- `algorithm`
- `leaf_size`
- `metric_params`

## Key Difference

| Parameters | Hyperparameters |
|---|---|
| Usually learned from training data | Chosen before or during model selection |
| Define a fitted parametric model | Control how KNN performs neighbor search and prediction |
| Example: model coefficient | Example: `n_neighbors` |
| KNN has no conventional learned coefficients | KNN depends heavily on hyperparameter choices |

### Interview point

> **KNN has no conventional parameter-learning step like Linear Regression; the main model controls are hyperparameters such as $k$, the distance metric, and the weighting rule.**

---

# 13. Important Hyperparameters

| Hyperparameter | Meaning | Increase → | Decrease → | Main Effect |
|---|---|---|---|---|
| `n_neighbors` | Number of neighbors $k$ | Smoother, lower variance, higher bias | More local, higher variance, lower bias | Controls locality |
| `weights` | Neighbor influence | `"distance"` gives more influence to nearby points | `"uniform"` gives equal influence | Controls voting strength |
| `metric` | Distance definition | Depends on chosen metric | Depends on chosen metric | Defines similarity |
| `p` | Minkowski power | Changes the geometry of distance | Changes the geometry of distance | Controls Minkowski metric |
| `algorithm` | Neighbor-search strategy | May use tree-based search | May use brute force | Controls computation |
| `leaf_size` | Leaf size for tree-based search | Can reduce tree depth | Can increase tree depth | Speed/memory trade-off |
| `n_jobs` | Parallel neighbor-search jobs | More parallelism | Less parallelism | Computational speed |
| `metric_params` | Additional metric configuration | Metric-specific | Metric-specific | Customizes distance calculation |

## Most Important Hyperparameters

Focus especially on:

1. `n_neighbors`
2. `weights`
3. `metric`
4. `p`
5. `algorithm`

## `n_neighbors`

`n_neighbors` controls $k$.

### Smaller $k$

```text
k ↓
↓
Smaller neighborhood
↓
More flexible model
↓
Lower bias
↓
Higher variance
```

### Larger $k$

```text
k ↑
↓
Larger neighborhood
↓
Smoother model
↓
Higher bias
↓
Lower variance
```

### Important Interview Point

> A very small $k$ can make KNN sensitive to noise, while a very large $k$ can oversmooth local patterns.

## `weights`

Common choices include:

```text
weights="uniform"
```

Every selected neighbor has equal influence.

Or:

```text
weights="distance"
```

Closer neighbors have greater influence.

Conceptually:

```text
Uniform:
Neighbor 1 → equal weight
Neighbor 2 → equal weight
Neighbor 3 → equal weight

Distance:
Nearest neighbor → greater weight
Farther neighbor → smaller weight
```

## `metric`

The metric determines what "near" means.

Common choices include:

- Euclidean
- Manhattan
- Minkowski

The correct metric depends on the structure and meaning of the features.

## `p`

For the Minkowski metric:

```math
d_p(x,z)
=
\left(
\sum_{j=1}^{d}|x_j-z_j|^p
\right)^{1/p}
```

Examples:

```text
p = 1 → Manhattan distance
p = 2 → Euclidean distance
```

## `algorithm`

Scikit-Learn supports:

- `auto`
- `ball_tree`
- `kd_tree`
- `brute`

`auto` attempts to select an appropriate method based on the fitted data and configuration.

## Hyperparameter Interactions

Hyperparameters should not be treated independently.

For example:

```text
k ↓
    ↓
More local decisions
    ↓
Variance ↑
Bias ↓
```

while:

```text
k ↑
    ↓
Smoother decisions
    ↓
Variance ↓
Bias ↑
```

Metric and scaling interact strongly:

```text
Poor scaling
    +
Distance-based metric
    ↓
Bad neighborhood structure
    ↓
Poor predictions
```

Therefore, KNN hyperparameter tuning should be performed together with an appropriate preprocessing workflow.

---

# 14. Bias-Variance & Generalization

## Bias

Bias is the error caused by a model being systematically too simple or restrictive.

In KNN:

- Very large $k$ tends to increase bias.
- Very small $k$ tends to reduce bias.

Why?

A large neighborhood averages over more observations and can ignore local structure.

## Variance

Variance measures how much model predictions change when the training data changes.

In KNN:

- Small $k$ tends to produce high variance.
- Large $k$ tends to produce lower variance.

Why?

With $k=1$, a small change in training data can change the nearest neighbor and therefore the prediction.

## Overfitting

### Why Can KNN Overfit?

KNN can overfit when:

- $k$ is too small.
- The training data contains noise.
- Features contain irrelevant information.
- The distance metric does not represent meaningful similarity.
- The data has outliers or mislabeled observations.

With $k=1$, a noisy training point can completely determine the prediction in its neighborhood.

### Signs of Overfitting

```text
Training performance → very high
Validation performance → significantly lower
```

A very flexible KNN model may closely follow local noise.

## Underfitting

### Why Can KNN Underfit?

Underfitting may occur when:

- $k$ is too large.
- The neighborhood becomes too broad.
- Important local structure is averaged away.
- Features are not informative.

### Signs of Underfitting

```text
Training performance → poor
Validation performance → also poor
```

## Bias-Variance Trade-off

```text
Model Complexity

Very small k
    ↓
Underfitting? No
    ↓
High flexibility
    ↓
High variance

Balanced k
    ↓
Good generalization

Very large k
    ↓
Low flexibility
    ↓
High bias
    ↓
Underfitting
```

## Controlling Overfitting

- Increase `n_neighbors` when validation results indicate excessive variance.
- Improve feature quality.
- Scale numerical features appropriately.
- Remove irrelevant features where justified.
- Tune the distance metric.
- Use cross-validation.

## Controlling Underfitting

- Reduce `n_neighbors`.
- Use a more suitable distance metric.
- Improve feature representation.
- Remove excessive smoothing.
- Check whether local similarity is actually meaningful for the problem.

---

# 15. Assumptions

| Assumption | Required? | Why? | What If Violated? |
|---|---|---|---|
| Nearby observations tend to have similar targets | Important | This is the main locality assumption | Predictions may be poor |
| Distance is meaningful | Important | KNN depends on ranking observations by distance | Wrong neighbors may be selected |
| Features are on comparable scales | Practically important | Large-scale features can dominate distance | Neighborhood structure becomes distorted |
| Data points are representative | Important | Neighbors must reflect the population being predicted | Generalization may fail |
| Linear relationship | No | KNN does not assume a global linear form | No direct issue |
| Normal distribution | No | KNN does not require Gaussian features | No direct issue |
| Low-dimensional data | Not required, but often helpful | Distance tends to become less informative in high dimensions | Curse of dimensionality |
| Independent observations | Preferable for standard evaluation | Dependence can distort validation results | Evaluation may be misleading |

## Important Interview Distinction

Do not say:

> "KNN has no assumptions."

A better answer is:

> **KNN does not require a specific global functional form such as linearity or normality, but it relies strongly on meaningful distance, local similarity, appropriate feature representation, and representative data.**

The most important assumption is:

> **Points that are close in feature space should tend to have similar target values.**

---

# 16. Data Preprocessing

## 16.1 Missing Values

KNN requires a distance calculation.

If a feature is missing, the distance may not be directly computable for a standard implementation.

A practical workflow is:

- Inspect missing values.
- Impute missing numerical values when appropriate.
- Encode categorical features appropriately.
- Fit preprocessing only on training data.
- Use a `Pipeline` to prevent leakage.

## 16.2 Feature Scaling

**Required?** Usually yes for numerical features when using distance-based KNN.

**Why?**

Consider two features:

```text
Age      → 18 to 80
Income   → 20,000 to 2,00,000
```

Without scaling, the income feature can dominate Euclidean distance.

For standardized features:

```math
z_j
=
\frac{x_j-\mu_j}{\sigma_j}
```

the features are placed on a more comparable numerical scale.

### Interview answer

> **Feature scaling is usually important for KNN because the algorithm relies on distances. A feature with a much larger numerical scale can dominate the distance calculation and distort the neighborhood.**

## 16.3 Categorical Variables

KNN's standard numerical distance metrics do not automatically understand nominal categories.

For example:

```text
City:
Mumbai
Delhi
Pune
```

Encoding these as:

```text
Mumbai → 1
Delhi  → 2
Pune   → 3
```

creates an artificial numerical ordering.

Better choices depend on the implementation and problem:

- One-hot encoding for nominal categories in many workflows.
- Carefully chosen ordinal encoding when an actual order exists.
- Specialized distance measures for mixed data when appropriate.

## 16.4 Outliers

KNN can be sensitive to outliers because an unusual point can become a nearest neighbor for some query samples.

Outliers can:

- Distort local neighborhoods.
- Affect distance-based predictions.
- Cause misleading nearest neighbors.

However, whether an observation is an outlier should be determined using domain knowledge and data quality analysis.

## 16.5 Multicollinearity

KNN does not estimate linear coefficients, so multicollinearity does not cause the same coefficient instability seen in ordinary least squares.

However, highly redundant features can still affect distances.

For example:

```text
Feature A
Feature B
Feature C

All carry nearly the same information
```

Then that information can be effectively counted multiple times in the distance calculation.

Therefore:

> KNN does not have the same multicollinearity problem as Linear Regression, but redundant features can still distort the geometry of the feature space.

## 16.6 Feature Engineering

Feature engineering can greatly affect KNN because the geometry of the feature space determines which points are considered neighbors.

Useful transformations may include:

- Creating domain-relevant features.
- Scaling numerical variables.
- Removing irrelevant variables.
- Reducing redundant variables.
- Applying dimensionality reduction when appropriate.

The important principle is:

> **In KNN, feature representation defines the neighborhood structure.**

---

# 17. Model Complexity

## What Controls Complexity?

The main complexity control is:

```text
k = n_neighbors
```

Other important controls include:

- Distance metric
- Feature representation
- Weighting method
- Feature selection

## Simple Model

A relatively simple KNN model may use:

```text
Large k
```

Behavior:

- Smooth predictions
- Lower variance
- Higher bias
- Less sensitivity to individual observations

## Complex Model

A highly flexible KNN model may use:

```text
Small k
```

Behavior:

- Very local predictions
- Lower bias
- Higher variance
- Greater sensitivity to noise

## Effect on Generalization

The typical pattern is:

```text
k very small
    ↓
Training error often low
Validation error may be high

k increases
    ↓
Validation error may decrease

k becomes too large
    ↓
Validation error may increase again
```

The practical goal is to choose $k$ using validation or cross-validation rather than selecting it arbitrarily.

---

# 18. Computational Complexity

## Training Complexity

For a simple brute-force KNN implementation, the fitting stage is mainly data storage.

A high-level view is:

$$
O(nd)
$$

or effectively storage/copy cost proportional to the size of the training matrix.

The exact cost depends on the implementation.

For tree-based neighbor-search structures, additional preprocessing can be required to build the search structure.

## Why?

KNN does not usually optimize a large parameter vector during fitting.

Most computation happens later when the model receives a new query point.

## Prediction Complexity

For brute-force search, a query point is compared with all $n$ training observations across $d$ features:

$$
O(nd)
$$

per query point for distance computation.

If the distances are sorted completely, sorting can add:

$$
O(n\log n)
$$

although implementations can use more efficient neighbor-selection strategies.

Therefore, a simple interview answer is:

> **Brute-force KNN prediction is approximately $O(nd)$ per query for distance computation, with additional cost for selecting the nearest neighbors.**

## Space Complexity

The model stores the training data:

$$
O(nd)
$$

Additional memory may be required for a neighbor-search data structure.

## Scalability

### More Samples

As $n$ increases:

- Brute-force prediction becomes more expensive.
- More memory is required.
- Neighbor search can become slower.

### More Features

As $d$ increases:

- Distance computation becomes more expensive.
- Distance can become less informative.
- The curse of dimensionality can reduce model quality.

### High-Dimensional Data

KNN can struggle in very high-dimensional spaces because:

- Data becomes sparse.
- Many observations may appear similarly distant.
- The distinction between nearest and farthest points can become less useful.

This is called the **curse of dimensionality**.

Possible approaches include:

- Feature selection
- Dimensionality reduction
- Domain-specific feature engineering
- Alternative models when local distance is no longer reliable

---

# 19. Regularization / Optimization Improvements

KNN does not have regularization in the same coefficient-penalty sense as Ridge or Lasso.

However, its flexibility can be controlled.

## Why Is It Needed?

The main risk is excessive sensitivity to individual training observations.

This can happen when $k$ is too small.

## Method 1 — Increase $k$

Increasing `n_neighbors` makes the prediction depend on more observations.

```text
k ↑
    ↓
Larger neighborhood
    ↓
Smoother prediction
    ↓
Lower variance
    ↓
Potentially less overfitting
```

## Method 2 — Use Distance Weighting

Instead of treating all neighbors equally, use:

```python
weights="distance"
```

This gives closer neighbors greater influence.

Conceptually:

$$
w_i \propto \frac{1}{d_i}
$$

The exact implementation behavior should be understood from the library documentation.

## Method 3 — Improve the Feature Space

A better feature representation can be more important than changing the model itself.

Possible improvements:

- Feature scaling
- Feature selection
- Removing noisy features
- Dimensionality reduction
- Better distance metric

## Effect on Model

```text
Better feature space
        +
Appropriate k
        +
Suitable distance metric
        ↓
Better neighborhoods
        ↓
Better predictions
```

---

# 20. Evaluation

## Relevant Metrics

### Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

### Regression

- MAE
- MSE
- RMSE
- $R^2$

## Which Metrics Should I Use?

### Balanced classification

Accuracy can be useful when class proportions and error costs are reasonably balanced.

### Imbalanced classification

Accuracy can hide poor minority-class performance.

Use metrics such as:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC

depending on the business cost of false positives and false negatives.

### Regression

- MAE is easy to interpret and less sensitive to large errors than MSE.
- MSE penalizes large errors more strongly.
- RMSE is in the target's original unit.
- $R^2$ compares performance against a baseline but should not be used alone.

## Cross-Validation

Cross-validation is especially important for selecting $k$ and comparing distance metrics.

A common pattern is:

```text
Candidate k values
    ↓
Cross-validation
    ↓
Compare validation performance
    ↓
Select suitable k
    ↓
Retrain on available training data
```

For example:

```text
k = 1, 3, 5, 7, 9, 11
```

can be evaluated using cross-validation.

### Important Rule

> **Preprocessing must be fitted inside the cross-validation workflow to avoid data leakage.**

For KNN, this is particularly important because scaling directly changes the distance calculations.

---

# 21. Advantages

## 1. Simple Concept

**Why?**

The basic algorithm is easy to understand:

```text
Find nearby points
    ↓
Use their outcomes
```

## 2. Can Model Non-Linear Relationships

**Why?**

KNN does not assume a global linear relationship. The local neighborhood can produce complex decision boundaries.

## 3. Few Distributional Assumptions

**Why?**

KNN does not require the features to follow a normal distribution or satisfy a global linear functional form.

## 4. Naturally Supports Local Patterns

**Why?**

Predictions are based on nearby observations rather than one global equation.

## 5. Works for Classification and Regression

**Why?**

The same neighbor-search mechanism can be followed by majority voting or target averaging.

---

# 22. Disadvantages

## 1. Prediction Can Be Expensive

**Why?**

The model may need to calculate distances to many training observations for each new query.

## 2. Sensitive to Feature Scaling

**Why?**

Distance depends directly on feature magnitudes.

## 3. Sensitive to Irrelevant Features

**Why?**

Irrelevant dimensions can distort the distance and cause the wrong observations to become neighbors.

## 4. Suffers in High Dimensions

**Why?**

Distances become less informative as the dimensionality grows.

## 5. Needs to Store Training Data

**Why?**

The model depends on the original observations during prediction.

## 6. Sensitive to $k$

**Why?**

A poor value of $k$ can lead to either excessive variance or excessive bias.

---

# 23. Failure Modes

## When Does It Perform Poorly?

KNN can perform poorly when:

- The number of features is very large.
- Many features are irrelevant.
- Features are badly scaled.
- Distance does not represent meaningful similarity.
- The dataset is extremely large and prediction latency matters.
- The data is highly sparse in a high-dimensional space.
- Local similarity does not imply target similarity.
- Classes are severely imbalanced without appropriate handling.

## Why Does It Fail?

### Bad Feature Geometry

KNN depends on geometry.

If the geometry is misleading, the neighbors are misleading.

### Curse of Dimensionality

As dimensions increase, observations become sparse and distance-based neighborhoods become less informative.

### Irrelevant Features

Suppose:

```text
5 useful features
+
95 irrelevant features
```

The 95 irrelevant dimensions can distort distances and hide the useful local structure.

### Poor Scaling

One large-scale feature can dominate the distance.

### Distribution Shift

If new data comes from a different population, its nearest training points may not be representative.

## Warning Signs

- Validation performance changes strongly as $k$ changes.
- Small changes in preprocessing produce large performance changes.
- High-dimensional data produces weak neighborhood separation.
- Training and validation performance differ substantially.
- Neighbors have very different target values.

## How Can We Detect the Problem?

Use:

- Cross-validation.
- Learning curves.
- Distance distribution analysis.
- Neighbor inspection.
- Feature scaling checks.
- Feature relevance analysis.
- Confusion matrix.
- Residual analysis for regression.
- Time-based validation for temporal data.

## Possible Solutions

Depending on the cause:

- Scale features.
- Remove irrelevant or redundant features.
- Tune $k$.
- Try distance weighting.
- Choose a more suitable metric.
- Reduce dimensionality.
- Collect more representative data.
- Compare with models that do not rely directly on local distance.

---

# 24. Algorithm-Specific Edge Cases

## Case 1 — $k=1$

Problem:

```text
k = 1
```

The prediction depends entirely on one training observation.

This can produce:

- Very low bias.
- High variance.
- Strong sensitivity to noise.

## Case 2 — $k \geq n$

If $k$ is equal to or greater than the number of available training observations, the neighborhood can become the entire training dataset or exceed the intended training population.

This removes the local nature of KNN.

In practice, choose:

$$
1 \leq k < n
$$

unless the implementation and use case explicitly justify another configuration.

## Case 3 — Tied Votes

Suppose:

```text
k = 4

Class A → 2 votes
Class B → 2 votes
```

There is no unique majority.

The final result depends on the implementation's tie-handling behavior.

Therefore:

> Avoid relying on an assumed tie-breaking rule unless the library documentation defines it.

## Case 4 — Duplicate Points

Two or more training samples may have identical feature values but different labels.

This creates ambiguity because the same input location corresponds to different targets.

Possible causes:

- Label noise.
- Conflicting records.
- Different hidden variables not represented in the features.

## Case 5 — Distance Equals Zero

A query point may be exactly identical to one or more training points.

This matters especially for distance weighting because an inverse-distance expression such as:

$$
w_i = \frac{1}{d_i}
$$

is undefined when:

$$
d_i = 0
$$

A practical implementation must define how zero-distance neighbors are handled.

## Case 6 — Highly Imbalanced Classes

A majority class can dominate neighborhood voting.

Possible approaches:

- Use appropriate metrics.
- Investigate class-aware weighting.
- Consider resampling.
- Inspect confusion matrix and recall.
- Compare against class-balanced models.

## Case 7 — Time-Series Data

Do not randomly split time-ordered data when the goal is future prediction.

Use a time-aware validation strategy.

Otherwise, future information can influence model selection.

## Case 8 — High-Dimensional Sparse Data

KNN can become weak when meaningful neighborhoods are difficult to identify.

Consider:

- Feature selection.
- Dimensionality reduction where appropriate.
- Specialized models for the data representation.

---

# 25. Practical Implementation — Scikit-Learn

## Classification Import

```python
from sklearn.neighbors import KNeighborsClassifier
```

## Regression Import

```python
from sklearn.neighbors import KNeighborsRegressor
```

## Create Classification Model

```python
model = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    metric="minkowski",
    p=2,
    n_jobs=-1
)
```

## Create Regression Model

```python
model = KNeighborsRegressor(
    n_neighbors=5,
    weights="uniform",
    metric="minkowski",
    p=2,
    n_jobs=-1
)
```

With Minkowski distance:

```text
p = 1 → Manhattan
p = 2 → Euclidean
```

## Train

```python
model.fit(X_train, y_train)
```

For KNN, this step mainly stores the training data and prepares any required neighbor-search state.

## Predict

```python
y_pred = model.predict(X_test)
```

## Classification Probabilities

```python
y_proba = model.predict_proba(X_test)
```

The classifier can produce class probabilities based on the neighborhood structure.

## Evaluate Classification

```python
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Evaluate Regression

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
```

## Recommended Pipeline With Scaling

Because KNN depends on distances, scaling should generally be fitted only on the training data.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    ))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

# 26. From-Scratch Implementation

> Implement the core algorithm without using the KNN model provided by Scikit-Learn.

## Classification

```python
import numpy as np


class SimpleKNNClassifier:
    def __init__(self, k=5):
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        if self.k > len(X):
            raise ValueError("k cannot exceed the number of training samples.")

        self.X_train = X
        self.y_train = y

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        predictions = []

        for x in X:
            distances = np.sqrt(
                np.sum((self.X_train - x) ** 2, axis=1)
            )

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]

            values, counts = np.unique(
                nearest_labels,
                return_counts=True
            )

            predictions.append(values[np.argmax(counts)])

        return np.array(predictions)
```

## Regression

```python
import numpy as np


class SimpleKNNRegressor:
    def __init__(self, k=5):
        if k <= 0:
            raise ValueError("k must be greater than 0.")

        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        if self.k > len(X):
            raise ValueError("k cannot exceed the number of training samples.")

        self.X_train = X
        self.y_train = y

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        predictions = []

        for x in X:
            distances = np.sqrt(
                np.sum((self.X_train - x) ** 2, axis=1)
            )

            nearest_indices = np.argsort(distances)[:self.k]
            nearest_targets = self.y_train[nearest_indices]

            predictions.append(np.mean(nearest_targets))

        return np.array(predictions)
```

## Code ↔ Mathematics

| Code Component | Mathematical / Algorithmic Concept |
|---|---|
| `(self.X_train - x) ** 2` | Squared coordinate differences |
| `np.sum(..., axis=1)` | Sum over feature dimensions |
| `np.sqrt(...)` | Euclidean distance |
| `np.argsort(distances)` | Ranking neighbors by distance |
| `[:self.k]` | Selecting the $k$ nearest points |
| `np.argmax(counts)` | Majority vote |
| `np.mean(nearest_targets)` | Regression neighborhood average |

## Important Implementation Details

### 1. Scaling

The from-scratch implementation assumes the input features are already prepared appropriately.

### 2. Complexity

The implementation uses brute-force distance calculation.

For each query, it computes distances to all training observations.

### 3. Storage

The model retains the training dataset.

### 4. Tie Handling

A production implementation should explicitly define tie behavior.

### 5. Zero Distance

Distance-weighted KNN needs special handling when a query point exactly matches a training point.

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
Feature Scaling
    ↓
Baseline KNN
    ↓
Choose Candidate k Values
    ↓
Cross-Validation
    ↓
Tune Distance Metric / Weighting
    ↓
Final Evaluation
    ↓
Neighbor / Error Analysis
```

## Algorithm-Specific Considerations

For KNN, pay special attention to:

- Feature scaling.
- Whether distance represents meaningful similarity.
- Number of features.
- Irrelevant and redundant variables.
- Choice of $k$.
- Choice of distance metric.
- Prediction latency.
- Dataset size.
- Class imbalance.

A KNN pipeline can fail even when the code is correct if the feature geometry is poor.

---

# 28. Comparison With Related Algorithms

## KNN vs Logistic Regression

| Aspect | KNN | Logistic Regression |
|---|---|---|
| Core Idea | Local neighbors | Global linear relationship in log-odds |
| Assumptions | Meaningful local distance | Linear relationship in log-odds |
| Bias | Depends strongly on $k$ | Can be high if relationship is non-linear |
| Variance | Can be high for small $k$ | Often lower than very local KNN |
| Scaling | Usually important | Often useful, especially with regularization |
| Interpretability | Moderate | High |
| Training Speed | Usually very fast | Requires parameter optimization |
| Prediction Speed | Can be expensive | Usually fast |
| Overfitting Control | Tune $k$ and feature space | Regularization |
| Strength | Captures local non-linear patterns | Simple global decision function |
| Weakness | High-dimensional distance problems | Limited non-linearity without feature engineering |
| Typical Use Case | Small or moderate data with meaningful local structure | Binary or multiclass classification with approximately linear structure |

## KNN vs Decision Tree

| Aspect | KNN | Decision Tree |
|---|---|---|
| Core Idea | Neighbor similarity | Recursive feature splits |
| Model | Instance-based | Tree-based |
| Scaling | Usually important | Usually unnecessary |
| Interpretability | Moderate | High |
| Prediction | Distance search | Tree traversal |
| High-Dimensional Behavior | Often difficult | Can still be difficult, but does not rely on distance |
| Main Complexity Control | $k$ | Depth and split constraints |

## KNN vs SVM

| Aspect | KNN | SVM |
|---|---|---|
| Core Idea | Local neighbors | Maximum-margin boundary |
| Model Type | Instance-based | Parametric / kernel-based depending on formulation |
| Training | Minimal optimization for basic KNN | Explicit optimization |
| Prediction | Can be expensive | Often efficient after training |
| Scaling | Important | Usually important |
| Non-Linearity | Naturally local | Via kernels or feature mapping |
| Main Hyperparameters | $k$, metric, weights | $C$, kernel, gamma for RBF |
| High-Dimensional Behavior | Can degrade from distance issues | Can work well depending on representation |

## KNN vs Random Forest

| Aspect | KNN | Random Forest |
|---|---|---|
| Core Idea | Nearby training observations | Ensemble of randomized trees |
| Model Type | Instance-based | Tree ensemble |
| Scaling | Usually important | Usually unnecessary |
| Training Cost | Low | Higher |
| Prediction Cost | Can be high | Depends on number/depth of trees |
| Non-Linearity | Yes | Yes |
| High-Dimensional Behavior | Often sensitive | Can handle many features but irrelevant variables can still hurt |
| Interpretability | Moderate | Lower than one tree |
| Main Hyperparameters | $k$, metric, weights | `n_estimators`, depth, `max_features`, etc. |
| Main Strength | Simple local prediction | Strong general-purpose tabular baseline |

## Key Distinction

> **KNN learns the prediction rule from local neighborhoods at prediction time, while models such as Logistic Regression, Decision Trees, and Random Forest build an explicit model during training.**

---

# 29. When Should I Use This Algorithm?

Use KNN when:

- The dataset is small or moderate in size and prediction cost is acceptable.
- Local similarity is meaningful.
- The decision boundary may be strongly non-linear.
- You want a simple baseline without imposing a global functional form.
- Features can be represented in a meaningful metric space.
- You have enough data density around the query regions.

A particularly useful question is:

> **"Does being close in feature space actually imply being similar in target?"**

If the answer is yes, KNN becomes a reasonable candidate.

---

# 30. When Should I Avoid This Algorithm?

Consider another approach when:

- The dataset is extremely large and prediction latency is important.
- The number of dimensions is very high.
- Distance is not semantically meaningful.
- Many features are irrelevant or noisy.
- The data is extremely sparse.
- The problem requires strong interpretability through learned global coefficients.
- The target depends on long-range global structure rather than local similarity.

The issue is not that KNN is "bad" for these datasets.

The issue is that its central mechanism—distance-based locality—may not fit the problem.

---

# 31. Algorithm Selection Guide

When facing a new ML problem:

```text
What type of problem?
        ↓
Classification / Regression
        ↓
Is local similarity meaningful?
        ↓
Yes ---------------- No
 ↓                    ↓
Is the feature space   Consider models that
low/moderate-dimensional  learn a stronger
and well-scaled?           global structure
 ↓
Yes
 ↓
Is the dataset small/moderate
or prediction cost acceptable?
 ↓
Yes
 ↓
Try KNN as a baseline
 ↓
Tune k + metric + weighting
 ↓
Compare using cross-validation
```

### Practical reasoning

If:

```text
Local similarity matters
+
Distance is meaningful
+
Data is not excessively high-dimensional
```

KNN is worth testing.

If:

```text
Distance is meaningless
OR
Dimensionality is extremely high
OR
Prediction latency is critical
```

another model family may be more appropriate.

---

# 32. Common Misconceptions

## Misconception 1

> "KNN has no training."

**Correction:**

KNN performs little conventional parameter learning during `fit()`, but it still has a training/fitting phase in which the training data is stored and a search structure may be prepared.

The important distinction is that **most predictive computation is deferred until prediction time**.

## Misconception 2

> "Larger $k$ is always better because it uses more data."

**Correction:**

A large $k$ can reduce variance, but it can also increase bias and oversmooth important local patterns.

## Misconception 3

> "KNN does not require feature scaling."

**Correction:**

Scaling is usually important because KNN depends directly on distances.

## Misconception 4

> "KNN cannot model non-linear relationships."

**Correction:**

KNN can represent highly non-linear relationships because its prediction depends on local neighborhoods.

## Misconception 5

> "KNN is always slow."

**Correction:**

Training is usually cheap, but prediction can be expensive. Efficient neighbor-search structures can reduce query cost in suitable low-dimensional settings.

## Misconception 6

> "KNN is automatically interpretable."

**Correction:**

The neighborhood of a prediction can be inspected, but the global behavior of KNN can become difficult to summarize, especially with many features and observations.

---

# 33. Common Implementation Mistakes

## Mistake 1 — Forgetting Feature Scaling

Using raw features with very different scales can distort distances.

Use a preprocessing pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(...))
])
```

## Mistake 2 — Tuning `k` on the Test Set

Do not repeatedly choose $k$ based on test performance.

Correct workflow:

```text
Training data
    ↓
Cross-validation
    ↓
Select k
    ↓
Final test evaluation
```

## Mistake 3 — Data Leakage During Scaling

Wrong:

```text
Scale entire dataset
    ↓
Train / test split
```

Better:

```text
Train / test split
    ↓
Fit scaler on training data
    ↓
Transform training data
    ↓
Transform test data
```

A Scikit-Learn `Pipeline` is the cleanest way to enforce this workflow.

## Mistake 4 — Ignoring Irrelevant Features

Every irrelevant feature can influence distance.

Feature selection can therefore matter greatly.

## Mistake 5 — Choosing $k$ Arbitrarily

Do not assume:

```text
k = 5
```

is universally correct.

Use validation or cross-validation.

## Mistake 6 — Ignoring the Distance Metric

Euclidean distance is not automatically correct for every dataset.

The metric should reflect the problem and feature representation.

## Mistake 7 — Using Accuracy Only for Imbalanced Classification

Use metrics appropriate to the error costs and class distribution.

## Mistake 8 — Random Splitting Time-Series Data

Time-ordered data requires time-aware validation when predicting future observations.

---

# 34. Interview Questions

## Basic

### Q1. What is KNN?

**Answer:**

> K-Nearest Neighbors is a non-parametric, instance-based supervised learning algorithm that predicts a new observation using the target values of its $k$ nearest training observations according to a selected distance metric.

### Q2. How does KNN work?

**Answer:**

> KNN calculates the distance between a new observation and the training observations, selects the $k$ closest ones, and aggregates their target values. For classification it usually uses majority voting, while for regression it usually uses averaging.

### Q3. Is KNN supervised or unsupervised?

**Answer:**

> KNN is a supervised learning algorithm because it requires labeled training data.

---

## Intermediate

### Q4. Why is KNN called a lazy learner?

**Answer:**

> Because it performs relatively little computation during training and delays most of the predictive computation until a new sample must be classified or regressed.

### Q5. Why is KNN called non-parametric?

**Answer:**

> KNN does not assume a fixed finite-dimensional parametric form such as a linear equation. Its predictions depend directly on the local arrangement of training observations.

### Q6. Why is feature scaling important in KNN?

**Answer:**

> KNN uses distance calculations. If features have very different scales, a large-scale feature can dominate the distance and distort which observations are considered nearest.

### Q7. What happens when $k$ increases?

**Answer:**

> Increasing $k$ generally smooths the model, reduces variance, and increases bias. Decreasing $k$ makes the model more local, usually reducing bias but increasing variance.

---

## Advanced

### Q8. What is the curse of dimensionality?

**Answer:**

> The curse of dimensionality refers to the difficulties that arise as the number of dimensions increases. In KNN, points become increasingly sparse and distance differences can become less informative, making nearest-neighbor relationships less useful.

### Q9. Why can KNN perform poorly with irrelevant features?

**Answer:**

> Irrelevant features contribute to the distance calculation even though they contain little predictive information. They can therefore distort the geometry of the feature space and cause the algorithm to select poor neighbors.

### Q10. How would you choose the value of $k$?

**Answer:**

> I would evaluate candidate values of $k$ using cross-validation on the training data. I would then choose a value that provides good validation performance while considering prediction cost and stability.

### Q11. What distance metrics can KNN use?

**Answer:**

> Common choices include Euclidean distance, Manhattan distance, and the broader Minkowski family. The appropriate metric depends on the structure of the data.

### Q12. Why does KNN have low training cost but potentially high prediction cost?

**Answer:**

> KNN does not need to learn a large parameterized model during fitting. Instead, when a new observation arrives, it must search the training data for nearby observations, which can be computationally expensive.

---

# 35. Interview Follow-Up Drill

The first answer is rarely the end of the interview.

### Interviewer: "Why does feature scaling matter so much?"

**Your answer:**

Because the prediction depends on distance. A feature measured on a much larger scale contributes much more to the distance than a small-scale feature, even if the small-scale feature is more informative.

### Interviewer: "What happens if we decrease k?"

**Your answer:**

The model becomes more local and flexible. Bias generally decreases, but variance increases because predictions become more sensitive to individual training observations and noise.

### Interviewer: "Why not always use k=1?"

**Your answer:**

Because $k=1$ can overfit noise. The prediction depends on a single training observation, making the model highly sensitive to small changes in the training set.

### Interviewer: "Why not always use a very large k?"

**Your answer:**

A very large $k$ can oversmooth the model. It may combine observations from different local regions and increase bias.

### Interviewer: "Why not use Euclidean distance every time?"

**Your answer:**

Euclidean distance assumes that straight-line geometric distance is an appropriate measure of similarity. That may not be true for every feature space, so the metric should match the data representation.

### Interviewer: "What happens with 1,000 features?"

**Your answer:**

KNN may suffer from the curse of dimensionality. The data becomes sparse and nearest-neighbor distances can become less informative. I would examine feature selection, dimensionality reduction, and alternative model families.

### Interviewer: "Why is KNN called non-parametric?"

**Your answer:**

Because it does not impose a fixed parametric form with a predetermined number of coefficients. The effective complexity can grow with the training data and depends on the local neighborhood structure.

---

# 36. Explain This Algorithm in an Interview

## 30-Second Explanation

> KNN is a supervised, non-parametric, instance-based algorithm used for classification and regression. For a new data point, it calculates distances to the training points, finds the $k$ nearest neighbors, and uses them to make the prediction. For classification it typically uses majority voting, while for regression it usually takes an average. The most important factors are the choice of $k$, the distance metric, and feature scaling.

## 1-Minute Explanation

> KNN is a lazy learning algorithm because it performs very little parameter learning during training. Instead, it stores the training data and performs the main computation during prediction. For a new observation, it calculates distances to training observations, selects the $k$ closest points, and aggregates their target values. Small values of $k$ make the model flexible but high-variance, while large values make it smoother but higher-bias. Because KNN is distance-based, feature scaling is usually important. Its main limitations are prediction cost, sensitivity to irrelevant features, and the curse of dimensionality.

## 3-Minute Explanation

> KNN is a supervised, non-parametric, instance-based learning algorithm. Its main assumption is locality: observations that are close in feature space tend to have similar target values.
>
> During fitting, KNN mainly stores the training observations. For a new query point, it computes distances between the query and the training points using a metric such as Euclidean or Manhattan distance. It then selects the $k$ nearest observations.
>
> For classification, the standard approach is majority voting:
>
> ```math
> \hat{y}
> =
> \underset{c}{\mathrm{arg\,max}}
> \sum_{i\in N_k(x)}
> \mathbf{1}\{y_i=c\}
> ```
>
> For regression, the standard prediction is the mean:
>
> ```math
> \hat{y}
> =
> \frac{1}{k}
> \sum_{i\in N_k(x)}y_i
> ```
>
> The most important hyperparameter is $k$. A small $k$ produces local, flexible predictions with lower bias and higher variance. A large $k$ averages over a wider neighborhood, increasing bias and reducing variance.
>
> Feature scaling is especially important because distance depends on numerical magnitude. KNN can also struggle in high-dimensional spaces because of the curse of dimensionality.
>
> Its main advantage is that it can model complex non-linear local patterns without assuming a global functional form. Its main limitation is that prediction can become expensive and distance becomes less useful as dimensionality increases.

---

# 37. Key Takeaways

## Core Idea

> **Find the $k$ closest training observations and use their outcomes to predict the new observation.**

## Mathematical Idea

> **Define similarity using a distance metric, select the $k$ smallest distances, and aggregate the corresponding targets.**

## Training Idea

> **Store the training observations; most predictive computation is delayed until query time.**

## Prediction Idea

> **Calculate distances → select $k$ nearest neighbors → vote or average.**

## Main Strength

> **KNN can capture non-linear local relationships without assuming a global functional form.**

## Main Limitation

> **KNN can be computationally expensive at prediction time and becomes less effective when distance is not informative, especially in high-dimensional spaces.**

## Most Important Hyperparameters

> **`n_neighbors`, `weights`, `metric`, `p`, and `algorithm`.**

## Most Important Assumption

> **Nearby observations in the chosen feature space should tend to have similar target values.**

## Most Important Interview Concept

> **The value of $k$ controls the bias-variance trade-off: small $k$ gives low bias and high variance, while large $k$ gives higher bias and lower variance.**

---

# 38. Completion Checklist

Before marking this algorithm as complete, I should be able to answer:

- [ ] What problem does KNN solve?
- [ ] Why do we need KNN?
- [ ] What is the formal definition?
- [ ] What is the core intuition?
- [ ] Why is KNN called a lazy learner?
- [ ] Why is KNN called non-parametric?
- [ ] Why is KNN instance-based?
- [ ] How is the problem mathematically formulated?
- [ ] How is distance calculated?
- [ ] What is Euclidean distance?
- [ ] What is Manhattan distance?
- [ ] What is Minkowski distance?
- [ ] How does KNN perform classification?
- [ ] How does KNN perform regression?
- [ ] What is majority voting?
- [ ] What is distance-weighted voting?
- [ ] What happens when $k$ changes?
- [ ] How does $k$ affect bias and variance?
- [ ] What assumptions does KNN rely on?
- [ ] Why does feature scaling matter?
- [ ] What happens with irrelevant features?
- [ ] What is the curse of dimensionality?
- [ ] How does KNN behave with outliers?
- [ ] What happens with class imbalance?
- [ ] How are ties handled?
- [ ] What are the key hyperparameters?
- [ ] What happens when `n_neighbors` increases?
- [ ] What are the main distance metrics?
- [ ] What are the computational costs?
- [ ] Why is training cheap but prediction expensive?
- [ ] What are the strengths and weaknesses?
- [ ] When should I use KNN?
- [ ] When should I avoid KNN?
- [ ] How does KNN compare with Logistic Regression?
- [ ] How does KNN compare with Decision Trees?
- [ ] How does KNN compare with SVM?
- [ ] How does KNN compare with Random Forest?
- [ ] Can I implement KNN using Scikit-Learn?
- [ ] Can I implement basic KNN from scratch?
- [ ] Can I explain the implementation line-by-line?
- [ ] Can I answer "why?" follow-ups?
- [ ] Can I explain KNN in 30 seconds, 1 minute, and 3 minutes?

---

# 39. References

- [GitHub Docs — Writing Mathematical Expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
- [GitHub Docs — Basic Writing and Formatting Syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Scikit-Learn — KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Scikit-Learn — KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
- [Scikit-Learn — Nearest Neighbors User Guide](https://scikit-learn.org/stable/modules/neighbors.html)
- Cover, T. & Hart, P. (1967), *Nearest Neighbor Pattern Classification*
