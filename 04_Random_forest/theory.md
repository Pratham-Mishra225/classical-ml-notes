# Random Forest

> Complete theory, intuition, mathematics, practical understanding, implementation, and interview preparation.

---

## 0. Learning Objectives

By the end of this chapter, I should be able to:

- Explain Random Forest intuitively.
- Give a formal, interview-ready definition.
- Explain why a single Decision Tree can overfit.
- Explain how bagging and random feature selection make Random Forest different from a single tree.
- Explain the mathematical foundation of bootstrap sampling and ensemble prediction.
- Explain classification and regression in Random Forest.
- Explain the role of Gini impurity, entropy, and variance/MSE.
- Explain the complete training and prediction process.
- Explain Out-of-Bag (OOB) samples and OOB error.
- Understand bias, variance, correlation, and why averaging helps.
- Tune the important hyperparameters.
- Explain why Random Forest usually does not need feature scaling.
- Explain how Random Forest behaves with outliers, correlated features, and high-dimensional data.
- Compare Random Forest with Decision Trees, Bagging, Extra Trees, Gradient Boosting, and Logistic Regression.
- Implement Random Forest using Scikit-Learn.
- Understand a basic from-scratch implementation.
- Answer placement/interview follow-up questions.

---

# 1. Prerequisites

## Concepts I Should Know First

- Decision Trees
- Gini impurity
- Entropy and information gain
- Regression tree splitting using variance/MSE reduction
- Overfitting and underfitting
- Bias-variance trade-off
- Bootstrap sampling
- Ensemble learning
- Classification and regression evaluation metrics

## Connection With Previous Algorithms

Random Forest is built directly on top of Decision Trees.

A Decision Tree is a high-variance model: if the training data changes slightly, the learned tree can change significantly.

Random Forest keeps the basic Decision Tree idea, but trains many different trees and combines their predictions.

The two main sources of randomness are:

1. **Random samples:** each tree is trained on a bootstrap sample of the training data.
2. **Random features:** at each split, the tree considers only a random subset of features.

The final prediction combines the predictions of all trees.

---

# 2. Why Do We Need This Algorithm?

## 2.1 The Problem

A single Decision Tree can learn complex non-linear relationships and interactions between features.

However, if a tree is allowed to grow too deeply, it can fit noise and peculiarities of the training set.

This gives the model:

- Low training error
- High variance
- Potentially poor test performance

The central question is:

> **Can we keep the flexibility of Decision Trees while making predictions more stable and less sensitive to the training data?**

Random Forest is one answer to this problem.

## 2.2 Limitations of Previous Approaches

### Single Decision Tree

A single deep tree can:

- Overfit the training data.
- Have high variance.
- Be sensitive to small changes in the dataset.
- Become unstable when strong features dominate many splits.

### Bagging Alone

Bagging trains multiple trees on bootstrap samples and averages/votes their predictions.

This reduces variance.

However, if all trees repeatedly select the same very strong feature near the top of the tree, the trees can become highly correlated.

Highly correlated trees provide less benefit from averaging.

### Need for Random Feature Selection

Random Forest adds a second source of randomness:

> At each split, a tree is allowed to search only a random subset of the available features.

This reduces similarity between trees.

---

## 2.3 What Should a Better Approach Do?

A useful ensemble of trees should:

- Keep the low-bias behavior of flexible trees.
- Reduce the high variance of individual trees.
- Make individual trees different from one another.
- Combine many weakly correlated errors.
- Work for both classification and regression.
- Require relatively little preprocessing.
- Handle non-linear relationships and feature interactions.

---

## 2.4 Core Idea

> **Random Forest trains many Decision Trees on different bootstrap samples of the data and uses a random subset of features at each split. It then combines the predictions of all trees using voting for classification and averaging for regression.**

The key idea is:

```text
Different trees
      ↓
Different errors
      ↓
Combine predictions
      ↓
More stable prediction
      ↓
Lower variance and better generalization
```

---

# 3. Formal Definition

## Definition

> **Random Forest is a supervised ensemble learning algorithm that constructs multiple randomized Decision Trees using bootstrap samples of the training data and random subsets of features at each split, then combines the tree predictions by majority voting for classification or averaging for regression.**

### Interview-ready one-line definition

> **Random Forest is an ensemble of Decision Trees trained with bootstrap sampling and random feature selection, whose predictions are aggregated to reduce variance and improve generalization.**

## Algorithm Classification

| Property | Description |
|---|---|
| Learning Type | Supervised |
| Task | Classification and Regression |
| Parametricity | Non-parametric |
| Model Type | Tree-based ensemble |
| Learning Approach | Bagging + random feature selection |
| Base Learner | Decision Tree |
| Main Goal | Reduce variance and improve generalization |
| Prediction | Voting / averaging |
| Feature Scaling | Usually not required |

### Important interview point

Random Forest is **not a linear model** and does not assume a linear relationship between the features and target.

---

# 4. Intuition

## 4.1 Core Intuition

Imagine asking one person to make an important decision.

That person may make a mistake because of limited information.

Now ask many different people.

Each person sees a slightly different version of the information and makes a prediction.

Finally:

- For classification, take the majority vote.
- For regression, take the average.

The group decision is usually more stable than one individual's decision.

Random Forest applies the same idea to Decision Trees.

Each tree is deliberately made different using randomness.

---

## 4.2 Real-World Analogy

Suppose you want to decide whether a student should be selected for an internship.

Instead of asking one interviewer, imagine 100 interviewers.

Each interviewer receives:

- A different sample of historical candidate data.
- A different subset of features at each decision point.

Each interviewer produces a prediction.

The final result is the combined decision.

The important part is not simply having many interviewers.

The interviewers should also make **different errors**.

That is why Random Forest needs randomness.

---

## 4.3 Simple Example

Suppose 5 trees predict whether a customer will churn.

```text
Tree 1 → Churn
Tree 2 → Not Churn
Tree 3 → Churn
Tree 4 → Churn
Tree 5 → Not Churn
```

Votes:

```text
Churn     = 3
Not Churn = 2
```

Final prediction:

```text
Churn
```

For regression:

```text
Tree 1 → 80
Tree 2 → 90
Tree 3 → 85
Tree 4 → 95
Tree 5 → 100
```

Final prediction:

$$
\hat{y} = \frac{80+90+85+95+100}{5}=90
$$

---

## 4.4 Mental Model

> **Many different trees + uncorrelated errors + aggregation = a more stable model.**

---

# 5. Problem Formulation

## Given

Training dataset:

$$
D = \{(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)\}
$$

where:

- $x_i \in \mathbb{R}^d$ = feature vector for sample $i$
- $y_i$ = target value for sample $i$
- $n$ = number of training samples
- $d$ = number of features

We construct $B$ trees:

$$
T_1,T_2,\ldots,T_B
$$

Each tree is trained using randomization.

## Goal

Learn an ensemble function whose prediction generalizes well to unseen data.

For classification:

$$
\hat{y} = \operatorname{mode}\{T_1(x),T_2(x),\ldots,T_B(x)\}
$$

For regression:

$$
\hat{y} = \frac{1}{B}\sum_{b=1}^{B}T_b(x)
$$

## Input

The model receives:

- Training features $X$
- Training targets $y$

During prediction, it receives an unseen feature vector $x$.

## Output

For classification:

- Predicted class
- Optional class probabilities

For regression:

- Predicted continuous value

---

# 6. Mathematical Foundation

## 6.1 Model Representation

A Random Forest is represented as an ensemble of decision trees:

$$
\mathcal{F}=\{T_1,T_2,\ldots,T_B\}
$$

where $B$ is the number of trees.

Each tree is produced using two kinds of randomization:

1. Bootstrap sampling of training observations.
2. Random feature selection at each split.

The final model is therefore a function of many randomized trees.

---

## 6.2 Objective

Random Forest does **not** optimize one single smooth global loss function using gradient descent.

Instead:

1. Each Decision Tree is built greedily by selecting useful splits.
2. Each tree uses a randomized training sample and feature subset.
3. Tree predictions are aggregated.

The tree-level split objective depends on the task.

### Classification

Typical split criteria:

- Gini impurity
- Entropy

### Regression

Typical split criterion:

- Mean squared error / variance reduction

At the forest level, the main statistical objective is to reduce generalization error by averaging diverse trees.

---

## 6.3 Loss / Cost / Objective Function

### Classification: Gini Impurity

For a node $t$:

$$
G(t)=1-\sum_{k=1}^{K}p_k^2
$$

where:

- $K$ = number of classes
- $p_k$ = proportion of samples in node $t$ belonging to class $k$

A pure node has:

$$
G(t)=0
$$

### Classification: Entropy

$$
H(t)=-\sum_{k=1}^{K}p_k\log_2(p_k)
$$

Entropy is also:

$$
H(t)=-\sum_{k=1}^{K}p_k\log(p_k)
$$

The base of the logarithm changes only the scale.

### Weighted Impurity After a Split

Suppose a parent node $P$ is split into left child $L$ and right child $R$.

The weighted impurity is:

$$
I_{\text{split}} = \frac{n_L}{n_P}I(L) + \frac{n_R}{n_P}I(R)
$$

where:

- $n_P$ = number of samples in parent
- $n_L$ = number of samples in left child
- $n_R$ = number of samples in right child
- $I(\cdot)$ = impurity measure

The algorithm prefers a split that gives lower weighted child impurity.

### Information Gain

For entropy:

$$
IG = H(P) - \left[ \frac{n_L}{n_P}H(L) + \frac{n_R}{n_P}H(R) \right]
$$

Higher information gain is better.

### Gini Gain

Similarly:

$$
\Delta G = G(P) - \left[ \frac{n_L}{n_P}G(L) + \frac{n_R}{n_P}G(R) \right]
$$

Higher impurity reduction is better.

### Regression: Mean Squared Error

For node $t$ containing target values $y_i$:

$$
MSE(t) = \frac{1}{n_t} \sum_{i\in t}(y_i-\bar{y}_t)^2
$$

where:

$$
\bar{y}_t = \frac{1}{n_t}\sum_{i\in t}y_i
$$

The tree chooses splits that reduce the weighted MSE.

---

## 6.4 Why This Objective Function?

The purpose of a split is to create child nodes that are more homogeneous.

For classification:

- A pure node contains mostly one class.
- Lower impurity means the class distribution is more concentrated.

For regression:

- A good node contains target values that are close to one another.
- Lower MSE means less variation inside the node.

The forest then combines many such trees to improve stability.

---

## 6.5 Optimization

Random Forest does **not** use gradient descent.

Instead, each tree is constructed using a greedy recursive partitioning process.

At each node:

1. Randomly select a subset of features.
2. Search for a good split among those features.
3. Choose the split that gives the largest impurity reduction.
4. Repeat recursively.

The tree-building procedure is greedy because the best split is chosen locally at each node.

The complete forest is then obtained by repeating this process many times with different randomness.

---

## 6.6 Bootstrap Sampling

Suppose the training set contains $n$ observations.

For each tree, draw $n$ observations **with replacement**.

This creates a bootstrap dataset:

$$
D_b^*
$$

Some original observations can appear multiple times.

Some observations may not appear at all.

### Probability That One Observation Is Not Selected

For one draw:

$$
P(\text{not selected})=1-\frac{1}{n}
$$

After $n$ draws:

$$
P(\text{not selected}) = \left(1-\frac{1}{n}\right)^n
$$

As $n\to\infty$:

$$
\left(1-\frac{1}{n}\right)^n \to e^{-1} \approx 0.368
$$

So about **36.8%** of the observations are left out of a bootstrap sample on average.

These observations are called **Out-of-Bag (OOB) samples** for that tree.

The remaining roughly 63.2% are the distinct observations expected to appear at least once.

> Important: 63.2% is the expected fraction of **unique** observations in a bootstrap sample, not the fraction of draws.

---

## 6.7 Random Feature Selection

Suppose there are $d$ total features.

At a tree node, Random Forest selects only $m_{\text{try}}$ features:

$$
m_{\text{try}} < d
$$

The split is optimized only over those selected features.

This forces different trees to consider different features.

It reduces correlation between trees.

---

## 6.8 Classification Prediction

Let tree $b$ predict class:

$$
T_b(x) \in \{1,\ldots,K\}
$$

The forest predicts:

$$
\hat{y}
=
\mathrm{mode}
\left\{
T_1(x), T_2(x), \ldots, T_B(x)
\right\}
$$

Equivalently:

$$
\hat{y}
=
\underset{k}{\mathrm{arg\,max}}
\sum_{b=1}^{B}
\mathbf{1}\{T_b(x)=k\}
$$

where $\mathbf{1}[\cdot]$ is 1 when the condition is true and 0 otherwise.

---

## 6.9 Regression Prediction

For regression:

$$
\hat{y} = \frac{1}{B} \sum_{b=1}^{B}T_b(x)
$$

The forest simply averages the predictions of the trees.

---

## 6.10 Why Does Averaging Reduce Variance?

This is one of the most important Random Forest interview concepts.

Assume each tree has variance $\sigma^2$ and the pairwise correlation between trees is $\rho$.

For a simplified ensemble of $B$ trees:

$$
Var(\bar{T}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

or equivalently:

$$
Var(\bar{T}) = \sigma^2 \left[ \rho+\frac{1-\rho}{B} \right]
$$

As $B$ becomes large:

$$
\frac{1-\rho}{B}\to0
$$

so:

$$
Var(\bar{T})\to\rho\sigma^2
$$

### Key conclusion

- More trees reduce the independent part of variance.
- Lower correlation between trees makes the ensemble more effective.
- Therefore Random Forest needs both:
  - **Strong trees**
  - **Low correlation between trees**

This explains why random feature selection is important.

---

## 6.11 Bias-Variance Interpretation

A deep Decision Tree usually has:

- Low bias
- High variance

A Random Forest keeps flexible trees but reduces variance through averaging.

Therefore, compared with one deep tree:

> **Random Forest usually keeps relatively low bias while substantially reducing variance.**

Random feature selection may slightly increase the bias of individual trees, but it can reduce correlation enough to improve the overall forest.

---

## 6.12 Important Mathematical Properties

- Convex / Non-convex: not naturally described as one global convex optimization problem.
- Differentiable / Non-differentiable: split criteria are based on discrete partitioning; gradient-based optimization is not used.
- Closed-form / Iterative: trees are built recursively using greedy split selection.
- Local vs Global Optimum: each split is optimized locally; the tree-growing process is greedy rather than a global optimization of all possible trees.
- Main statistical mechanism: variance reduction through aggregation and decorrelation.
- Ensemble limit: as the number of trees increases, the empirical forest prediction becomes more stable.

---

# 7. Geometric / Visual Understanding

## 7.1 What Does the Data Look Like?

Decision Trees divide feature space into axis-aligned regions.

For example:

```text
Feature 2
   ↑
   |
   |      Class B
   |   +----------+
   |   |          |
   |   |          |
   |---+----------+----→ Feature 1
   |   |
   |   | Class A
   |   |
```

A single tree creates one hierarchical partition of the feature space.

A Random Forest creates many different partitions because every tree sees different data and different feature subsets.

## 7.2 What Does the Model Learn?

A Random Forest learns:

- Multiple tree structures.
- Split features.
- Split thresholds.
- Leaf predictions.
- The collection of all trees.

There is no single line or simple global equation like:

$$
y=w_1x_1+w_2x_2+b
$$

Instead, the model learns a collection of piecewise decision rules.

## 7.3 Effect of Model Complexity

### Fewer / Shallower Trees

- Lower representation power.
- Possible underfitting.

### Deep Trees

- Each tree can fit complex interactions.
- Individual trees may have high variance.

### More Trees

Increasing the number of trees usually:

- Improves stability.
- Reduces the Monte Carlo noise of the ensemble.
- Reduces variance up to a point.
- Increases training and prediction cost.

A very large number of trees usually does not cause the same type of overfitting seen by simply making one tree deeper.

---

### Diagram

```text
                 Training Data
                       |
          +------------+------------+
          |            |            |
          ↓            ↓            ↓
   Bootstrap 1   Bootstrap 2  Bootstrap 3   ... Bootstrap B
          |            |            |
          ↓            ↓            ↓
      Tree 1        Tree 2       Tree 3      ... Tree B
          |            |            |
          +------------+------------+
                       |
                    Aggregate
                       |
             Classification → Vote
             Regression     → Average
                       |
                    Prediction
```

At each split:

```text
All Features
     |
     ↓
Random Feature Subset
     |
     ↓
Best Split Among Selected Features
```

---

# 8. How the Algorithm Works

## Step-by-Step

### Step 1 — Start With the Training Dataset

Suppose the original dataset is:

$$
D=\{(x_i,y_i)\}_{i=1}^{n}
$$

The forest will contain $B$ trees.

---

### Step 2 — Create a Bootstrap Sample for Each Tree

For tree $b$:

- Sample $n$ observations from the original dataset.
- Sample with replacement.
- Create bootstrap dataset $D_b^*$.

Therefore:

```text
Original Data
     |
     +---- Bootstrap sample for Tree 1
     +---- Bootstrap sample for Tree 2
     +---- Bootstrap sample for Tree 3
     +---- ...
     +---- Bootstrap sample for Tree B
```

Different trees receive different training samples.

---

### Step 3 — Grow a Decision Tree

For each node in the tree:

1. Randomly select $m_{\text{try}}$ features.
2. Evaluate candidate splits using only those features.
3. Select the best split according to the chosen criterion.
4. Split the node.
5. Repeat recursively.

---

### Step 4 — Repeat for Many Trees

Repeat Steps 2 and 3 for:

$$
b=1,2,\ldots,B
$$

This creates:

$$
T_1,T_2,\ldots,T_B
$$

---

### Step 5 — Aggregate Predictions

For classification:

```text
Tree 1 → Class A
Tree 2 → Class B
Tree 3 → Class A
...
Tree B → Class A

Final → Majority vote
```

For regression:

```text
Tree 1 → 10
Tree 2 → 12
Tree 3 → 9
...
Tree B → 11

Final → Mean
```

---

## Algorithm Flow

```text
Input Data
    ↓
Choose number of trees B
    ↓
For each tree:
    ↓
Draw bootstrap sample
    ↓
Grow decision tree
    ↓
At every split:
Randomly select feature subset
    ↓
Choose best split
    ↓
Continue until stopping condition
    ↓
Store tree
    ↓
Repeat for B trees
    ↓
Aggregate tree predictions
    ↓
Final Prediction
```

## Pseudocode

```text
Input:
    Training data D
    Number of trees B
    Number of candidate features m_try

For b = 1 to B:

    1. Draw a bootstrap sample D_b from D.

    2. Start growing tree T_b.

    3. At each node:
        a. Randomly choose m_try features.
        b. Find the best split using only those features.
        c. Split the node.
        d. Continue recursively.

    4. Store T_b.

For a new sample x:

    Classification:
        Get prediction from every tree.
        Return majority vote.

    Regression:
        Get prediction from every tree.
        Return average prediction.
```

---

# 9. Training vs Prediction

## 9.1 Training Phase

When calling:

```python
model.fit(X_train, y_train)
```

the model:

1. Creates bootstrap samples.
2. Grows many Decision Trees.
3. Randomly selects features at each node.
4. Finds tree splits using the selected criterion.
5. Stores the resulting trees.

Conceptually:

```text
X_train + y_train
        ↓
Bootstrap sampling
        ↓
Random feature selection
        ↓
Build many Decision Trees
        ↓
Random Forest
```

## 9.2 Prediction Phase

When calling:

```python
y_pred = model.predict(X_test)
```

the model:

1. Sends each test sample through every tree.
2. Collects all tree predictions.
3. Aggregates them.

Classification:

```text
Predictions from trees
        ↓
Majority vote
        ↓
Class
```

Regression:

```text
Predictions from trees
        ↓
Average
        ↓
Continuous value
```

## 9.3 What Does the Model Actually Learn?

A trained Random Forest stores:

- The number of trees.
- The structure of every tree.
- Feature chosen at each split.
- Threshold chosen at each split.
- Child-node relationships.
- Leaf values or class distributions.
- Additional training information required for prediction.
- OOB-related information when enabled.

It does **not** primarily learn one set of global coefficients such as $w_1,w_2,\ldots,w_d$.

## 9.4 What Is Stored After Training?

Conceptually:

```text
Random Forest
│
├── Tree 1
│   ├── root split
│   ├── internal nodes
│   └── leaves
│
├── Tree 2
│   ├── root split
│   ├── internal nodes
│   └── leaves
│
└── ...
    └── Tree B
```

In Scikit-Learn, fitted trees can be accessed through the estimator's tree collection attributes.

---

# 10. Worked Example

## Dataset

Consider a customer churn problem.

| Age | MonthlyCharges | ContractMonths | Churn |
|---:|---:|---:|---|
| 22 | 80 | 2 | Yes |
| 25 | 75 | 3 | Yes |
| 35 | 60 | 24 | No |
| 42 | 55 | 36 | No |
| 29 | 90 | 2 | Yes |
| 50 | 45 | 48 | No |

Suppose:

- Number of trees = 3
- Each bootstrap sample contains 6 draws
- At each split, only a random subset of features is considered

## Step 1

Create a bootstrap sample for Tree 1.

Example:

```text
Rows: 1, 2, 2, 4, 5, 6
```

Row 2 appears twice.

Row 3 is not selected.

Tree 1 is trained only on this bootstrap sample.

---

## Step 2

Create a different bootstrap sample for Tree 2.

Example:

```text
Rows: 1, 3, 3, 4, 5, 5
```

Now Tree 2 sees a different distribution.

---

## Step 3

Create another bootstrap sample for Tree 3.

```text
Rows: 2, 2, 3, 4, 6, 6
```

At the root of each tree, suppose only two random features are considered.

Example:

```text
Tree 1 → Age, MonthlyCharges
Tree 2 → MonthlyCharges, ContractMonths
Tree 3 → Age, ContractMonths
```

This makes the tree structures more diverse.

---

## Step 4

Suppose a new customer receives these predictions:

```text
Tree 1 → Yes
Tree 2 → No
Tree 3 → Yes
```

Final prediction:

```text
Yes
```

because Yes receives 2 of 3 votes.

---

## Step 5 — OOB Idea

For Tree 1, row 3 was not part of its bootstrap sample.

Therefore row 3 is OOB for Tree 1.

If row 3 is OOB for several trees, those trees can predict row 3 even though those trees did not train on row 3.

This makes it possible to estimate performance without using that observation in the training set of those trees.

---

## Final Result

Random Forest combines many unstable individual trees into a more stable ensemble by controlling two things:

```text
Bootstrap sampling
        +
Random feature selection
        ↓
Different trees
        ↓
Lower correlation
        ↓
Aggregation
        ↓
Lower variance
```

> Goal: I should be able to mentally execute the core Random Forest process on a tiny dataset.

---

# 11. Important Concepts & Terminology

## 11.1 Ensemble Learning

**Definition:** Ensemble learning combines multiple models to produce one final prediction.

**Why it matters:** Different models can make different errors. Combining them can improve stability and generalization.

---

## 11.2 Bootstrap Sampling

**Definition:** Bootstrap sampling creates a new dataset by drawing observations from the original dataset **with replacement**.

**Why it matters:** Every tree receives a different training sample.

---

## 11.3 Bagging

**Definition:** Bagging, or Bootstrap Aggregating, trains multiple models on bootstrap samples and combines their predictions.

**Why it matters:** Bagging is primarily used to reduce variance.

---

## 11.4 Random Feature Selection

**Definition:** At each split, Random Forest considers only a randomly selected subset of the available features.

**Why it matters:** It reduces correlation between trees.

---

## 11.5 Base Learner

**Definition:** The individual model used inside an ensemble.

**For Random Forest:** The base learner is a Decision Tree.

---

## 11.6 Tree Correlation

**Definition:** Tree correlation describes how similarly different trees behave or how similar their prediction errors are.

**Why it matters:** Highly correlated trees provide less variance reduction when averaged.

---

## 11.7 Out-of-Bag Sample

**Definition:** An OOB sample is a training observation that was not selected in the bootstrap sample used to train a particular tree.

**Why it matters:** OOB observations can be used to estimate generalization performance.

---

## 11.8 OOB Score

**Definition:** The OOB score estimates generalization performance using predictions made by trees for which each training observation was OOB.

**Why it matters:** It can provide an internal performance estimate when bootstrap sampling is used.

---

## 11.9 Feature Importance

**Definition:** Feature importance measures how much a feature contributes to the predictive behavior of the forest.

Two common approaches are:

- Mean decrease in impurity (MDI)
- Permutation importance

**Why it matters:** It helps understand which features contribute to model predictions.

> Interview caution: feature importance is not automatically the same as causal importance.

---

## 11.10 MDI

**Definition:** Mean Decrease in Impurity measures the total reduction in node impurity contributed by a feature, averaged across the trees.

**Why it matters:** It is fast, but can be biased toward some high-cardinality or continuous features.

---

## 11.11 Permutation Importance

**Definition:** Permutation importance measures performance degradation after randomly shuffling one feature while keeping other features unchanged.

Conceptually:

$$
Importance_j = Score_{\text{original}} - Score_{\text{permuted feature }j}
$$

**Why it matters:** It evaluates feature usefulness based on its effect on model performance.

---

## 11.12 Randomness in Random Forest

The name "Random Forest" comes from the randomized construction of the trees.

The major sources are:

1. Bootstrap sampling.
2. Random feature selection at each split.

Depending on the implementation, additional randomness may also occur in how candidate split points are searched.

---

# 12. Parameters vs Hyperparameters

## Parameters

**Definition:** Parameters are quantities learned from the training data.

### Examples

In Random Forest, examples include:

- Split thresholds.
- Selected features at tree nodes.
- Tree structure.
- Leaf predictions.
- Class proportions in leaves.

These are not manually chosen in the usual training process.

## Hyperparameters

**Definition:** Hyperparameters are settings chosen by the practitioner before or during model training and are not learned as ordinary model parameters.

### Examples

- `n_estimators`
- `max_depth`
- `max_features`
- `min_samples_split`
- `min_samples_leaf`
- `bootstrap`
- `max_samples`
- `criterion`
- `max_leaf_nodes`
- `class_weight`

## Key Difference

| Parameters | Hyperparameters |
|---|---|
| Learned from data | Chosen before training or by tuning |
| Define the fitted model | Control how the model is built |
| Example: split threshold | Example: `max_depth` |
| Example: leaf value | Example: `n_estimators` |

---

# 13. Important Hyperparameters

| Hyperparameter | Meaning | Increase → | Decrease → | Main Effect |
|---|---|---|---|---|
| `n_estimators` | Number of trees | More stable, more compute | Faster, potentially noisier | Controls ensemble size |
| `max_depth` | Maximum depth of each tree | More complex trees | Simpler trees | Controls tree complexity |
| `max_features` | Features considered at each split | More information per split, potentially more correlation | More diversity | Controls tree correlation |
| `min_samples_split` | Minimum samples needed to split a node | More regularization | Easier splitting | Controls complexity |
| `min_samples_leaf` | Minimum samples in a leaf | Smoother leaves | More detailed leaves | Controls complexity |
| `max_samples` | Samples drawn for each tree when bootstrap is enabled | More data per tree | More sample diversity | Controls sample randomness |
| `bootstrap` | Whether bootstrap samples are used | Enables bagging/OOB | Full-data training per tree | Controls sampling method |
| `max_leaf_nodes` | Maximum leaf count | More complex tree | Simpler tree | Controls complexity |
| `criterion` | Split quality measure | Depends on criterion | Depends on criterion | Controls split evaluation |
| `class_weight` | Class weighting | More focus on minority class if weighted | Less class adjustment | Helps with imbalance |
| `oob_score` | Whether to calculate OOB score | Enables OOB estimate | No OOB calculation | Internal evaluation |

## Most Important Hyperparameters

Focus especially on:

1. `n_estimators`
2. `max_depth`
3. `max_features`
4. `min_samples_leaf`
5. `min_samples_split`

### `n_estimators`

Controls the number of trees.

As it increases:

- Ensemble variance usually decreases.
- Predictions become more stable.
- Training and prediction take longer.
- Memory usage increases.

A very large number of trees usually gives diminishing returns.

### `max_depth`

Controls the maximum depth of each tree.

Higher:

- More complex trees.
- Lower bias.
- Potentially higher variance.

Lower:

- Simpler trees.
- Higher bias.
- Lower variance.

### `max_features`

Controls how many features are considered at each split.

Higher:

- Each tree sees more information.
- Trees can become more similar.
- Correlation can increase.

Lower:

- More randomness.
- Lower tree correlation.
- Each tree may become weaker.

This is one of the most important bias-variance-correlation controls in Random Forest.

### `min_samples_leaf`

Controls the minimum number of samples allowed in a leaf.

Higher:

- Prevents extremely small leaves.
- Smooths the model.
- Reduces overfitting.

Lower:

- Allows more detailed partitions.
- Can fit training data more closely.

### `min_samples_split`

Controls the minimum number of samples required before a node can split.

Higher:

- Fewer splits.
- Simpler trees.

Lower:

- More splits.
- More complex trees.

---

## Hyperparameter Interactions

Hyperparameters should not be treated independently.

For example:

```text
max_depth ↓
    ↓
simpler trees
    ↓
less overfitting
```

while:

```text
max_features ↓
    ↓
more tree diversity
    ↓
lower tree correlation
```

A common tuning process is:

1. Use enough trees to make the ensemble stable.
2. Tune tree complexity.
3. Tune feature subsampling.
4. Tune leaf/split constraints.
5. Evaluate with cross-validation or an OOB estimate where appropriate.

---

# 14. Bias-Variance & Generalization

## Bias

Bias is the error caused by a model being systematically too simple or making restrictive assumptions.

Random Forest generally has relatively low bias because Decision Trees can represent complex, non-linear relationships.

However, very strong feature randomness, shallow trees, or strong regularization can increase bias.

## Variance

Variance measures how much model predictions change when the training dataset changes.

A single deep Decision Tree can have high variance.

Random Forest reduces variance by averaging many trees.

## Overfitting

### Why Can This Algorithm Overfit?

Random Forest is resistant to overfitting compared with a single deep tree, but it is not immune.

Overfitting can still occur when:

- The data is very noisy.
- The dataset is small relative to problem complexity.
- There are many weak/noisy features.
- The evaluation setup contains leakage.
- Hyperparameters lead to unsuitable tree complexity.
- The training and deployment distributions differ.

## Signs of Overfitting

Typical signs:

```text
Training score      → very high
Validation score    → significantly lower
Test score          → significantly lower
```

## Underfitting

### Why Can This Algorithm Underfit?

Underfitting may occur when:

- Trees are too shallow.
- `min_samples_leaf` is too large.
- `min_samples_split` is too large.
- `max_features` is too small.
- Strong regularization is used.
- The feature set contains weak information.

### Signs of Underfitting

Typical signs:

```text
Training score      → poor
Validation score    → also poor
```

## Bias-Variance Trade-off

```text
Model Complexity
       ↓
Underfitting → Good Generalization → Overfitting
      ↓               ↓                 ↓
  High Bias       Balanced         High Variance
```

For Random Forest, an additional concept is important:

```text
Tree strength + tree correlation
          ↓
      Forest quality
```

Very strong but highly correlated trees may provide less benefit than strong, less correlated trees.

## Controlling Overfitting

- Reduce `max_depth`.
- Increase `min_samples_leaf`.
- Increase `min_samples_split`.
- Reduce `max_features` when appropriate.
- Use OOB evaluation or cross-validation.
- Remove leakage.
- Improve data quality and feature selection.
- Tune the model instead of blindly increasing tree complexity.

## Controlling Underfitting

- Increase `max_depth`.
- Reduce `min_samples_leaf`.
- Reduce `min_samples_split`.
- Increase `max_features` when appropriate.
- Add informative features.
- Check whether the problem is too noisy for the available data.

---

# 15. Assumptions

| Assumption | Required? | Why? | What If Violated? |
|---|---|---|---|
| Linear relationship | No | Trees model non-linear relationships | No direct issue |
| Normally distributed features | No | Trees do not require Gaussian input | No direct issue |
| Feature scaling | No | Splits depend on feature thresholds, not distance | Usually no issue |
| Independent observations | Preferable | Standard statistical evaluation assumes meaningful sampling | Correlated samples can distort evaluation |
| All features must be informative | No | Forest can ignore weak features | Too much noise can reduce performance |
| Features must have the same scale | No | Tree split rules are scale-invariant | Usually no issue |
| Low multicollinearity | No | Trees can handle correlated predictors | Importance can be distributed or become unstable |

## Important Interview Distinction

Do not say:

> "Random Forest has no assumptions."

A better interview answer is:

> **Random Forest makes far fewer distributional assumptions than many parametric models. It does not require linearity, normality, or feature scaling, but data quality, representative sampling, and leakage-free evaluation still matter.**

---

# 16. Data Preprocessing

## 16.1 Missing Values

Random Forest is less sensitive to preprocessing requirements than distance-based or many linear models.

For a robust workflow:

- Inspect missing values.
- Use imputation when the selected implementation does not natively support the missing-value pattern.
- Ensure the imputer is fitted only on training data when using a train/test split.

### Interview answer

> **Random Forest does not require feature scaling, but missing values still need to be handled according to the implementation being used.**

Scikit-Learn support for missing values depends on the exact estimator and version, so check the documentation for the version being used.

## 16.2 Feature Scaling

**Required?** No.

**Why?**

Tree splits are based on comparisons such as:

$$
x_j < t
$$

Suppose a feature is transformed by a positive scaling:

$$
x'_j = ax_j,\quad a>0
$$

A threshold transforms as:

$$
t'=at
$$

The ordering of observations does not change.

Therefore, a tree can create an equivalent partition of the data.

### Interview answer

> **Random Forest does not require standardization or normalization because Decision Trees split based on feature thresholds rather than distances or gradient magnitudes.**

---

## 16.3 Categorical Variables

A standard Scikit-Learn Random Forest workflow generally expects numerical input.

Therefore categorical variables often need encoding.

Common approach:

- One-hot encoding for nominal categories.
- Ordinal encoding only when an actual order is meaningful or when the chosen implementation supports the intended semantics appropriately.

Avoid assigning arbitrary numbers to nominal categories and then interpreting the numbers as meaningful order.

---

## 16.4 Outliers

Random Forest is relatively robust to outliers compared with models based on distances or linear coefficients.

Why?

Trees mainly care about split ordering and threshold partitions.

However, extreme or noisy observations can still:

- Influence candidate splits.
- Affect leaf distributions.
- Reduce performance when the outlier reflects data-quality problems.

Therefore:

> Robustness to outliers does not mean outliers can always be ignored.

---

## 16.5 Multicollinearity

Random Forest can work with correlated features.

Unlike ordinary least squares regression, it does not require low multicollinearity for the basic model to function.

However:

- Correlated features may compete for the same splits.
- Feature importance can become distributed across correlated variables.
- MDI importance may be misleading in the presence of many correlated or high-cardinality features.

For importance analysis, permutation importance and domain knowledge should be considered together.

---

## 16.6 Feature Engineering

Random Forest can learn:

- Non-linear relationships.
- Threshold effects.
- Feature interactions.

Therefore, it often needs less manual transformation than linear models.

Still, useful feature engineering can improve performance when:

- Raw variables are poorly aligned with the target.
- Domain relationships are known.
- Important information can be represented more clearly.

---

# 17. Model Complexity

## What Controls Complexity?

Main controls include:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_leaf_nodes`
- `max_features`

The number of trees mainly controls ensemble size and stability rather than the complexity of each individual tree.

## Simple Model

Example:

```text
Shallow trees
Large leaf sizes
```

Behavior:

- Faster.
- Easier to regularize.
- May have higher bias.

## Complex Model

Example:

```text
Deep trees
Small leaves
Many possible splits
```

Behavior:

- Low training bias.
- Can fit detailed patterns.
- Individual trees have high variance.

## Effect on Generalization

Random Forest generalization depends on both:

```text
Individual tree complexity
+
Number of trees
+
Correlation between trees
```

A useful mental model:

> Deep trees can be acceptable in a Random Forest because aggregation reduces their variance, but unrestricted complexity is not automatically optimal.

---

# 18. Computational Complexity

## Training Complexity

There is no single exact Big-O expression that describes every implementation because complexity depends on:

- Number of trees $B$
- Number of samples $n$
- Number of features $d$
- Number of candidate features per split
- Tree depth
- Split-search implementation
- Number of nodes

A useful high-level view is:

$$
O\left( B \times \text{cost of building one tree} \right)
$$

For a rough comparison, a tree-growing cost is often expressed in terms related to:

$$
O(B \cdot n \log n \cdot m_{\text{try}})
$$

for simplified settings, but actual implementation complexity can differ.

### Why?

Training must:

1. Build many trees.
2. Search candidate splits.
3. Repeat recursively over tree nodes.

Because trees can be trained independently, the forest is highly parallelizable.

## Prediction Complexity

Prediction requires passing each sample through all trees.

A useful high-level form is:

$$
O(B \cdot D)
$$

per sample, where $D$ is the average or maximum tree depth.

More precisely, prediction cost also depends on:

- Number of trees.
- Number of nodes traversed.
- Number of samples being predicted.

## Space Complexity

The model stores all trees, so memory grows roughly with:

$$
O(\text{total number of tree nodes})
$$

Increasing:

- `n_estimators`
- Tree depth
- Leaf count

can increase memory usage.

## Scalability

### More Samples

More samples:

- Increase training time.
- Increase memory requirements.
- Can improve generalization when more representative data is available.

### More Features

More features:

- Increase split-search cost.
- Can increase memory and training time.
- May increase noise if many features are irrelevant.

### High-Dimensional Data

Random Forest can work well when:

- There are many features.
- Non-linear relationships matter.
- Feature selection is not known in advance.

However, extremely high-dimensional sparse data may sometimes favor specialized linear or sparse methods.

---

# 19. Regularization / Optimization Improvements

## Why Is It Needed?

Random Forest already has built-in variance reduction, but trees can still be unnecessarily complex.

Tree-level regularization can:

- Reduce overfitting.
- Improve computational efficiency.
- Produce smoother predictions.

## Method 1 — Limit Tree Depth

Use:

```python
max_depth
```

Smaller maximum depth produces simpler trees.

## Method 2 — Increase Minimum Leaf Size

Use:

```python
min_samples_leaf
```

Larger leaf size prevents the model from creating extremely small terminal regions.

## Method 3 — Limit Splits

Use:

```python
min_samples_split
```

A larger value requires more observations before a node can split.

## Method 4 — Limit Number of Leaves

Use:

```python
max_leaf_nodes
```

This directly limits the number of terminal regions.

## Method 5 — Reduce Correlation

Use:

```python
max_features
```

A smaller feature subset can make trees more different.

## Effect on Model

Regularization typically produces:

```text
Simpler trees
      ↓
Higher bias
      +
Lower variance / less overfitting
      ↓
Potentially better test performance
```

The goal is not the simplest possible forest.

The goal is good generalization.

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

Prefer metrics such as:

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
- $R^2$ measures explained variance relative to a baseline, but should not be used alone.

## Cross-Validation

Cross-validation is commonly used to estimate generalization and tune hyperparameters.

For classification, use stratified cross-validation when appropriate.

For regression, standard K-fold cross-validation is common.

The important rule is:

> **Preprocessing and tuning must be performed inside the cross-validation workflow to avoid data leakage.**

For example, use a `Pipeline` when imputation or encoding must be learned from training folds.

---

# 21. Advantages

## 1. Reduces Variance

**Why?**

Averaging many trees makes the overall model less sensitive to the peculiarities of one training set.

---

## 2. Handles Non-Linear Relationships

**Why?**

Decision Trees naturally model threshold effects and feature interactions.

---

## 3. Little Need for Feature Scaling

**Why?**

Tree splits depend on ordering and thresholds rather than distance or coefficient magnitude.

---

## 4. Works for Classification and Regression

**Why?**

The same ensemble idea can aggregate class predictions or continuous predictions.

---

## 5. Robust and Flexible Baseline

**Why?**

Random Forest often performs strongly with relatively little preprocessing and can capture complex interactions without requiring a linear functional form.

---

# 22. Disadvantages

## 1. Less Interpretable Than One Decision Tree

**Why?**

A single tree can often be visualized directly.

A forest contains many trees, so explaining one global decision path is harder.

---

## 2. Larger Computational and Memory Cost

**Why?**

Instead of storing one tree, the model stores many.

Training and prediction costs increase with the number and size of trees.

---

## 3. Feature Importance Can Mislead

**Why?**

Impurity-based feature importance can favor certain feature types and can distribute importance in unintuitive ways when predictors are correlated.

---

## 4. Can Be Weaker Than Boosting on Some Structured Tabular Problems

**Why?**

Boosting methods optimize sequentially to correct previous errors, whereas Random Forest trains trees independently.

This does not mean boosting always wins.

---

# 23. Failure Modes

## When Does It Perform Poorly?

Random Forest can perform poorly when:

- The dataset has very weak predictive signal.
- Labels are very noisy.
- The dataset is too small for the problem complexity.
- Important patterns require extrapolation beyond the observed training range.
- Very high-dimensional data contains large amounts of irrelevant noise.
- The target relationship is better captured by a model family with stronger domain structure.
- The evaluation setup contains leakage or distribution shift.

## Why Does It Fail?

### Weak Signal

If features contain little information about the target, a more complex ensemble cannot create useful signal.

### Extrapolation

Trees partition observed ranges.

For regression, they generally do not extrapolate like linear models.

For example, if all training target relationships are observed between certain feature values, the forest does not naturally extend a straight-line trend beyond the training range.

### Distribution Shift

Performance can fall when test data comes from a different distribution.

## Warning Signs

- Large train-test gap.
- Large cross-validation variance.
- Strong OOB score but poor real-world test performance.
- Performance collapses on future or shifted data.
- Feature importance changes drastically between splits.

## How Can We Detect the Problem?

Use:

- Train/validation/test comparison.
- Cross-validation.
- OOB score when applicable.
- Error analysis.
- Confusion matrix.
- Residual analysis for regression.
- Feature distribution checks.
- Leakage checks.
- Time-based validation for temporal data.

## Possible Solutions

Depending on the cause:

- Improve data quality.
- Engineer better features.
- Tune tree complexity.
- Handle imbalance.
- Collect more representative data.
- Use a suitable validation strategy.
- Compare against boosting, linear, or other model families.

---

# 24. Algorithm-Specific Edge Cases

## Case 1 — Very Small Dataset

Problem:

- Different bootstrap samples may contain limited unique information.
- Individual trees can become highly variable.
- OOB estimates may be unstable.

Approach:

- Use cross-validation.
- Keep the model simple.
- Compare against simpler baselines.

---

## Case 2 — Highly Imbalanced Classes

Problem:

A forest can still favor the majority class.

Approach:

- Use appropriate metrics.
- Consider `class_weight`.
- Consider resampling strategies.
- Inspect precision, recall, F1, PR-AUC, and confusion matrix.

---

## Case 3 — Many Correlated Features

Problem:

Correlated features may:

- compete for splits,
- increase tree similarity,
- make feature importance harder to interpret.

Approach:

- Compare model performance with/without redundant features.
- Use permutation importance carefully.
- Evaluate correlated feature groups rather than interpreting one importance number as causal evidence.

---

## Case 4 — Regression Extrapolation

Random Forest regression predicts using terminal-node averages.

Therefore it is generally poor at extrapolating beyond the range of patterns represented in training data.

---

## Case 5 — Noisy Labels

If labels are heavily corrupted, trees may fit noise.

Increasing model complexity cannot recover information that is not present in the labels.

---

## Case 6 — High-Cardinality Features

Features with many unique values can receive high impurity-based importance even when their true predictive value is limited.

Use permutation importance and validation-based analysis when interpreting feature importance.

---

## Case 7 — Time-Series Data

Do not blindly use random train/test splitting when the data has time order.

Use time-aware validation when future prediction is the goal.

Otherwise, information from the future may leak into training.

---

## Case 8 — Special Hyperparameter Values

Some parameter combinations can cause:

- Extremely large trees.
- High memory consumption.
- Long training times.
- Unstable estimates on small datasets.

Always inspect the fitted model and validation behavior.

---

# 25. Practical Implementation — Scikit-Learn

## Classification Import

```python
from sklearn.ensemble import RandomForestClassifier
```

## Regression Import

```python
from sklearn.ensemble import RandomForestRegressor
```

## Create Classification Model

```python
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    max_features="sqrt",
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

## Create Regression Model

```python
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    max_features=1.0,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

> The exact defaults can vary across Scikit-Learn versions. For interview answers, explain the role of a parameter rather than memorizing every default value.

## Train

```python
model.fit(X_train, y_train)
```

## Predict

```python
y_pred = model.predict(X_test)
```

## Classification Probability

```python
y_prob = model.predict_proba(X_test)
```

## Evaluate Classification

```python
from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_test, y_pred))
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
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)
```

## OOB Evaluation

For a forest using bootstrap sampling:

```python
model = RandomForestClassifier(
    n_estimators=300,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print(model.oob_score_)
```

The OOB score is based on OOB predictions rather than predictions from trees that trained on the same observation.

## Feature Importance

```python
import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(importance)
```

> Treat impurity-based importance as a model diagnostic, not as proof that a feature causes the target.

## Permutation Importance

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance = pd.Series(
    result.importances_mean,
    index=X_test.columns
).sort_values(ascending=False)

print(importance)
```

---

# 26. From-Scratch Implementation

> Implement the core algorithm without using the model provided by Scikit-Learn.

A complete production-quality implementation requires many details, but the learning version can demonstrate the core mechanism.

```python
import numpy as np
from collections import Counter


class SimpleRandomForestClassifier:
    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        max_features=None,
        min_samples_split=2,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees_ = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = X.shape[0]

        self.trees_ = []

        for _ in range(self.n_estimators):

            # Bootstrap sample
            indices = rng.integers(
                0,
                n_samples,
                size=n_samples
            )

            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # In a real implementation, a Decision Tree
            # would be created here with random feature
            # selection at every split.
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                random_state=rng.integers(0, 1_000_000)
            )

            tree.fit(X_bootstrap, y_bootstrap)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)

        tree_predictions = np.array([
            tree.predict(X)
            for tree in self.trees_
        ])

        forest_predictions = []

        for sample_predictions in tree_predictions.T:
            forest_predictions.append(
                Counter(sample_predictions).most_common(1)[0][0]
            )

        return np.array(forest_predictions)
```

The exact `SimpleDecisionTree` implementation is omitted here because the important Random Forest logic is:

```text
Bootstrap sample
       ↓
Build randomized tree
       ↓
Repeat many times
       ↓
Collect tree predictions
       ↓
Majority vote / average
```

## Code ↔ Mathematics

| Code Component | Mathematical / Algorithmic Concept |
|---|---|
| `rng.integers(...)` | Bootstrap sampling |
| `X[indices]` | Bootstrap dataset $D_b^*$ |
| `n_estimators` | Number of trees $B$ |
| `tree.fit(...)` | Decision Tree training |
| `tree.predict(...)` | $T_b(x)$ |
| `Counter(...).most_common(1)` | Majority vote |
| Mean of tree predictions | Regression aggregation |
| `max_features` | Random feature selection |

## Important Implementation Details

A learning implementation should handle:

- Bootstrap sampling.
- Random feature subsets.
- Recursive tree construction.
- Stopping conditions.
- Split scoring.
- Leaf prediction.
- Majority voting for classification.
- Averaging for regression.
- Random seeds for reproducibility.

A production implementation also needs careful handling of:

- Missing values.
- Sample weights.
- Class weights.
- Parallelism.
- Memory.
- Edge cases.
- Efficient split search.
- Probability prediction.

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
Scaling (usually not required)
    ↓
Baseline Model
    ↓
Random Forest
    ↓
Cross-Validation / OOB Evaluation
    ↓
Hyperparameter Tuning
    ↓
Final Evaluation
    ↓
Interpretation
```

## Algorithm-Specific Considerations

Pay special attention to:

1. Whether the validation strategy matches the data-generating process.
2. Whether the target is imbalanced.
3. Whether bootstrap sampling is enabled.
4. Whether OOB evaluation is appropriate.
5. Whether tree complexity is excessive.
6. Whether correlated features distort feature importance.
7. Whether the model needs to extrapolate.
8. Whether a boosting method should also be evaluated.

### Practical Baseline

A strong first baseline is often:

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
```

or the regression equivalent.

Then tune based on validation performance rather than changing many parameters at once.

---

# 28. Comparison With Related Algorithms

## Random Forest vs Decision Tree

| Aspect | Random Forest | Decision Tree |
|---|---|---|
| Core Idea | Many randomized trees | One tree |
| Assumptions | Few distributional assumptions | Few distributional assumptions |
| Bias | Usually low | Usually low if tree is deep |
| Variance | Lower due to aggregation | Often high |
| Scaling | Not required | Not required |
| Interpretability | Lower | Higher |
| Training Speed | Slower | Faster |
| Prediction Speed | Slower | Faster |
| Overfitting | More resistant | More prone |
| Strength | Stability and robustness | Simplicity and interpretability |
| Weakness | Less interpretable, more compute | High variance |
| Typical Use Case | Strong tabular baseline | Explainable rule-based model |

## Random Forest vs Bagging

| Aspect | Random Forest | Bagging |
|---|---|---|
| Core Idea | Bootstrap + random features | Bootstrap + aggregation |
| Base Model | Decision Trees | Can use many model types |
| Random Features | Yes | Not required |
| Main Diversity Mechanism | Samples + features | Mainly samples |
| Goal | Reduce variance and tree correlation | Reduce variance |
| Typical Implementation | Randomized trees | Generic bagging framework |

### Key Distinction

> **Bagging randomizes the training observations; Random Forest additionally randomizes the candidate features at each split.**

---

## Random Forest vs Extra Trees

| Aspect | Random Forest | Extra Trees |
|---|---|---|
| Bootstrap samples | Usually yes by default in common RF formulation | Often no by default in Scikit-Learn |
| Feature randomness | Yes | Yes |
| Threshold selection | Searches for a good threshold among candidate splits | Uses more randomized thresholds |
| Diversity | High | Often higher |
| Bias | Can be lower | Can be slightly higher |
| Variance | Low | Can be very low |
| Typical role | Robust tree ensemble | Highly randomized tree ensemble |

### Key Distinction

> **Extra Trees injects more randomness into split thresholds, while Random Forest generally searches for a better split on the selected features.**

---

## Random Forest vs Gradient Boosting

| Aspect | Random Forest | Gradient Boosting |
|---|---|---|
| Core Idea | Independent randomized trees | Sequential trees correcting previous errors |
| Training | Mostly parallel | Sequential dependency across stages |
| Main Objective | Variance reduction / averaging | Reduce loss stage by stage |
| Tree Relationship | Trees are largely independent | Later trees depend on earlier trees |
| Overfitting Control | Tree + feature + sample randomization | Learning rate, depth, number of stages, regularization |
| Interpretability | Moderate to low | Moderate to low |
| Training Parallelism | High | Lower across boosting stages |
| Common Strength | Robust baseline | Often strong predictive performance on structured tabular data |
| Main Risk | Large memory / model size | Sensitive to tuning |

### Key Distinction

> **Random Forest builds trees independently and averages them; Gradient Boosting builds trees sequentially to correct previous errors.**

---

## Random Forest vs Logistic Regression

| Aspect | Random Forest | Logistic Regression |
|---|---|---|
| Core Idea | Ensemble of trees | Linear model for log-odds |
| Relationship | Non-linear | Linear in features after transformation |
| Scaling | Usually not required | Often useful depending on regularization/solver |
| Interactions | Learned automatically | Usually require explicit feature terms |
| Interpretability | Lower | Higher |
| Extrapolation | Poor | Linear behavior can extrapolate |
| Feature Effects | Complex | Coefficients are easier to inspect |

## Key Distinction

> **Logistic Regression learns a linear decision function, while Random Forest learns a set of non-linear decision regions through many trees.**

---

# 29. When Should I Use This Algorithm?

Use it when:

- You have structured/tabular data and expect non-linear relationships.
- You need a strong baseline with relatively little preprocessing.
- You want a model that handles feature interactions automatically.
- You do not want to spend large effort on feature scaling.
- You need classification or regression.
- You value robustness more than a simple global explanation.

Typical examples:

- Customer churn prediction.
- Fraud detection.
- Loan default classification.
- Customer segmentation support tasks.
- Demand prediction.
- Risk scoring.
- Tabular business analytics problems.

---

# 30. When Should I Avoid This Algorithm?

Consider another approach when:

- Interpretability must be extremely simple.
- You need smooth extrapolation in regression.
- The data is extremely sparse/high-dimensional and a linear model is more suitable.
- Latency or memory constraints make a large forest impractical.
- The problem is naturally sequential and requires time-aware structure.
- A carefully tuned boosting method is expected to provide better performance for the dataset and computational budget.

Do not reject Random Forest merely because another model is more modern.

Use validation evidence.

---

# 31. Algorithm Selection Guide

When facing a new ML problem:

```text
What type of problem?
        ↓
Classification / Regression
        ↓
Is the data mainly structured/tabular?
        ↓
Yes
        ↓
Do I need a strong non-linear baseline?
        ↓
Yes
        ↓
Try Random Forest
        ↓
Evaluate with appropriate CV / OOB strategy
        ↓
Compare with:
    - Logistic / Linear models
    - Gradient Boosting
    - Extra Trees
    - Other domain-appropriate methods
        ↓
Select using validation evidence
```

### Quick Decision Logic

```text
Need simple explanation?
    → Decision Tree / Linear model

Need non-linear tabular baseline?
    → Random Forest

Need sequential error correction and strong tabular performance?
    → Gradient Boosting family

Need linear decision boundary?
    → Linear / Logistic Regression
```

---

# 32. Common Misconceptions

## Misconception 1

> "Random Forest is just many Decision Trees."

**Correction:**

It is an ensemble of Decision Trees **with deliberate randomization**.

The important ideas are:

- Bootstrap sampling.
- Random feature selection.
- Aggregation.

Simply training many identical trees would not provide the same benefit.

---

## Misconception 2

> "More trees always improve the model's test accuracy."

**Correction:**

More trees usually make the ensemble more stable and can reduce variance, but performance eventually shows diminishing returns.

A larger forest also increases computation and memory.

---

## Misconception 3

> "Random Forest cannot overfit."

**Correction:**

Random Forest is resistant to overfitting compared with a single deep tree, but it can still overfit noisy data, suffer from leakage, or generalize poorly under distribution shift.

---

## Misconception 4

> "Random Forest needs StandardScaler."

**Correction:**

No.

Tree splits are based on ordered thresholds, so feature scaling is usually unnecessary.

---

## Misconception 5

> "Random Forest uses gradient descent."

**Correction:**

No.

Decision Trees are built using greedy split selection, not gradient descent.

---

## Misconception 6

> "Bootstrap means every tree sees only 63% of the training set."

**Correction:**

A bootstrap sample contains $n$ draws, but due to repeated sampling, only about 63.2% of the original observations are unique on average.

About 36.8% are OOB for a given tree.

---

## Misconception 7

> "Random Forest reduces bias."

**Correction:**

Its main statistical advantage is **variance reduction**.

Changing tree depth and feature randomness can affect bias too, but the core ensemble mechanism is variance reduction through aggregation and decorrelation.

---

## Misconception 8

> "Feature importance tells me which feature causes the target."

**Correction:**

Feature importance measures predictive contribution under a specific importance method.

It does not establish causality.

---

# 33. Common Implementation Mistakes

## Mistake 1 — Data Leakage

Example:

```text
Fit imputer/scaler/feature selector
on the full dataset
        ↓
Then split into train/test
```

This allows information from the test set to influence training.

Correct approach:

```text
Split data
   ↓
Fit preprocessing only on training data
   ↓
Transform validation/test data
```

Use a Scikit-Learn `Pipeline` when appropriate.

---

## Mistake 2 — Using Accuracy for Severe Class Imbalance

A model can get high accuracy while completely missing the minority class.

Use:

- Precision
- Recall
- F1
- PR-AUC
- Confusion matrix

as appropriate.

---

## Mistake 3 — Assuming More Trees Fix Bad Features

Increasing `n_estimators` cannot compensate for:

- Weak signal.
- Poor labels.
- Data leakage.
- Incorrect target definition.
- Distribution shift.

---

## Mistake 4 — Ignoring Correlation When Interpreting Importance

A feature can appear less important simply because correlated features share the predictive information.

Do not treat one importance ranking as absolute truth.

---

## Mistake 5 — Blind Hyperparameter Tuning

Changing many hyperparameters without a validation strategy can overfit the validation process itself.

Use:

- Cross-validation.
- A stable metric.
- A final untouched test set.

---

## Mistake 6 — Random Split for Time-Dependent Data

Random splitting can leak future information into training.

Use time-aware validation for forecasting or future prediction problems.

---

## Mistake 7 — Comparing Models on Different Data Splits

A fair comparison should use the same training/validation protocol.

Otherwise differences may come from sampling rather than model quality.

---

# 34. Interview Questions

## Basic

### Q1. What is Random Forest?

**Answer:**

> Random Forest is a supervised ensemble learning algorithm that trains multiple Decision Trees using bootstrap samples of the training data and random subsets of features at each split. It combines the tree predictions using majority voting for classification and averaging for regression. Its main purpose is to reduce variance and improve generalization.

---

### Q2. How does Random Forest work?

**Answer:**

> For each tree, Random Forest creates a bootstrap sample from the training data. While growing the tree, it randomly selects a subset of features at every split and chooses the best split among those features. It repeats this for many trees and then aggregates their predictions.

---

### Q3. What type of problems can it solve?

**Answer:**

> Random Forest can solve both classification and regression problems. It is especially useful for structured tabular data with non-linear relationships and feature interactions.

---

## Intermediate

### Q4. What are the assumptions of Random Forest?

**Answer:**

> Random Forest does not require assumptions such as linearity or normally distributed features, and it usually does not require feature scaling. However, representative data, correct labels, leakage-free evaluation, and an appropriate validation strategy are still important.

---

### Q5. Does it require feature scaling?

**Answer:**

> No. Random Forest is based on Decision Trees, and tree splits depend on feature thresholds and ordering rather than distances or gradient magnitudes. Therefore StandardScaler or MinMaxScaler is usually unnecessary.

---

### Q6. What are its important hyperparameters?

**Answer:**

> Important hyperparameters include `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, and `min_samples_leaf`. `n_estimators` controls ensemble size, while the others mainly control tree complexity and tree diversity.

---

### Q7. What happens when `n_estimators` increases?

**Answer:**

> More trees usually make the prediction more stable and reduce the variance of the ensemble. After enough trees, the improvement becomes smaller while training time, prediction time, and memory usage continue to increase.

---

### Q8. What happens when `max_features` decreases?

**Answer:**

> Each tree considers fewer features at each split. This generally increases randomness and decreases correlation between trees. It can reduce variance through better diversification, but if made too small it can weaken the individual trees and increase bias.

---

## Advanced

### Q9. Why does Random Forest reduce overfitting compared with a single Decision Tree?

**Answer:**

> A single deep tree has high variance. Random Forest trains many different trees and averages or votes over them. If the tree errors are not perfectly correlated, aggregation reduces variance. Bootstrap sampling and random feature selection help make the trees less correlated.

---

### Q10. Why is random feature selection necessary?

**Answer:**

> If every tree considered all features at every split, many trees could choose the same strong features and become highly correlated. Averaging highly correlated trees gives less variance reduction. Random feature selection forces trees to explore different feature subsets and therefore reduces correlation.

---

### Q11. Why can averaging reduce variance?

**Answer:**

> If model errors are partly independent, averaging cancels part of the random fluctuation. With correlated trees, the ensemble variance depends on both the variance of individual trees and their correlation. Lower correlation produces a larger benefit from averaging.

---

### Q12. What are OOB samples?

**Answer:**

> OOB samples are observations not selected in a tree's bootstrap sample. On average, about 36.8% of observations are OOB for a given tree. Those observations can be used to obtain predictions from trees that did not train on them.

---

### Q13. What is OOB error?

**Answer:**

> OOB error is an internal estimate of generalization error obtained by aggregating predictions for each training observation using only the trees for which that observation was OOB.

---

### Q14. Is Random Forest a bagging algorithm?

**Answer:**

> Yes, Random Forest contains the bagging idea because it trains trees on bootstrap samples and aggregates their predictions. It adds another important component: random feature selection at each split.

---

### Q15. Random Forest vs Bagging?

**Answer:**

> Bagging is a general ensemble technique based on bootstrap samples and aggregation. Random Forest is a specialized tree ensemble that adds random feature selection to reduce correlation between the trees.

---

### Q16. Random Forest vs Gradient Boosting?

**Answer:**

> Random Forest builds trees largely independently and aggregates them. Gradient Boosting builds trees sequentially, with later trees trying to reduce the errors or loss left by earlier trees. Random Forest is often easier to parallelize and acts as a strong baseline; boosting can be more sensitive to tuning and can achieve strong predictive performance on many tabular problems.

---

### Q17. Why does Random Forest not need feature scaling?

**Answer:**

> A split is based on whether a feature is above or below a threshold. A monotonic positive scaling changes the threshold but not the ordering of samples, so the same partition can still be created.

---

### Q18. Can Random Forest overfit?

**Answer:**

> Yes. It is more resistant to overfitting than a single deep tree, but it can still generalize poorly because of noisy data, weak signal, excessive complexity, leakage, or distribution shift.

---

### Q19. Does increasing the number of trees cause overfitting?

**Answer:**

> Increasing the number of trees usually reduces the variance of the forest and improves stability rather than causing the classic overfitting behavior associated with increasing the depth of one tree. However, computational cost continues to increase and performance eventually reaches diminishing returns.

---

### Q20. How does Random Forest handle outliers?

**Answer:**

> It is relatively robust to outliers because tree decisions are mainly based on feature ordering and thresholds. However, outliers can still influence splits and leaf predictions, especially when they represent noise or data-quality problems.

---

### Q21. How does Random Forest handle multicollinearity?

**Answer:**

> It can still train successfully with correlated features. The main concern is interpretation: correlated features may compete for splits, so importance can be distributed across them and impurity-based importance can become harder to interpret.

---

### Q22. Does Random Forest extrapolate well?

**Answer:**

> No. Random Forest regression predicts using terminal-node averages, so it generally does not extrapolate smoothly beyond the range represented in the training data.

---

### Q23. What is the main difference between a Random Forest and Extra Trees?

**Answer:**

> Both introduce randomness into tree construction. Random Forest usually uses bootstrap samples and then searches for good thresholds among randomly selected features. Extra Trees introduces even more randomness by selecting split thresholds more randomly.

---

### Q24. What is the role of `class_weight`?

**Answer:**

> `class_weight` changes the relative importance of classes during training. It can help when minority-class errors are more important or when classes are highly imbalanced.

---

### Q25. What is the difference between MDI and permutation importance?

**Answer:**

> MDI measures total impurity reduction caused by a feature across the forest. Permutation importance measures how model performance changes when a feature's values are shuffled. Permutation importance is often more directly tied to predictive usefulness on the evaluation data.

---

# 35. Interview Follow-Up Drill

The first answer is rarely the end of the interview.

### Interviewer: "Why?"

**Your answer:**

> Because averaging reduces variance, but the benefit is much larger when the individual trees are not highly correlated. Random feature selection makes the trees more diverse.

---

### Interviewer: "What happens if we increase `n_estimators`?"

**Your answer:**

> The forest usually becomes more stable and its variance decreases, with diminishing performance gains after enough trees. Training and prediction cost increase.

---

### Interviewer: "Why not use one very deep Decision Tree?"

**Your answer:**

> A single deep tree can fit the training set very closely and has high variance. Random Forest uses many trees and aggregation to retain flexible decision boundaries while reducing the variance of the final model.

---

### Interviewer: "Why not use all features at every split?"

**Your answer:**

> If all trees repeatedly use the same strongest features, their structures and errors can become highly correlated. Random feature selection lowers correlation and improves the benefit of averaging.

---

### Interviewer: "Does feature scaling matter here?"

**Your answer:**

> Usually no. Tree splits use threshold comparisons, so scaling does not change the ordering of observations or the basic partitions that the trees can create.

---

### Interviewer: "What happens with a very large dataset?"

**Your answer:**

> More data usually improves the reliability of the learned patterns but increases training time and memory usage. The forest is highly parallelizable, so multiple trees can be trained in parallel.

---

### Interviewer: "What if the data is highly imbalanced?"

**Your answer:**

> I would not rely on accuracy alone. I would inspect the confusion matrix and use metrics such as recall, precision, F1, PR-AUC, or ROC-AUC depending on the error costs. I would also consider class weighting or resampling.

---

### Interviewer: "Can Random Forest handle non-linear relationships?"

**Your answer:**

> Yes. Decision Trees partition feature space into regions, so the ensemble can model complex non-linear relationships and feature interactions without requiring a linear functional form.

---

### Interviewer: "Why can Random Forest still fail?"

**Your answer:**

> Because an ensemble cannot create signal that is absent from the data. It can also fail under severe distribution shift, poor labels, data leakage, extrapolation requirements, or highly noisy high-dimensional data.

---

### Interviewer: "Does Random Forest use gradient descent?"

**Your answer:**

> No. The individual trees use greedy split selection based on impurity reduction or a related criterion. There is no gradient-descent training process for the standard Random Forest algorithm.

---

# 36. Explain This Algorithm in an Interview

## 30-Second Explanation

> Random Forest is a supervised ensemble algorithm made of many Decision Trees. Each tree is trained on a bootstrap sample of the data, and at each split it considers only a random subset of features. This creates diverse, less-correlated trees. Their predictions are then combined using majority voting for classification or averaging for regression. The main advantage is variance reduction and better generalization compared with a single Decision Tree.

## 1-Minute Explanation

> Random Forest is essentially a combination of bagging and random feature selection for Decision Trees. Suppose we have a dataset with $n$ observations. For every tree, we draw a bootstrap sample of size $n$ with replacement. While building that tree, at every split we randomly choose a subset of features and find the best split among those features. We repeat this for many trees. For classification, the forest uses majority vote; for regression, it averages the tree predictions. The reason this works is that individual trees can have high variance, but averaging reduces variance, especially when the trees are not highly correlated. Random feature selection lowers this correlation.

## 3-Minute Explanation

> Random Forest is a non-parametric supervised ensemble algorithm for classification and regression. Its base learner is a Decision Tree.
>
> The motivation comes from the high variance of individual Decision Trees. A deep tree can fit complex patterns, but small changes in training data can produce a very different tree. Random Forest addresses this using two sources of randomness.
>
> First is bootstrap sampling. For every tree, we sample $n$ observations from the training set with replacement. Because of replacement, some samples appear multiple times and some are not selected. The observations not selected are called Out-of-Bag samples.
>
> Second is random feature selection. At each node, instead of allowing the tree to consider all features, Random Forest chooses only a random subset and finds the best split among those features. This makes the trees less correlated.
>
> We build many such trees. For a classification problem, each tree gives a class prediction and the final answer is the majority vote. For regression, we average the numeric predictions:
>
$$
\hat{y} = \frac{1}{B}\sum_{b=1}^{B}T_b(x)
$$
>
> The statistical reason this works is variance reduction. If the trees have variance $\sigma^2$ and are not perfectly correlated, averaging their predictions reduces the random part of the error. Therefore, Random Forest usually has much lower variance than a single deep tree while retaining the ability to model complex non-linear relationships.
>
> Important hyperparameters include `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, and `min_samples_leaf`.
>
> One of the strongest interview points is that Random Forest mainly reduces variance. It does not use gradient descent, it usually does not need feature scaling, and it can still fail if the data has weak signal, severe noise, leakage, or distribution shift.

---

# 37. Key Takeaways

## Core Idea

> **Train many diverse Decision Trees and aggregate their predictions.**

## Mathematical Idea

> **Bootstrap samples create different training sets, random feature subsets reduce tree correlation, and aggregation reduces variance.**

## Training Idea

> **For every tree: bootstrap the data, randomly select features at each split, greedily grow the tree, and repeat.**

## Prediction Idea

> **Classification → majority vote. Regression → average of tree predictions.**

## Main Strength

> **Strong non-linear modeling with variance reduction and relatively little preprocessing.**

## Main Limitation

> **Lower interpretability and higher computational/memory cost than a single tree; it also does not extrapolate naturally in regression.**

## Most Important Hyperparameters

> **`n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`, and `min_samples_split`.**

## Most Important Assumption

> **Random Forest has few strong distributional assumptions, but representative data and leakage-free evaluation are still essential.**

## Most Important Interview Concept

> **Random Forest works because it combines strong but diverse trees; averaging reduces variance, and random feature selection reduces correlation between trees.**

---

# 38. Completion Checklist

Before marking this algorithm as complete, I should be able to answer:

- [ ] What problem does it solve?
- [ ] Why do we need it?
- [ ] What is the formal definition?
- [ ] What is the core intuition?
- [ ] How is the problem mathematically formulated?
- [ ] What objective/loss function does it use at the tree level?
- [ ] Why are Gini, entropy, and MSE useful?
- [ ] Why does Random Forest not use gradient descent?
- [ ] What is bootstrap sampling?
- [ ] What is the 36.8% OOB result?
- [ ] What is random feature selection?
- [ ] Why does it reduce tree correlation?
- [ ] How does aggregation reduce variance?
- [ ] How does training work?
- [ ] What does the model actually learn?
- [ ] How does prediction work?
- [ ] What are OOB samples and OOB score?
- [ ] What assumptions does it make?
- [ ] Does feature scaling matter?
- [ ] How does it behave with outliers?
- [ ] How does it behave with correlated features?
- [ ] Why can it overfit?
- [ ] How can overfitting be controlled?
- [ ] What are the key hyperparameters?
- [ ] What happens when `n_estimators` increases?
- [ ] What happens when `max_depth` increases?
- [ ] What happens when `max_features` decreases?
- [ ] What happens when `min_samples_leaf` increases?
- [ ] What are its computational costs?
- [ ] Why is it easy to parallelize?
- [ ] What are its strengths and weaknesses?
- [ ] When should I use it?
- [ ] When should I avoid it?
- [ ] How does it compare with Decision Trees?
- [ ] How does it compare with Bagging?
- [ ] How does it compare with Extra Trees?
- [ ] How does it compare with Gradient Boosting?
- [ ] How does it compare with Logistic Regression?
- [ ] Can I explain the bootstrap derivation?
- [ ] Can I explain the bias-variance intuition?
- [ ] Can I explain why correlation between trees matters?
- [ ] Can I implement it using Scikit-Learn?
- [ ] Can I explain the implementation line-by-line?
- [ ] Can I answer "why?" follow-ups?
- [ ] Can I explain it in 30 seconds, 1 minute, and 3 minutes?

---

# 39. References

- Leo Breiman, **"Random Forests"**, Machine Learning, 2001.
- L. Breiman, **"Bagging Predictors"**, Machine Learning, 1996.
- Scikit-Learn — Random Forest User Guide:
  https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
- Scikit-Learn — `RandomForestClassifier`:
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Scikit-Learn — `RandomForestRegressor`:
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- Scikit-Learn — Permutation Importance:
  https://scikit-learn.org/stable/modules/permutation_importance.html
