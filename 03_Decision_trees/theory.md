# Decision Trees

> Complete theory, intuition, mathematics, practical understanding, implementation, and interview preparation.

---

## 0. Learning Objectives

By the end of this chapter, I should be able to:

- Explain Decision Trees intuitively.
- Give a formal, interview-ready definition.
- Explain why Decision Trees are needed and what problem they solve.
- Explain classification and regression trees.
- Understand entropy, Gini impurity, information gain, variance reduction, and mean squared error.
- Derive the split-selection criterion step-by-step.
- Explain recursive binary partitioning.
- Explain training and prediction completely.
- Manually build a small tree on paper.
- Explain overfitting, underfitting, pruning, and the bias-variance trade-off.
- Tune the important hyperparameters.
- Understand pre-pruning and post-pruning.
- Implement Decision Trees using Scikit-Learn.
- Implement the core idea from scratch.
- Explain how Decision Trees compare with Logistic Regression, k-NN, Random Forest, and Gradient Boosting.
- Answer placement/interview follow-up questions confidently.

---

# 1. Prerequisites

## Concepts I Should Know First

- **Supervised Learning** — learning a mapping from features `X` to a known target `y`.
- **Classification vs Regression** — classification predicts discrete classes; regression predicts continuous values.
- **Probability and proportions** — a tree node uses class proportions to estimate class probabilities.
- **Mean, variance, and MSE** — especially important for regression trees.
- **Logarithms** — required for understanding entropy and information gain.
- **Train/validation/test split and cross-validation** — required for measuring generalization.
- **Overfitting and underfitting** — trees can become extremely flexible, so model complexity matters.

## Connection With Previous Algorithms

Decision Trees are structurally different from Linear/Logistic Regression:

- Linear and Logistic Regression learn a global mathematical relationship using a fixed parametric form.
- A Decision Tree recursively partitions the feature space into smaller regions.
- Logistic Regression creates a linear decision boundary unless features are transformed.
- A Decision Tree can model highly non-linear interactions without manually creating polynomial or interaction terms.
- Unlike gradient-descent-based models, a standard Decision Tree does **not** learn weights by repeatedly taking derivatives.
- A single Decision Tree is a building block for ensemble methods such as **Random Forests** and **Gradient Boosted Trees**.

---

# 2. Why Do We Need This Algorithm?

## 2.1 The Problem

Suppose we want to predict whether a customer will churn.

We may have:

- Age
- Monthly bill
- Contract type
- Tenure
- Number of support calls
- Number of products

A simple linear model may assume that the effect of the features can be represented globally by a weighted sum. Real decision processes are often more like:

> If contract is month-to-month **and** support calls are high → high churn risk.

> Otherwise, if tenure is high **and** monthly bill is low → lower churn risk.

The underlying relationships can therefore be **non-linear** and **interaction-heavy**.

## 2.2 Limitations of Previous Approaches

A linear/logistic model can struggle when:

- the relationship between features and target is non-linear;
- interactions between variables are important;
- thresholds matter more than smooth changes;
- manually engineering interaction or polynomial features becomes expensive.

For example:

```text
Churn
 ^
 |       No      | Yes
 |               |
 |---------------|----------> Support Calls
                 5
```

A threshold such as `SupportCalls > 5` can be represented naturally by a tree.

## 2.3 What Should a Better Approach Do?

A useful model should ideally:

- capture non-linear relationships;
- model feature interactions automatically;
- handle mixed feature scales;
- be interpretable;
- require relatively little feature engineering;
- work for both classification and regression.

Decision Trees provide these properties, although they have their own limitations, especially high variance and overfitting.

## 2.4 Core Idea

> A Decision Tree repeatedly asks the question that creates the **purest useful child groups**.

At each node:

1. Consider candidate splits.
2. Measure how much each split improves the target homogeneity.
3. Choose the best split according to the criterion.
4. Repeat recursively on the child nodes.
5. Stop according to stopping rules or pruning.
6. Make predictions from the final leaf nodes.

The model therefore converts a complex prediction problem into a sequence of simple rules.

---

# 3. Formal Definition

## Definition

> A **Decision Tree** is a non-parametric supervised learning model that recursively partitions the feature space into regions using decision rules, with each internal node representing a split on a feature and each leaf representing a prediction.

For classification, a leaf generally predicts the majority class and can also provide class probabilities from class frequencies in that leaf.

For regression, a leaf generally predicts the mean target value of the training observations reaching that leaf.

The tree is built greedily: at each node, the algorithm selects the split that produces the largest reduction in impurity (or an equivalent improvement criterion).

## Algorithm Classification

| Property | Description |
|---|---|
| Learning Type | Supervised |
| Task | Classification and Regression |
| Parametricity | Non-parametric |
| Model Type | Hierarchical recursive partitioning |
| Learning Approach | Greedy recursive splitting |
| Core Structure | Tree of internal decision nodes and terminal leaf nodes |
| Optimization | Usually greedy/local at each node; not a global differentiable optimization |
| Main Risk | High variance / overfitting when allowed to grow too deep |

---

# 4. Intuition

## 4.1 Core Intuition

Imagine solving a problem by repeatedly asking yes/no questions.

Example:

> Will this customer churn?

```text
                 Contract = Month-to-month?
                    /               \
                  Yes                No
                  /                   \
       Support Calls > 5?          Predict No
             /      \
           Yes       No
           /          \
      Predict Yes   Predict No
```

The first question should be chosen because it makes the resulting groups more useful for prediction.

The algorithm does not directly search for the final tree in one step. It makes a locally optimal decision at the current node, then repeats the process for the resulting subsets.

## 4.2 Real-World Analogy

Consider a bank employee deciding whether a loan applicant is likely to default:

```text
Is income > ₹8 lakh?
        |
   +----+----+
  Yes       No
   |          |
Credit score? Existing debt?
  ...
```

The important idea is **sequential partitioning**:

- first separate the data using one rule;
- then make a more specific rule for each group;
- continue until the groups become sufficiently homogeneous.

This is an analogy, not the mathematical definition.

## 4.3 Simple Example

Suppose we have eight applicants:

| Age | Income | Buys |
|---:|---:|---|
| 22 | 25 | No |
| 25 | 30 | No |
| 28 | 35 | Yes |
| 30 | 40 | Yes |
| 32 | 45 | Yes |
| 35 | 50 | Yes |
| 42 | 55 | No |
| 45 | 60 | No |

The tree may discover a rule such as:

```text
Age <= 31?
   /     \
 Yes      No
  |        |
Predict    further split
Yes/No
```

The exact split depends on the impurity criterion and all candidate thresholds.

## 4.4 Mental Model

> **A Decision Tree learns a sequence of feature-based questions that recursively make the target values more homogeneous.**

---

# 5. Problem Formulation

## Given

Training dataset:

$$
D = \{(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)\}
$$

where:

- $x_i \in \mathbb{R}^d$ = feature vector for observation $i$;
- $y_i$ = target value for observation $i$;
- $n$ = number of training observations;
- $d$ = number of input features.

For classification:

$$
y_i \in \{1,2,\ldots,K\}
$$

For regression:

$$
y_i \in \mathbb{R}
$$

## Goal

Learn a tree $T$ that partitions the feature space into leaves:

$$
R_1,R_2,\ldots,R_M
$$

such that observations within the same region are sufficiently similar in terms of their target values.

For classification:

$$
\hat{y}(x)=\text{majority class in the leaf containing }x
$$

For regression:

$$
\hat{y}(x)=\frac{1}{|R_m|}\sum_{i:x_i\in R_m}y_i
$$

## Input

- Feature matrix $X$.
- Target vector $y$.
- Hyperparameters controlling tree growth and regularization.

## Output

A tree consisting of:

- root node;
- internal decision nodes;
- branches;
- terminal leaf nodes;
- a prediction rule associated with each leaf.

---

# 6. Mathematical Foundation

## 6.1 Model Representation

A tree can be viewed as a set of recursive partitions.

For a numeric feature $x_j$, a binary split has the form:

$$
x_j \le t
$$

versus

$$
x_j > t
$$

where $t$ is a candidate threshold.

For categorical data, a split can conceptually separate one or more categories into groups. The exact handling depends on the tree algorithm and implementation.

A complete tree maps every input $x$ to exactly one terminal leaf.

```text
                    Root
                      |
             x_j <= threshold?
                 /          \
               Yes           No
               /              \
            Node A          Node B
             /  \            /   \
           ...  ...         ...  ...
```

## 6.2 Objective

At a node containing a set of samples $S$, we want a split that produces child subsets with lower impurity.

For a candidate binary split:

$$
S_L=\{i\in S:x_{ij}\le t\}
$$

$$
S_R=\{i\in S:x_{ij}>t\}
$$

We choose the split minimizing the weighted child impurity:

$$
J(j,t)=
\frac{|S_L|}{|S|}I(S_L)
+
\frac{|S_R|}{|S|}I(S_R)
$$

Equivalent formulation:

$$
\text{Impurity Reduction}
=
I(S)-J(j,t)
$$

where:

- $I(S)$ = impurity of the parent;
- $I(S_L)$ = impurity of the left child;
- $I(S_R)$ = impurity of the right child.

## 6.3 Loss / Cost / Objective Function

### Classification — Gini Impurity

For a node containing $K$ classes:

$$
G(S)=1-\sum_{k=1}^{K}p_k^2
$$

where:

$$
p_k=\frac{\text{number of samples of class }k}{|S|}
$$

Interpretation:

- $G(S)=0$ means the node is perfectly pure.
- Higher Gini means the classes are more mixed.

For binary classification:

$$
G=1-(p^2+(1-p)^2)
$$

which simplifies to:

$$
G=2p(1-p)
$$

The maximum occurs at $p=0.5$.

### Classification — Entropy

Entropy measures uncertainty:

$$
H(S)=-\sum_{k=1}^{K}p_k\log_2 p_k
$$

For binary classification:

$$
H=-p\log_2 p-(1-p)\log_2(1-p)
$$

Interpretation:

- Entropy = 0 for a pure node.
- Entropy is maximum when class proportions are balanced.

### Information Gain

A split's information gain is:

$$
IG
=
H(S)
-
\left(
\frac{|S_L|}{|S|}H(S_L)
+
\frac{|S_R|}{|S|}H(S_R)
\right)
$$

Choose the split with maximum information gain.

### Regression — MSE / Variance Reduction

For regression, a common impurity measure is mean squared error:

$$
MSE(S)=\frac{1}{|S|}\sum_{i\in S}(y_i-\bar{y}_S)^2
$$

where:

$$
\bar{y}_S=\frac{1}{|S|}\sum_{i\in S}y_i
$$

For a candidate split:

$$
J(j,t)=
\frac{|S_L|}{|S|}MSE(S_L)
+
\frac{|S_R|}{|S|}MSE(S_R)
$$

The algorithm selects the split that minimizes $J(j,t)$, or equivalently maximizes variance/MSE reduction.

### 6.3.1 Classification Impurity Measures at a Glance

| Criterion | Formula | Pure Node | Main Interpretation |
|---|---|---:|---|
| Gini | $1-\sum p_k^2$ | 0 | Expected impurity under squared class probabilities |
| Entropy | $-\sum p_k\log_2 p_k$ | 0 | Uncertainty / information content |
| Misclassification Error | $1-\max_k p_k$ | 0 | Fraction not belonging to majority class |

Gini and entropy are generally more sensitive to changes in class probabilities than misclassification error, so they are more useful for tree splitting.

## 6.4 Why This Objective Function?

Suppose a parent node contains:

```text
Yes = 4
No  = 4
```

It is mixed.

A good split may create:

```text
Left child:  Yes = 4, No = 0
Right child: Yes = 0, No = 4
```

Both children are pure.

A bad split may create:

```text
Left child:  Yes = 3, No = 2
Right child: Yes = 1, No = 2
```

The first split is preferable because it reduces uncertainty much more.

The weighting by child size is critical. A tiny pure child should not automatically dominate the decision.

## 6.5 Optimization

Decision Tree training is typically **greedy**.

At the current node:

1. Enumerate candidate features.
2. Enumerate candidate thresholds or category partitions.
3. Compute the impurity of the resulting children.
4. Select the best split.
5. Recurse on each child.

The tree does **not** usually perform gradient descent over a smooth parameter space.

This means:

- optimization is local at each node;
- the final tree is not guaranteed to be the globally optimal tree among all possible tree structures;
- different stopping constraints can lead to different trees.

## 6.6 Derivation

### Derivation of Weighted Child Impurity

Suppose a parent has $N$ samples and a split creates:

- left child with $N_L$ samples;
- right child with $N_R$ samples.

Then:

$$
N=N_L+N_R
$$

The fraction of samples entering each child is:

$$
w_L=\frac{N_L}{N},
\qquad
w_R=\frac{N_R}{N}
$$

So the post-split impurity is:

$$
I_{\text{after}}
=
w_LI_L+w_RI_R
$$

To choose the best split, minimize:

$$
w_LI_L+w_RI_R
$$

Equivalently maximize:

$$
I_{\text{parent}}-I_{\text{after}}
$$

This gives the generic split-selection mechanism used across many tree algorithms.

### Why Is the Leaf Prediction the Majority Class?

For a leaf containing observations with class probabilities $p_1,\ldots,p_K$, the class probability estimate is the observed class proportion.

The class with largest observed proportion is the natural 0-1 loss minimizer:

$$
\hat{y}=\arg\max_k p_k
$$

### Why Is the Leaf Prediction the Mean in Regression?

For a leaf with targets $y_1,\ldots,y_m$, choose a constant prediction $c$ minimizing squared error:

$$
L(c)=\sum_{i=1}^{m}(y_i-c)^2
$$

Differentiate:

$$
\frac{dL}{dc}
=
-2\sum_{i=1}^{m}(y_i-c)
$$

Set derivative to zero:

$$
\sum_{i=1}^{m}(y_i-c)=0
$$

$$
mc=\sum_{i=1}^{m}y_i
$$

Therefore:

$$
\boxed{c=\frac{1}{m}\sum_{i=1}^{m}y_i}
$$

So the mean is the optimal constant prediction under squared loss.

## 6.7 Important Mathematical Properties

- **Convex / Non-convex:** The space of all possible tree structures is combinatorial and non-convex.
- **Differentiable / Non-differentiable:** Split selection is based on discrete feature/threshold choices rather than smooth differentiable optimization.
- **Closed-form / Iterative:** No single closed-form solution for the entire tree; the tree is built recursively.
- **Local vs Global Optimum:** Greedy split selection does not guarantee a globally optimal tree structure.
- **Parametric vs Non-parametric:** Non-parametric; complexity grows with the data and chosen tree structure.
- **Scale sensitivity:** The mathematical split ordering for a numeric feature is invariant to strictly monotonic transformations such as positive rescaling.
- **Prediction function:** Piecewise constant for standard Decision Tree Regression.

---

# 7. Geometric / Visual Understanding

## 7.1 What Does the Data Look Like?

Imagine two input features:

- $x_1$ = age
- $x_2$ = monthly income

A Decision Tree partitions the two-dimensional plane using axis-aligned rules such as:

$$
x_1\le 30
$$

or:

$$
x_2>50000
$$

This produces rectangular regions.

## 7.2 What Does the Model Learn?

A tree learns:

- **where to split**;
- **which feature to split on**;
- **the threshold**;
- **the final prediction associated with each leaf**.

For classification, each region corresponds to a class prediction or class probabilities.

For regression, each region corresponds to a constant predicted value.

## 7.3 Effect of Model Complexity

### Shallow Tree

```text
          x1 <= 30?
          /       \
       Class A   Class B
```

- few rules;
- low variance;
- potentially high bias;
- may underfit.

### Deep Tree

```text
                  Root
                 /    \
               ...    ...
              /         \
            ...         ...
           / \         / \
         leaf leaf   leaf leaf
```

- many rules;
- lower training error;
- high flexibility;
- potentially high variance;
- can memorize noise.

### Example of Axis-Aligned Partitioning

```text
x2
^
|       +-----------+
|       |     B     |
|  +----+-----------+
|  | A  |     B     |
|  |    +-----+-----+
|  | A  |  C  |  C  |
+--+----+-----+-----+----> x1
```

Each boundary corresponds to a tree threshold.

---

# 8. How the Algorithm Works

## Step-by-Step

### Step 1 — Start With the Entire Dataset

At the root node, all training examples are present.

Calculate the impurity of the current node.

For classification, this may be Gini or entropy.

For regression, this may be MSE/variance.

### Step 2 — Generate Candidate Splits

For each considered feature:

- sort or otherwise examine the feature values;
- identify possible thresholds.

For a feature with values:

```text
10, 20, 30, 40
```

candidate thresholds can be chosen between adjacent distinct values.

Example:

```text
x <= 15
x <= 25
x <= 35
```

The exact candidate-generation details depend on the implementation.

### Step 3 — Evaluate Every Candidate

For each candidate split:

1. Divide the data into left and right child nodes.
2. Compute the impurity of both children.
3. Compute weighted child impurity.
4. Measure impurity reduction.

Example:

$$
Gain=I_{parent}-I_{after}
$$

### Step 4 — Choose the Best Split

Select the feature and threshold producing the best valid objective according to the tree's criterion and constraints.

Example:

```text
Age <= 31
```

may be better than:

```text
Income <= 42,000
```

because it creates purer children.

### Step 5 — Recursively Split the Children

Apply the same procedure to each child.

```text
Root
 |
 +---- Left child
 |       |
 |      split again
 |
 +---- Right child
         |
        split again
```

This continues until a stopping rule is triggered.

### Step 6 — Stop or Prune

Possible stopping conditions include:

- maximum depth reached;
- too few samples for another split;
- child would contain too few samples;
- impurity improvement is too small;
- maximum number of leaves reached;
- post-pruning later removes weak subtrees.

### Step 7 — Assign Leaf Predictions

Classification:

```text
Leaf contains:
Yes = 8
No  = 2

Prediction = Yes
```

Regression:

```text
Leaf target values:
10, 12, 14, 16

Prediction = 13
```

## Algorithm Flow

```text
Input Data
    ↓
Root Node
    ↓
Measure Node Impurity
    ↓
Generate Candidate Splits
    ↓
Evaluate Candidate Splits
    ↓
Choose Best Split
    ↓
Create Child Nodes
    ↓
Repeat Recursively
    ↓
Stopping Condition?
   /        \
 No          Yes
 |            |
Split      Create Leaf
 Again        ↓
   \          |
    \---------/
         ↓
     Learned Tree
         ↓
      Prediction
```

## Pseudocode

```text
BuildTree(S):
    if stopping_condition(S):
        return Leaf(prediction(S))

    best_split = None
    best_score = worst_possible_score

    for each candidate feature j:
        for each candidate threshold t:
            partition S into S_left and S_right

            if split is invalid:
                continue

            score = weighted_impurity(S_left, S_right)

            if score is better than best_score:
                best_score = score
                best_split = (j, t)

    if no valid split exists:
        return Leaf(prediction(S))

    S_left, S_right = split(S, best_split)

    left_child = BuildTree(S_left)
    right_child = BuildTree(S_right)

    return Node(best_split, left_child, right_child)
```

---

# 9. Training vs Prediction

## 9.1 Training Phase

During `fit()`:

```text
X_train + y_train
        ↓
Create root node
        ↓
Search candidate splits
        ↓
Choose best split
        ↓
Recursively partition data
        ↓
Apply stopping rules
        ↓
Store leaf predictions
        ↓
Trained Decision Tree
```

The expensive part is training because the algorithm must search for useful split rules.

## 9.2 Prediction Phase

For a new observation:

```text
X_new
  ↓
Evaluate root rule
  ↓
Go left or right
  ↓
Evaluate next rule
  ↓
Continue until leaf
  ↓
Return leaf prediction
```

Prediction is therefore essentially a traversal from root to leaf.

## 9.3 What Does the Model Actually Learn?

A Decision Tree learns:

- feature index at each split;
- threshold at each split for numerical features;
- child relationships;
- depth/structure;
- sample counts associated with nodes;
- class counts/proportions for classification leaves;
- aggregate target statistics for regression leaves.

It does **not** primarily learn a weight vector such as:

$$
w_1,w_2,\ldots,w_d
$$

as Linear Regression does.

## 9.4 What Is Stored After Training?

Conceptually, the trained model stores:

```text
Node 0:
    feature = Age
    threshold = 31

Node 1:
    feature = SupportCalls
    threshold = 5

Node 2:
    leaf
    prediction = No
...
```

Scikit-Learn also stores node counts, impurity-related information, child indices, and tree structure internally.

---

# 10. Worked Example

## Dataset

Consider binary classification:

| Age | Target |
|---:|---|
| 18 | No |
| 22 | No |
| 25 | Yes |
| 28 | Yes |
| 30 | Yes |
| 32 | Yes |
| 40 | No |
| 45 | No |

There are 4 Yes and 4 No.

## Step 1 — Calculate Root Gini

At the root:

$$
p_{Yes}=\frac{4}{8}=0.5
$$

$$
p_{No}=\frac{4}{8}=0.5
$$

Therefore:

$$
G_{root}=1-(0.5^2+0.5^2)=0.5
$$

The root is highly mixed.

## Step 2 — Evaluate Candidate Split

Consider:

$$
Age\le 31
$$

Left child:

```text
18 No
22 No
25 Yes
28 Yes
30 Yes
```

Counts:

- Yes = 3
- No = 2
- Total = 5

Gini:

$$
G_L=1-\left(\frac35\right)^2-\left(\frac25\right)^2
$$

$$
=1-\frac{9}{25}-\frac{4}{25}
=\frac{12}{25}
=0.48
$$

Right child:

```text
32 Yes
40 No
45 No
```

Counts:

- Yes = 1
- No = 2

Gini:

$$
G_R=1-\left(\frac13\right)^2-\left(\frac23\right)^2
$$

$$
=1-\frac19-\frac49
=\frac49
\approx0.444
$$

Weighted child impurity:

$$
G_{after}
=
\frac58(0.48)+\frac38\left(\frac49\right)
$$

$$
=0.30+0.1667
\approx0.4667
$$

Gini reduction:

$$
0.5-0.4667\approx0.0333
$$

This is a modest improvement.

## Step 3 — Why Do We Need to Test Other Thresholds?

A tree cannot assume that `Age <= 31` is best.

It must compare other candidate thresholds, such as:

```text
Age <= 20
Age <= 23.5
Age <= 26.5
Age <= 29
Age <= 31
Age <= 36
Age <= 42.5
```

The best threshold is whichever gives the lowest weighted child impurity, subject to the tree's constraints.

## Step 4 — Recursive Splitting

Once the best root split is selected:

```text
                 Age <= t?
                /         \
             Left        Right
              /             \
        find best split   find best split
             / \             / \
           ... ...         ... ...
```

The left and right datasets are treated independently.

## Step 5 — Final Leaves

Eventually, a leaf might contain:

```text
Yes = 5
No = 0
```

so classification predicts:

```text
Yes
```

Another leaf might contain:

```text
Yes = 1
No = 4
```

so prediction becomes:

```text
No
```

## Final Result

The key idea is not memorizing a particular threshold from this example.

The important sequence is:

```text
Measure impurity
      ↓
Try candidate splits
      ↓
Choose best split
      ↓
Repeat recursively
      ↓
Stop / prune
      ↓
Predict from leaf
```

> Goal: I should be able to manually calculate impurity and compare at least two candidate splits on a tiny dataset.

---

# 11. Important Concepts & Terminology

## 11.1 Root Node

**Definition:** The first node containing the entire training dataset.

**Why it matters:** Every prediction begins at the root.

## 11.2 Internal / Decision Node

**Definition:** A non-terminal node that applies a decision rule to split observations.

**Why it matters:** It defines how the feature space is partitioned.

## 11.3 Leaf / Terminal Node

**Definition:** A node at which the tree stops splitting and produces a prediction.

**Why it matters:** The prediction for a new observation comes from the leaf it reaches.

## 11.4 Branch

**Definition:** The connection between nodes representing the outcome of a decision rule.

**Why it matters:** A branch determines the path an observation follows.

## 11.5 Split

**Definition:** A rule that partitions the observations in a node into child subsets.

Example:

$$
Age\le30
$$

**Why it matters:** Splits are the mechanism through which the tree learns.

## 11.6 Threshold

**Definition:** A cutoff value used in a numeric feature split.

Example:

$$
Income\le50000
$$

**Why it matters:** Thresholds determine the exact boundary of a region.

## 11.7 Impurity

**Definition:** A measure of how mixed the target values are within a node.

**Why it matters:** The tree uses impurity reduction to select useful splits.

## 11.8 Entropy

**Definition:**

$$
H=-\sum p_k\log_2 p_k
$$

**Why it matters:** Measures uncertainty and is used in information-gain-based splitting.

## 11.9 Gini Impurity

**Definition:**

$$
G=1-\sum p_k^2
$$

**Why it matters:** A commonly used classification splitting criterion.

## 11.10 Information Gain

**Definition:** Reduction in entropy produced by a split.

$$
IG=H(parent)-H(after)
$$

**Why it matters:** Larger information gain means the split decreases uncertainty more.

## 11.11 Purity

**Definition:** Degree to which observations in a node belong to one class.

**Why it matters:** Trees generally seek purer child nodes.

## 11.12 Recursive Partitioning

**Definition:** Repeating the split-selection process independently within each child node.

**Why it matters:** This creates the hierarchical tree structure.

## 11.13 Pruning

**Definition:** Removing unnecessary branches or limiting tree growth to improve generalization.

**Why it matters:** Uncontrolled trees can memorize noise.

## 11.14 CART

**Definition:** Classification and Regression Trees is a family/framework that commonly builds binary recursive partitions and supports both classification and regression.

**Why it matters:** Many practical implementations, including the approach used by Scikit-Learn's decision tree estimators, are based on this style of recursive binary partitioning.

---

# 12. Parameters vs Hyperparameters

## Parameters

**Definition:** Quantities learned from the training data as part of the fitted tree.

### Examples

- Feature selected at each internal node.
- Threshold selected at each split.
- Class counts/proportions in classification leaves.
- Mean target value in a regression leaf.

## Hyperparameters

**Definition:** Settings chosen before or around training that control how the model is built.

### Examples

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `criterion`
- `max_leaf_nodes`
- `ccp_alpha`

## Key Difference

| Parameters | Hyperparameters |
|---|---|
| Learned from data | Chosen by the practitioner / search procedure |
| Define the fitted tree | Control how the tree is grown |
| Feature + threshold at a node | `max_depth`, `min_samples_leaf`, etc. |
| Leaf statistics | `criterion`, pruning settings, etc. |

---

# 13. Important Hyperparameters

| Hyperparameter | Meaning | Increase → | Decrease → | Main Effect |
|---|---|---|---|---|
| `criterion` | Impurity/loss used for splitting | Changes split preference | Changes split preference | Controls split evaluation |
| `max_depth` | Maximum tree depth | More complex, lower training error | Simpler tree | Strong control of overfitting |
| `min_samples_split` | Minimum samples required to split an internal node | More conservative | Easier splitting | Regularization |
| `min_samples_leaf` | Minimum samples required in a leaf | Larger leaves, smoother model | Smaller leaves, more flexible | Strong regularization |
| `max_leaf_nodes` | Maximum number of leaves | More expressive tree | Simpler tree | Controls size |
| `max_features` | Features considered when looking for a split | More candidate features | More randomness/less computation | Can reduce correlation/variance |
| `min_impurity_decrease` | Minimum required impurity improvement | Fewer splits | More splits | Regularization |
| `class_weight` | Class weighting for classification | More weight to selected classes | Less weighting | Helps class imbalance |
| `ccp_alpha` | Cost-complexity pruning strength | More pruning | Less pruning | Post-pruning |
| `random_state` | Controls randomness where applicable | Not a complexity control | Not a complexity control | Reproducibility |

## Most Important Hyperparameters

For placements, focus especially on:

1. `max_depth`
2. `min_samples_split`
3. `min_samples_leaf`
4. `criterion`
5. `ccp_alpha`

### `max_depth`

This is one of the most important controls.

Small depth:

- fewer rules;
- simpler model;
- higher bias;
- lower variance;
- risk of underfitting.

Large depth:

- more rules;
- lower training error;
- lower bias;
- higher variance;
- risk of overfitting.

### `min_samples_split`

A node can split only if it contains at least this many samples.

Increase it:

- prevents very small nodes from splitting;
- reduces tree complexity;
- can improve generalization.

Decrease it:

- allows more splits;
- produces a more flexible tree;
- increases overfitting risk.

### `min_samples_leaf`

This is often especially useful for controlling noisy leaves.

Increase it:

```text
larger minimum leaf
        ↓
fewer tiny regions
        ↓
smoother / less variable tree
```

Decrease it:

```text
smaller minimum leaf
        ↓
more specific rules
        ↓
higher variance
```

### `criterion`

For classification, common choices include:

- Gini impurity;
- entropy/information gain.

For regression, a common choice is:

- squared error / variance reduction.

Usually the difference between reasonable criteria is smaller than the effect of controlling tree complexity.

### `ccp_alpha`

Cost-complexity pruning parameter.

Larger `ccp_alpha`:

```text
stronger penalty on complex trees
        ↓
more branches removed
        ↓
simpler tree
```

## Hyperparameter Interactions

These parameters should not be considered independently.

For example:

```text
Large max_depth
+
Small min_samples_leaf
        ↓
Very flexible tree
        ↓
High overfitting risk
```

Whereas:

```text
Moderate max_depth
+
Larger min_samples_leaf
        ↓
Constrained tree
        ↓
Better generalization in many datasets
```

Pruning with `ccp_alpha` can also simplify a tree that was initially grown more deeply.

---

# 14. Bias-Variance & Generalization

## Bias

A shallow or heavily regularized tree may be unable to capture important patterns.

High bias means:

- model is too simple;
- training error is already high;
- important structure is missed.

## Variance

A deep Decision Tree can be highly sensitive to small changes in training data.

High variance means:

- training performance can be extremely good;
- validation/test performance may fluctuate;
- tree structure can change significantly when data changes.

## Overfitting

### Why Can This Algorithm Overfit?

Decision Trees can keep splitting until very small groups are created.

At extreme depth, the model may learn:

```text
Sample-specific rules
        ↓
memorization of training noise
        ↓
very low training error
        ↓
poor generalization
```

A fully grown tree can effectively memorize the training set.

### Signs of Overfitting

- Training accuracy very high but validation accuracy much lower.
- Training MSE very low but validation MSE much higher.
- Large tree depth.
- Many leaves with very few observations.
- Performance changes significantly across validation folds.

## Underfitting

### Why Can This Algorithm Underfit?

The tree may be too restricted because:

- `max_depth` is too small;
- `min_samples_leaf` is too large;
- `min_samples_split` is too large;
- `min_impurity_decrease` is too high;
- `ccp_alpha` is too aggressive.

### Signs of Underfitting

- Training and validation performance are both poor.
- Important structure is not captured.
- Tree contains too few rules.

## Bias-Variance Trade-off

```text
Model Complexity
       ↓
Underfitting → Good Generalization → Overfitting
     ↓                 ↓                  ↓
  High Bias         Balanced          High Variance
```

For Decision Trees, increasing depth generally moves the model toward lower bias and higher variance.

## Controlling Overfitting

- Reduce `max_depth`.
- Increase `min_samples_leaf`.
- Increase `min_samples_split`.
- Limit `max_leaf_nodes`.
- Increase `min_impurity_decrease`.
- Use cost-complexity pruning with `ccp_alpha`.
- Use cross-validation to choose hyperparameters.
- Use an ensemble such as Random Forest when a single tree's variance is too high.

## Controlling Underfitting

- Increase `max_depth`.
- Reduce `min_samples_leaf`.
- Reduce `min_samples_split`.
- Reduce excessive pruning.
- Allow more leaves.
- Reassess whether informative features are present.

---

# 15. Assumptions

| Assumption / Property | Required? | Why? | What If Violated? |
|---|---|---|---|
| Relationship must be linear | No | Trees model non-linear rules | No issue |
| Feature scaling | No | Splits depend on ordering/thresholds | Scaling usually does not improve tree learning |
| Normal distribution | No | Trees are non-parametric | No issue from non-normality alone |
| Independent features | No | Trees can handle correlated features | Correlated features can make split selection unstable/redundant |
| Constant variance | No | Not required like in classical linear regression assumptions | No direct violation |
| Numerical representation understood by implementation | Practical requirement | Splits must operate on usable feature values | Encode categorical/string values appropriately |
| Target labels are available for training | Yes | Supervised learning requires targets | Cannot perform standard supervised tree training |
| Sufficient representative data | Practical | Generalization still depends on data quality | Tree can overfit spurious patterns |

## Important Interview Distinction

Decision Trees have **few strict statistical distributional assumptions**.

Do not say:

> "Decision Trees assume normally distributed data."

That is incorrect.

Do say:

> "Decision Trees are non-parametric and do not require the linearity, normality, or feature-scaling assumptions common in many parametric models."

Practical data-quality requirements still matter. A model cannot infer useful patterns from completely uninformative features or severely unrepresentative data.

---

# 16. Data Preprocessing

## 16.1 Missing Values

A standard Decision Tree implementation may have restrictions on missing values.

Scikit-Learn's tree estimators have version/model-specific behavior around missing values, so a safe general workflow is:

- inspect missing values;
- use an appropriate imputation strategy in a preprocessing pipeline;
- verify the estimator's supported missing-value behavior in the installed version.

Do not assume that every Decision Tree implementation automatically handles every type of missing value.

## 16.2 Feature Scaling

**Required?** No.

**Why?**

For a numeric feature, the split depends on ordering.

Suppose:

```text
Age = 20, 30, 40
```

and a threshold is:

```text
Age <= 30
```

Multiplying the feature by 100 changes:

```text
20, 30, 40
```

to:

```text
2000, 3000, 4000
```

The ordering is unchanged, so the equivalent split is:

```text
Age <= 3000
```

Therefore standardization:

$$
z=\frac{x-\mu}{\sigma}
$$

is generally unnecessary for tree splitting.

### Interview Answer

> Decision Trees do not require feature scaling because split decisions depend on feature order and thresholds, not on Euclidean distance or gradient magnitudes.

## 16.3 Categorical Variables

Tree algorithms need a representation they can split on.

Common approaches:

- one-hot encoding for nominal categories;
- ordinal encoding only when the representation/order is logically appropriate;
- native categorical handling in algorithms that explicitly support it.

Important:

> Arbitrary integer encoding of nominal categories can introduce a fake ordering.

For example:

```text
Red = 0
Blue = 1
Green = 2
```

may create artificial relationships if treated as numeric.

## 16.4 Outliers

Decision Trees are generally less sensitive to outliers than distance-based or mean-based linear methods.

Why?

A tree often needs only an ordering and threshold.

However, outliers can still affect:

- candidate thresholds;
- node sample distributions;
- regression leaf means;
- tree structure.

For regression, extreme target values can influence the mean prediction strongly.

## 16.5 Multicollinearity

Decision Trees do not require low multicollinearity.

If two features contain similar information:

```text
Age
YearsSinceBirth
```

the tree may choose one first and ignore the other.

However, strong feature correlation can make the chosen split and feature importance less stable.

## 16.6 Feature Engineering

Trees often need less feature engineering than linear models.

Still useful:

- domain-specific aggregations;
- meaningful ratios;
- time-based features;
- count/frequency features;
- handling leakage;
- meaningful interaction variables when required by the problem.

A tree can discover many interactions automatically, so adding many arbitrary polynomial features may increase complexity without adding useful information.

---

# 17. Model Complexity

## What Controls Complexity?

Main controls:

- depth;
- number of leaves;
- minimum samples per split;
- minimum samples per leaf;
- minimum impurity decrease;
- pruning strength.

## Simple Model

A shallow tree:

```text
       Age <= 30?
       /        \
    Class A    Class B
```

Characteristics:

- easy to interpret;
- low variance;
- potentially high bias.

## Complex Model

A deep tree:

```text
Age <= 30?
├── Yes
│   └── Income <= 40k?
│       ├── Yes
│       └── No
└── No
    └── Calls <= 5?
        ├── Yes
        └── No
```

Characteristics:

- more detailed decision rules;
- lower training error;
- greater sensitivity to noise;
- higher overfitting risk.

## Effect on Generalization

Typical pattern:

```text
Too shallow
   ↓
high bias

      ↓ increase complexity

Sweet spot
   ↓
good validation performance

      ↓ continue increasing complexity

Too deep
   ↓
high variance / overfitting
```

The optimal complexity must be selected using validation data or cross-validation, not by training accuracy alone.

---

# 18. Computational Complexity

## Training Complexity

A useful high-level description is:

$$
O(n\,d\,\log n)
$$

for favorable/efficient implementations under common assumptions, where:

- $n$ = number of samples;
- $d$ = number of features.

However, implementation details matter significantly.

A naive approach that repeatedly scans many unsorted candidates can be substantially more expensive, and worst-case behavior can approach quadratic dependence on the number of samples.

For interview purposes:

> Tree training is more expensive than tree prediction because training must search over many candidate splits.

## Why?

At each node, the algorithm may need to examine:

```text
many features
      ×
many candidate thresholds
      ×
multiple nodes
```

As the tree grows, the node datasets become smaller.

## Prediction Complexity

Prediction requires traversal from root to leaf:

$$
O(h)
$$

where $h$ is the depth of the tree.

For a balanced tree:

$$
h\approx O(\log n)
$$

For a highly unbalanced tree:

$$
h\approx O(n)
$$

Worst-case prediction can therefore be linear in the number of samples used to create the tree.

## Space Complexity

The tree requires space proportional to the number of nodes:

$$
O(M)
$$

where $M$ is the number of tree nodes.

In an extreme tree, $M$ can scale roughly with the number of training observations.

## Scalability

### More Samples

- More samples can improve statistical reliability.
- Training cost increases.
- A very deep tree may become large.

### More Features

- More candidate features can increase training cost.
- Irrelevant features may increase split-search work.
- `max_features` can reduce the feature-search burden.

### High-Dimensional Data

Trees can still work in high-dimensional settings, but:

- split search becomes more expensive;
- irrelevant/noisy dimensions can create unstable trees;
- one-hot encoding may greatly increase feature count;
- ensembles often perform better than a single tree when variance is a major issue.

---

# 19. Regularization / Optimization Improvements

## Why Is It Needed?

A completely grown tree can fit the training set extremely well but generalize poorly.

Regularization prevents the tree from becoming unnecessarily complex.

## Method 1 — Pre-Pruning

Pre-pruning stops tree growth early using controls such as:

- `max_depth`;
- `min_samples_split`;
- `min_samples_leaf`;
- `max_leaf_nodes`;
- `min_impurity_decrease`.

Example:

```text
Unlimited growth
      ↓
many tiny leaves
      ↓
overfitting

Constrained growth
      ↓
fewer, larger leaves
      ↓
better generalization potential
```

## Method 2 — Cost-Complexity Pruning

Define a subtree objective:

$$
R_\alpha(T)=R(T)+\alpha|L(T)|
$$

where:

- $R(T)$ = empirical tree error/impurity measure;
- $|L(T)|$ = number of leaves;
- $\alpha$ = complexity penalty.

As $\alpha$ increases:

```text
penalty for extra leaves increases
        ↓
larger branches become less worthwhile
        ↓
tree gets pruned
```

This is post-pruning because a larger tree can first be grown and then simplified.

Scikit-Learn exposes this through:

```python
ccp_alpha
```

## Effect on Model

```text
Regularization ↑
     ↓
Tree complexity ↓
     ↓
Variance ↓
Bias may ↑
     ↓
Potentially better validation performance
```

Regularization is useful only to the point where it stops harmful complexity; excessive regularization causes underfitting.

---

# 20. Evaluation

## Relevant Metrics

### Classification

- Accuracy — useful when classes are reasonably balanced and error costs are similar.
- Precision — important when false positives are costly.
- Recall — important when false negatives are costly.
- F1 Score — balances precision and recall.
- ROC-AUC — measures ranking performance across classification thresholds.
- PR-AUC — often more informative for strongly imbalanced positive classes.
- Confusion Matrix — directly shows TP, TN, FP, FN.

### Regression

- MAE — robust and interpretable in target units.
- MSE — penalizes large errors more strongly.
- RMSE — same units as the target while emphasizing larger errors.
- $R^2$ — relative goodness-of-fit measure.

## Which Metrics Should I Use?

### Balanced classification

Accuracy can be reasonable.

### Imbalanced classification

Do not rely on accuracy alone.

Use:

- precision;
- recall;
- F1;
- PR-AUC;
- confusion matrix.

Choose based on the business cost of false positives and false negatives.

### Regression

Use the metric that matches the business loss.

For example:

- MAE when every unit of absolute error should matter roughly equally;
- RMSE when large mistakes should be penalized more strongly.

## Cross-Validation

Cross-validation is useful for:

- comparing hyperparameter settings;
- estimating generalization more reliably;
- reducing dependence on one train/validation split.

For classification, **stratified cross-validation** is often appropriate so that class proportions remain approximately stable across folds.

A common tuning workflow:

```text
Training data
     ↓
Cross-validation
     ↓
Try different tree hyperparameters
     ↓
Select configuration with best validation metric
     ↓
Refit on full training data
     ↓
Evaluate once on untouched test data
```

---

# 21. Advantages

## 1. High Interpretability

**Why?**

The model can be represented as human-readable rules:

```text
IF income > 50k
AND support_calls > 4
THEN churn = Yes
```

This is often valuable when explaining individual decisions.

## 2. Captures Non-Linear Relationships and Interactions

**Why?**

The tree can use different rules in different regions of the feature space.

Example:

```text
If age <= 30:
    use income
else:
    use contract type
```

A feature can therefore have different effects in different branches.

## 3. Little Need for Feature Scaling

**Why?**

Tree splits depend on thresholds/order rather than distances or gradient magnitudes.

---

# 22. Disadvantages

## 1. High Variance

**Why?**

Small changes in training data can change:

- the first split;
- downstream branches;
- the entire tree structure.

This instability is a major reason ensemble methods such as Random Forest are popular.

## 2. Overfitting

**Why?**

A tree can keep creating smaller regions until it models noise.

Unrestricted depth and tiny leaves are especially risky.

## 3. Axis-Aligned Partitioning Can Be Inefficient

**Why?**

Standard trees make splits such as:

$$
x_j\le t
$$

They do not directly learn oblique boundaries like:

$$
2x_1+3x_2>10
$$

As a result, some diagonal patterns may require many rectangular regions and therefore many splits.

---

# 23. Failure Modes

## When Does It Perform Poorly?

- Extremely noisy data.
- Very small datasets with unstable patterns.
- Problems requiring smooth extrapolation.
- Highly oblique decision boundaries.
- Regression tasks where leaf means cannot capture needed smooth trends.
- Data with significant measurement or label noise when the tree is not regularized.

## Why Does It Fail?

The model is highly flexible and greedy.

It may:

- choose a locally attractive but globally suboptimal split;
- chase noise;
- produce unstable boundaries;
- create many small regions;
- extrapolate poorly in regression.

## Warning Signs

- Huge gap between train and validation performance.
- Very deep tree.
- Very large number of leaves.
- Leaves with tiny sample counts.
- Unstable cross-validation scores.
- Validation performance deteriorates as depth increases.

## How Can We Detect the Problem?

Use:

- learning curves;
- train vs validation metrics;
- cross-validation;
- tree visualization;
- leaf sample counts;
- feature-importance sanity checks.

## Possible Solutions

- Pre-prune.
- Post-prune.
- Use cross-validation to tune complexity.
- Improve data quality.
- Remove leakage.
- Use an ensemble such as Random Forest or Gradient Boosting when a single tree is too unstable.

---

# 24. Algorithm-Specific Edge Cases

## Case 1 — All Targets in a Node Are Identical

Classification:

```text
Yes
Yes
Yes
Yes
```

Impurity is zero.

There is no prediction advantage in splitting further.

Regression:

```text
10, 10, 10, 10
```

MSE is zero.

The node is already perfectly homogeneous.

## Case 2 — A Feature Has No Useful Split

Suppose every candidate threshold produces almost the same child impurity as the parent.

Then:

- impurity improvement may be zero or below the required threshold;
- the node should become a leaf.

## Case 3 — Duplicate or Repeated Feature Values

When many values are identical:

- there may be fewer distinct candidate thresholds;
- some candidate splits are impossible or redundant.

## Case 4 — Extremely Imbalanced Classes

Example:

```text
99,000 negatives
1,000 positives
```

A tree can achieve very high accuracy while ignoring the minority class.

Use:

- class-aware metrics;
- `class_weight` where appropriate;
- careful validation.

## Case 5 — Single-Sample Leaves

A fully grown tree may create leaves containing one observation.

Training error may approach zero, especially for classification with separable training data.

This is a classic overfitting pattern.

## Case 6 — Regression Outliers

A leaf prediction based on the mean can be strongly influenced by extreme targets.

For example:

```text
10, 11, 12, 1000
```

Mean is:

$$
258.25
$$

which may be a poor representation of the typical observations.

This is one reason alternative regression criteria or metrics may matter.

## Case 7 — Constant Feature

If a feature has the same value for every observation:

```text
5, 5, 5, 5, 5
```

it cannot create a meaningful threshold-based partition.

## Case 8 — High-Cardinality Categorical Feature

A feature with many unique categories can create very specific splits and potentially encourage overfitting, depending on the encoding and tree algorithm.

Consider:

- target leakage;
- cardinality;
- encoding strategy;
- minimum leaf size.

## Case 9 — Numerical Precision

Candidate thresholds on floating-point data can be affected by finite numerical precision.

In practice, use stable libraries and avoid unnecessary numerical transformations.

## Case 10 — Special Hyperparameter Values

Examples:

- `max_depth=None` may permit growth until another stopping condition is met.
- `min_samples_leaf=1` permits singleton leaves.
- Large `min_samples_leaf` can prevent useful local splits.
- `ccp_alpha=0` means no cost-complexity penalty from that parameter.

Exact defaults and supported values can depend on the Scikit-Learn version.

---

# 25. Practical Implementation — Scikit-Learn

## Import

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```

For regression:

```python
from sklearn.tree import DecisionTreeRegressor
```

## Create Model

```python
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
```

For regression:

```python
model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
```

## Train

```python
model.fit(X_train, y_train)
```

## Predict

```python
y_pred = model.predict(X_test)
```

For classification probabilities:

```python
y_proba = model.predict_proba(X_test)
```

## Evaluate

```python
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(classification_report(y_test, y_pred))
```

For regression:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mae, mse, r2)
```

## Visualize the Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=True,
    filled=True
)
plt.show()
```

The visualization is useful for learning:

- which feature is selected first;
- what threshold is used;
- how many samples reach each node;
- how the tree branches.

Do not interpret a single visualization as proof that the tree is well-generalized; validation performance still matters.

---

# 26. From-Scratch Implementation

> Implement the core algorithm without using the model provided by Scikit-Learn.

The following simplified classifier uses Gini impurity and binary numeric splits.

```python
import numpy as np


class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifierScratch:
    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0 or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        return 1.0 - np.sum(probabilities ** 2)

    def _split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        return X_left, y_left, X_right, y_right

    def _weighted_gini(self, y_left, y_right):
        n = len(y_left) + len(y_right)

        if len(y_left) == 0 or len(y_right) == 0:
            return np.inf

        weight_left = len(y_left) / n
        weight_right = len(y_right) / n

        return (
            weight_left * self._gini(y_left)
            + weight_right * self._gini(y_right)
        )

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _best_split(self, X, y):
        n_samples, n_features = X.shape

        best_feature = None
        best_threshold = None
        best_score = np.inf

        for feature_index in range(n_features):
            values = np.unique(X[:, feature_index])

            if len(values) <= 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split(
                    X,
                    y,
                    feature_index,
                    threshold
                )

                if len(y_left) < self.min_samples_leaf:
                    continue

                if len(y_right) < self.min_samples_leaf:
                    continue

                score = self._weighted_gini(y_left, y_right)

                if score < best_score:
                    best_score = score
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        # Stop if node is pure.
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        # Stop if depth limit is reached.
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._majority_class(y))

        # Stop if there are too few samples to split.
        if len(y) < self.min_samples_split:
            return Node(value=self._majority_class(y))

        feature_index, threshold = self._best_split(X, y)

        # No valid split found.
        if feature_index is None:
            return Node(value=self._majority_class(y))

        X_left, y_left, X_right, y_right = self._split(
            X, y, feature_index, threshold
        )

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")

        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)

        return self._predict_one(x, node.right)

    def predict(self, X):
        if self.root is None:
            raise ValueError("Model has not been fitted")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        return np.array([
            self._predict_one(x, self.root)
            for x in X
        ])
```

## Code ↔ Mathematics

| Code Component | Mathematical / Algorithmic Concept |
|---|---|
| `_gini()` | $1-\sum p_k^2$ |
| `thresholds` | Candidate split points |
| `_weighted_gini()` | Weighted child impurity |
| `_best_split()` | Greedy split selection |
| `_build_tree()` | Recursive partitioning |
| `min_samples_leaf` | Regularization / minimum child size |
| `max_depth` | Complexity control |
| `_majority_class()` | Leaf classification rule |
| `_predict_one()` | Root-to-leaf traversal |

## Important Implementation Details

The simplified implementation above is for learning, not production use.

Important improvements would include:

- efficient sorting/caching rather than repeatedly scanning all thresholds;
- support for sample weights;
- probability estimates;
- class weights;
- missing-value handling;
- pruning;
- categorical features;
- more split criteria;
- iterative node handling to avoid deep Python recursion issues;
- better memory usage;
- optimized low-level operations.

The important learning goal is the algorithmic correspondence:

```text
impurity formula
      ↓
candidate split search
      ↓
best split
      ↓
recursive node creation
      ↓
leaf prediction
```

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
Scaling (usually NOT required)
    ↓
Baseline Model
    ↓
Hyperparameter Tuning
    ↓
Cross-Validation
    ↓
Final Evaluation
    ↓
Interpretation
```

## Algorithm-Specific Considerations

Before training:

- check target imbalance;
- inspect missing values;
- inspect leakage;
- encode categorical variables appropriately;
- do not add scaling by default just because other ML algorithms use it.

During tuning:

- monitor tree depth;
- monitor number of leaves;
- inspect train-vs-validation performance;
- tune `min_samples_leaf`;
- consider pruning.

After training:

- inspect feature importance carefully;
- visualize the tree when it is small enough;
- verify that the performance metric matches the business goal;
- validate on untouched test data.

---

# 28. Comparison With Related Algorithms

## `Decision Tree` vs `Logistic Regression`

| Aspect | Decision Tree | Logistic Regression |
|---|---|---|
| Core Idea | Recursive feature splits | Weighted linear combination + sigmoid |
| Assumptions | Few distributional assumptions | Linear relationship in log-odds |
| Bias | Can be low with deep tree | Higher when boundary is non-linear |
| Variance | High for deep trees | Typically lower |
| Scaling | Usually not required | Often important depending on optimization/data |
| Interpretability | Rule-based | Feature coefficients / odds ratios |
| Training Speed | Moderate | Usually fast |
| Prediction Speed | Fast traversal | Very fast matrix operation |
| Overfitting | Can be severe | Can also overfit with feature complexity, but regularization is explicit |
| Strength | Non-linear rules/interactions | Stable linear boundary and probability model |
| Weakness | High variance, axis-aligned splits | Limited by linear decision boundary |
| Typical Use Case | Rule-like non-linear relationships | Linear/semi-linear classification |

## `Decision Tree` vs `Random Forest`

| Aspect | Decision Tree | Random Forest |
|---|---|---|
| Core Idea | One tree | Many trees combined |
| Variance | High | Lower in many settings |
| Interpretability | High | Lower |
| Overfitting | More likely | Usually more controlled |
| Training | Faster for one tree | More expensive overall |
| Prediction | One path | Many paths across trees |
| Stability | Sensitive to data changes | More stable |
| Feature Importance | Can be unstable | More robust but still not causal |
| Main Strength | Simple, explainable rules | Stronger generalization through averaging |
| Main Weakness | High variance | Less interpretable and larger |

## `Decision Tree` vs `k-NN`

| Aspect | Decision Tree | k-NN |
|---|---|---|
| Core Idea | Learn recursive rules | Predict from nearby training points |
| Scaling | Usually not required | Important |
| Training | Relatively expensive split search | Very little fitting |
| Prediction | Fast | Can be expensive |
| Geometry | Axis-aligned partitions | Distance-based neighborhoods |
| Interpretability | High | Moderate/low |
| Main Risk | Overfitting through depth | Sensitive to scaling/distance and irrelevant features |

## Key Distinction

> **Decision Tree = learn a hierarchy of if/else rules.**

> **Random Forest = average many diversified Decision Trees to reduce variance.**

---

# 29. When Should I Use This Algorithm?

Use it when:

- **Interpretability matters** — the model can be expressed as rules.
- **Relationships are non-linear or interaction-heavy** — trees discover conditional structure automatically.
- **Feature scaling is inconvenient** — scaling is generally unnecessary.
- **You need a strong baseline quickly** — one tree can be trained and inspected quickly.
- **You want a building block for ensembles** — Decision Trees are the base learners in Random Forest and many boosting methods.

---

# 30. When Should I Avoid This Algorithm?

Consider another approach when:

- **You need smooth extrapolation in regression** — tree regression is piecewise constant and does not extrapolate smoothly beyond observed patterns.
- **The true structure is strongly diagonal/oblique** — standard axis-aligned splits may need many branches.
- **You need very stable predictions from small perturbations of the data** — a single tree can have high variance.
- **The tree becomes extremely large** — the model may be difficult to explain and likely overfit.
- **A simpler linear model is already appropriate** — a tree may add unnecessary complexity.

---

# 31. Algorithm Selection Guide

When facing a new ML problem:

```text
What type of problem?
        ↓
Classification / Regression
        ↓
Do I need a highly interpretable rule-based model?
        |
      Yes
        ↓
Try Decision Tree
        |
       No
        ↓
Is the relationship approximately linear?
        |
      Yes -------------------- No
       |                         |
Linear / Logistic          Tree / k-NN /
Regression                ensemble methods
       |                         |
       +-----------+-------------+
                   ↓
What constraints matter?
                   ↓
Interpretability / Scale / Speed / Noise / Data Size
                   ↓
Benchmark sensible candidates
                   ↓
Use cross-validation
                   ↓
Choose based on the evaluation metric
```

For practical ML:

> Do not choose a model solely from theory. Start with a baseline, validate several reasonable candidates, and select using the problem's metric and constraints.

---

# 32. Common Misconceptions

## Misconception 1

> "Decision Trees always require feature scaling."

**Correction:** No. Standard Decision Trees are generally insensitive to feature scale because split decisions depend on feature ordering and thresholds rather than distances.

## Misconception 2

> "A deeper tree is always better because it reduces training error."

**Correction:** Lower training error does not imply better generalization. Deep trees can memorize noise and have high variance.

## Misconception 3

> "Decision Trees are linear models because they make simple equations at every node."

**Correction:** A tree is a non-linear, piecewise rule-based model. Its overall decision boundary can be highly non-linear even though each individual split is simple.

## Misconception 4

> "A Decision Tree has zero assumptions."

**Correction:** It has far fewer statistical distributional assumptions than many parametric models, but it still depends on informative data, appropriate representation, and suitable target/features.

## Misconception 5

> "Gini impurity and entropy give completely different kinds of trees."

**Correction:** They use different impurity formulas and can select different splits, but both aim to create purer child nodes. Performance differences are often dataset-dependent.

## Misconception 6

> "Feature importance from a tree proves that a feature causes the target."

**Correction:** Feature importance indicates predictive usefulness under the model's split process. It does not establish causation.

---

# 33. Common Implementation Mistakes

## Mistake 1 — Looking Only at Training Accuracy

A tree can achieve nearly perfect training performance while generalizing poorly.

**Fix:** Always inspect validation/test metrics and preferably cross-validation.

## Mistake 2 — Scaling Features Unnecessarily

Adding standardization is not usually harmful in a correctly designed pipeline, but it adds complexity without being needed for standard tree splits.

**Fix:** Know why preprocessing is required instead of applying the same preprocessing to every algorithm.

## Mistake 3 — Using Accuracy on a Highly Imbalanced Dataset

Example:

```text
99% negative
1% positive
```

A model predicting all negatives gets:

```text
99% accuracy
```

while being useless for positive-class detection.

**Fix:** Evaluate precision, recall, F1, PR-AUC, and the confusion matrix as appropriate.

## Mistake 4 — Allowing Unlimited Growth Without Validation

**Fix:** Tune complexity controls such as:

```python
max_depth
min_samples_leaf
min_samples_split
ccp_alpha
```

## Mistake 5 — Treating Integer-Encoded Categories as Ordered Without Thinking

Example:

```text
Mumbai = 0
Delhi = 1
Pune = 2
```

A numeric split may incorrectly interpret this as an ordered relationship.

**Fix:** Use an encoding strategy compatible with the meaning of the category and the tree implementation.

## Mistake 6 — Data Leakage

Do not derive preprocessing parameters from the entire dataset before splitting.

Use a pipeline where appropriate.

## Mistake 7 — Assuming Feature Importance Is Stable

With correlated features, small data changes can alter which feature the tree chooses.

**Fix:** Treat individual-tree feature importance as model-specific and potentially unstable.

## Mistake 8 — Ignoring Tree Depth in Production

A huge tree is difficult to inspect and may be expensive or brittle.

**Fix:** Keep complexity visible and justified.

---

# 34. Interview Questions

## Basic

### Q1. What is a Decision Tree?

**Answer:**

A Decision Tree is a non-parametric supervised learning algorithm that recursively partitions the feature space using decision rules. Internal nodes contain feature-based splits, branches represent outcomes, and leaves contain predictions. It can be used for both classification and regression.

### Q2. How does a Decision Tree work?

**Answer:**

At each node, it considers candidate feature splits, measures the resulting child impurity, selects the best split, and recursively repeats the process on the child nodes until a stopping condition is reached. Final predictions are made from the leaf reached by a new observation.

### Q3. What type of problems can it solve?

**Answer:**

Decision Trees can solve:

- classification;
- regression.

They are especially useful for non-linear relationships, feature interactions, and rule-based decision structure.

---

## Intermediate

### Q4. What are the assumptions of Decision Trees?

**Answer:**

Decision Trees have few strict statistical assumptions. They do not require linearity, normality, or feature scaling. However, useful performance still requires informative and reasonably representative data, appropriate feature representation, and sensible regularization.

### Q5. Does it require feature scaling?

**Answer:**

No. Tree splits depend primarily on the ordering of feature values and selected thresholds, not on Euclidean distance or gradient magnitude. Therefore standardization is generally unnecessary.

### Q6. What are its important hyperparameters?

**Answer:**

The most important ones are:

- `max_depth`;
- `min_samples_split`;
- `min_samples_leaf`;
- `criterion`;
- `max_leaf_nodes`;
- `ccp_alpha`.

These control how the tree is grown and how strongly it is regularized.

### Q7. What happens when `max_depth` increases?

**Answer:**

The tree is allowed to become more complex. Training error usually decreases or stays the same, bias tends to decrease, and variance tends to increase. Beyond a certain point, validation performance can worsen because of overfitting.

---

## Advanced

### Q8. Why can a Decision Tree overfit so badly?

**Answer:**

Because it can keep creating smaller regions until individual observations or tiny groups are isolated. That gives the tree enough flexibility to model random noise in the training data. The result is low training error but potentially poor test performance.

### Q9. What happens if an assumption is violated?

**Answer:**

There are relatively few strict distributional assumptions to violate. Practical problems such as poor encoding, missing values unsupported by the chosen implementation, target leakage, or unrepresentative data can still damage performance.

### Q10. Why would you choose this algorithm over Logistic Regression?

**Answer:**

I would consider a Decision Tree when the relationship is strongly non-linear, feature interactions are important, or I want a rule-based model. Logistic Regression is preferable when a stable linear decision boundary in feature space is a reasonable approximation and coefficient interpretability is important.

### Q11. How would you reduce overfitting?

**Answer:**

I would control complexity using `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `min_impurity_decrease`, or `ccp_alpha`, then choose the settings using cross-validation rather than training accuracy alone.

### Q12. Explain the mathematical intuition behind the algorithm.

**Answer:**

For each candidate split, the tree calculates the weighted impurity of the child nodes:

$$
J=
\frac{N_L}{N}I_L+
\frac{N_R}{N}I_R
$$

It selects the split that minimizes this quantity, or equivalently maximizes impurity reduction:

$$
I_{parent}-J
$$

For classification, $I$ can be Gini or entropy. For regression, it can be squared error/variance. The process is then repeated recursively.

---

# 35. Interview Follow-Up Drill

The first answer is rarely the end of the interview.

### Interviewer: "Why does feature scaling not matter?"

**Your answer:**

Because a tree chooses thresholds based on the ordering of feature values. A monotonic rescaling changes the numeric threshold but not the ordering, so the corresponding partition of samples can remain identical.

### Interviewer: "What happens if we increase `max_depth`?"

**Your answer:**

The tree gains more capacity. Training error can decrease, bias generally decreases, and variance generally increases. If depth becomes excessive, the tree can overfit.

### Interviewer: "Why not use Random Forest instead?"

**Your answer:**

A single Decision Tree is more interpretable and easier to visualize. Random Forest usually provides lower variance and better stability by averaging many diversified trees, but it is less transparent and computationally larger.

### Interviewer: "Does feature scaling matter here?"

**Your answer:**

Usually no. Unlike k-NN or many gradient-based methods, standard tree learning does not depend on feature distance or feature magnitude in the same way. It uses threshold comparisons.

### Interviewer: "What happens with a very large dataset?"

**Your answer:**

Training generally becomes more expensive because there are more samples and candidate split evaluations. Prediction remains relatively inexpensive because each observation only traverses a single root-to-leaf path for one tree. For very large datasets, optimized libraries and ensembles need to be considered based on compute and accuracy requirements.

### Interviewer: "Why does a tree have high variance?"

**Your answer:**

Because early split decisions influence the entire downstream structure. A small data change can alter the best root split, which changes the subsets seen by all later nodes.

### Interviewer: "Why are Random Forests less sensitive to this?"

**Your answer:**

Random Forests train many diversified trees and average their predictions. Averaging reduces the effect of the instability of any one tree.

### Interviewer: "Why is Gini zero for a pure node?"

**Your answer:**

If one class has probability 1 and all others have probability 0:

$$
G=1-(1^2+0+\cdots+0)=0
$$

There is no class uncertainty within the node.

### Interviewer: "Why does regression predict the mean in a leaf?"

**Your answer:**

Because under squared-error loss, the constant value that minimizes:

$$
\sum_i(y_i-c)^2
$$

is the arithmetic mean of the target values.

---

# 36. Explain This Algorithm in an Interview

## 30-Second Explanation

> A Decision Tree is a supervised non-parametric algorithm used for classification and regression. It recursively splits the data using feature-based rules so that the resulting child nodes become more homogeneous. For classification it commonly uses Gini impurity or entropy, while regression can use squared error. The tree continues until stopping criteria are reached, and prediction is made by following a new observation from the root to a leaf. The main issue is overfitting, so tree depth, minimum leaf size, and pruning are important.

## 1-Minute Explanation

> A Decision Tree learns a hierarchy of if/else rules. At each node, it evaluates candidate feature thresholds and chooses the split that gives the best reduction in impurity. For classification, impurity can be measured using Gini impurity or entropy; for regression, a common criterion is squared error. This process is repeated recursively on the child nodes. When a stopping condition is reached, a leaf is created. A classification leaf usually predicts the majority class, while a regression leaf predicts the mean target. Decision Trees are interpretable and handle non-linear interactions well, but a deep tree has high variance and can overfit. We control that using parameters such as `max_depth`, `min_samples_leaf`, and pruning.

## 3-Minute Explanation

> A Decision Tree is a non-parametric supervised learning model that partitions the feature space recursively. Suppose we have a node containing $N$ observations. For every candidate feature and threshold, we split the node into a left and right child. We then calculate the weighted child impurity:
>
> $$
> J=
> \frac{N_L}{N}I_L+
> \frac{N_R}{N}I_R
> $$
>
> We choose the split that minimizes this value, or equivalently maximizes the reduction:
>
> $$
> I_{parent}-J
> $$
>
> In classification, one common impurity measure is Gini:
>
> $$
> G=1-\sum_kp_k^2
> $$
>
> and another is entropy:
>
> $$
> H=-\sum_kp_k\log_2p_k
> $$
>
> In regression, a common criterion is:
>
> $$
> MSE=\frac{1}{N}\sum_i(y_i-\bar y)^2
> $$
>
> We recursively apply the same procedure to child nodes. The algorithm is greedy, so it optimizes the best split at the current node rather than solving for the globally optimal tree structure. Once a stopping condition such as maximum depth or minimum leaf size is reached, we create a leaf. For classification, the leaf usually predicts the most frequent class; for regression, the leaf commonly predicts the mean target.
>
> During prediction, a new observation starts at the root, evaluates each rule, moves left or right, and continues until it reaches a leaf.
>
> The biggest practical issue is overfitting because a deep tree can create very small regions and memorize noise. Therefore I would tune `max_depth`, `min_samples_split`, `min_samples_leaf`, and pruning parameters using cross-validation. A single tree is highly interpretable, but if variance and instability are a major concern, I would consider Random Forest or boosting.

---

# 37. Key Takeaways

## Core Idea

> A Decision Tree recursively asks the most useful feature-based questions to make target values more homogeneous within each child node.

## Mathematical Idea

> Select the feature and threshold that minimize weighted child impurity, or equivalently maximize impurity reduction.

## Training Idea

> Repeatedly find the best local split, partition the data, and recurse until stopping criteria are reached.

## Prediction Idea

> Traverse from root to leaf according to the learned rules and return the leaf's prediction.

## Main Strength

> Easy-to-understand non-linear decision rules with little need for feature scaling or manual interaction engineering.

## Main Limitation

> High variance and overfitting when the tree becomes too complex.

## Most Important Hyperparameters

> `max_depth`, `min_samples_leaf`, `min_samples_split`, `criterion`, and `ccp_alpha`.

## Most Important Assumption

> There are few strict distributional assumptions; however, useful performance still requires appropriate data representation, informative features, and sensible complexity control.

## Most Important Interview Concept

> **The core mathematical mechanism is weighted impurity reduction at every node.**

---

# 38. Completion Checklist

Before marking this algorithm as complete, I should be able to answer:

- [x] What problem does it solve?
- [x] Why do we need it?
- [x] What is the formal definition?
- [x] What is the core intuition?
- [x] How is the problem mathematically formulated?
- [x] What objective/loss function does it use?
- [x] Why that objective function?
- [x] How does training work?
- [x] What does the model actually learn?
- [x] How does prediction work?
- [x] What assumptions does it make?
- [x] Does feature scaling matter?
- [x] How does it behave with outliers?
- [x] Why can it overfit?
- [x] How can overfitting be controlled?
- [x] What are the key hyperparameters?
- [x] What happens when each key hyperparameter changes?
- [x] What are its computational costs?
- [x] What are its strengths and weaknesses?
- [x] When should I use it?
- [x] When should I avoid it?
- [x] How does it compare with related algorithms?
- [x] Can I derive/explain the important mathematics?
- [x] Can I implement it using Scikit-Learn?
- [x] Can I explain the implementation line-by-line?
- [x] Can I answer "why?" follow-ups?
- [x] Can I explain it in 30 seconds, 1 minute, and 3 minutes?

---

# 39. Suggested References

For deeper study and interview preparation:

1. **An Introduction to Statistical Learning (ISLP/ISLR)** — Decision Trees, bagging, Random Forests, and boosting.
2. **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** — practical tree-based modeling.
3. **Scikit-Learn User Guide — Decision Trees** — API behavior, parameters, pruning, and implementation details.
4. **CART: Classification and Regression Trees** — foundational tree methodology by Breiman et al.

## Final Mental Summary

```text
Decision Tree
    ↓
Recursive Partitioning
    ↓
Try feature + threshold
    ↓
Measure child impurity
    ↓
Choose best split
    ↓
Repeat recursively
    ↓
Stop / prune
    ↓
Leaf prediction
    ↓
Generalization depends heavily on tree complexity
```

> **Placement one-liner:** Decision Trees are non-parametric supervised models that recursively split the feature space using impurity reduction, producing interpretable if/else rules but requiring complexity control because a single deep tree has high variance.
