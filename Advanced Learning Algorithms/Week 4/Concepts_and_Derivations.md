# Decision Tree Learning: Concepts and Measures

Decision Tree Learning is a non-parametric supervised learning method used for both classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

---

## 1. Core Decisions in Tree Construction

The process of building a decision tree revolves around two main decisions at every stage:

### 1.1 Decision 1: How to Choose the Optimal Split?

The best feature to split on at a given node is chosen by maximizing **Purity** or, conversely, minimizing **Impurity**.

| Metric | Goal | Description |
| :--- | :--- | :--- |
| **Information Gain** | **Maximize** | This is the primary criterion, calculated as the reduction in entropy (impurity) achieved by a split. |
| **Gini Impurity** | **Minimize** | Measures the probability of misclassifying a randomly chosen element in the dataset if it were randomly labeled according to the class distribution in the node. |

### 1.2 Decision 2: When to Stop Splitting (Pruning/Stopping Criteria)?

Splitting is halted to prevent overfitting and ensure the tree remains generalizable and interpretable. Common stopping conditions include:

* **Maximum Purity Reached:** When a node is **100% one class** (i.e., its impurity is 0).
* **Maximum Depth Exceeded:** The tree has reached a predefined maximum limit on its height.
* **Impurity Threshold:** When the **improvement in purity score (Information Gain)** is **below a certain threshold** (i.e., the split doesn't add enough value).
* **Node Size Threshold:** When the **number of examples in a node** is **below a threshold** (i.e., there are too few data points to warrant a reliable split).

---

## 2. Measuring Impurity: Entropy and Information Gain

**Entropy** is a fundamental concept in information theory that measures the **impurity, disorder, or uncertainty** of a set of data.

### 2.1 The Entropy Formula

For a binary classification problem (e.g., Cat or Not Cat) where $p_1$ is the fraction of examples belonging to class 1 and $p_0 = 1 - p_1$ is the fraction of examples belonging to class 0:

$$\mathbf{H}(p_1) = -p_1 \cdot \log_2(p_1) - p_0 \cdot \log_2(p_0)$$

* We use **base 2** for the logarithm because the result ($\mathbf{H}(p_1)$) represents the minimum number of bits needed to encode the classification of an outcome.

### 2.2 Entropy vs. Purity

| Class Distribution | Example | $\mathbf{H}(p_1)$ Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Maximum Impurity** | $p_1 = 3/6$ (50/50 split) | **1** | Highest uncertainty; worst split. |
| **High Impurity** | $p_1 = 2/6 \approx 0.33$ | $\approx 0.92$ | High uncertainty. |
| **Low Impurity** | $p_1 = 5/6 \approx 0.83$ | $\approx 0.65$ | Lower uncertainty; good purity. |
| **Maximum Purity** | $p_1 = 1$ or $p_1 = 0$ | **0** | Zero uncertainty; node is pure. |

### 2.3 Information Gain (IG)

The goal of splitting is to achieve a **reduction of entropy**, and this reduction is called **Information Gain ($\mathbf{IG}$)**. It measures how much the entropy decreases after a node is split.

$$\mathbf{IG} = \mathbf{H}(p_1^{\text{root}}) - \left( w^{\text{left}} \cdot \mathbf{H}(p_1^{\text{left}}) + w^{\text{right}} \cdot \mathbf{H}(p_1^{\text{right}}) \right)$$

Where:
* $\mathbf{H}(p_1^{\text{root}})$: Entropy of the node **before** the split.
* $w^{\text{left}}$: Weight (fraction) of examples that go to the **left** child node.
* $\mathbf{H}(p_1^{\text{left}})$: Entropy of the **left** child node.
* $w^{\text{right}}$: Weight (fraction) of examples that go to the **right** child node.
* $\mathbf{H}(p_1^{\text{right}})$: Entropy of the **right** child node.

**The split that yields the maximum Information Gain is chosen as the optimal split.**
