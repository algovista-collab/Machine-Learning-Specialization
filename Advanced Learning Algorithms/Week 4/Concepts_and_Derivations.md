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

* We use **base 2** for the logarithm because the result ($$\mathbf{H}(p_1)$$) represents the minimum number of bits needed to encode the classification of an outcome.

<img width="407" height="277" alt="image" src="https://github.com/user-attachments/assets/325d24fb-9fc0-4a32-804a-a1764b64a701" />

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

## 3. Decision Tree Algorithm, Ensembles, and Feature Engineering

---

## The Decision Tree Algorithm: High-Level Steps

The process of building a Decision Tree is an iterative, greedy process that starts with all examples at the root node:

1.  **Initialize Root:** Start with all training examples at the root node.
2.  **Calculate Information Gain (IG):** For the current node, calculate the $\mathbf{IG}$ for **all possible features** and all possible split points.
3.  **Select Best Split:** **Pick the feature and split point with the highest Information Gain.**
4.  **Split Node:** Split the dataset according to the selected feature and create **right and left branches** (child nodes) of the tree.
5.  **Iterate:** **Keep repeating the splitting process** for each new child node.
6.  **Stop:** The process continues until a **stopping criterion** is met (e.g., maximum depth, minimum sample size, or zero impurity).

---

## Feature Handling Techniques

Decision Trees must handle various data types effectively to find optimal splits.

### 3.1 Categorical Features with Multiple Values

When a categorical feature can take **more than two values** ($k > 2$):

* **Method:** We typically use **One-Hot Encoding**.
* **Process:** If a categorical feature can take on $k$ unique values, you **create $k$ binary features** (0 or 1 valued). The tree then makes a binary decision on each new binary feature (e.g., "Is Feature\_Color\_Red = 1?").

### 3.2 Continuous Valued Features

For a feature with continuous values (like height or temperature):

* **Process:** All unique values in the feature are considered potential split points.
    * For a dataset with $N$ examples (e.g., 10 animals), there are at most **$N-1$ unique split points** to test.
    * The possible split points are the **mid-points** between the sorted unique values in the feature (e.g., choosing the 9 mid-points between 10 examples).
* **Goal:** The algorithm tests all these potential split points and selects the one that results in the **highest Information Gain**.

---

## 4. Tree Ensemble Methods

A **Tree Ensemble** is a collection of multiple decision trees that work together to make a final prediction. Ensembles make the algorithm **less sensitive** to minor variations in the data and more **robust** overall.

| Ensemble Type | Description | Key Mechanism |
| :--- | :--- | :--- |
| **Random Forest** | A tree ensemble algorithm that often works much better than using a single decision tree. | **Bootstrapping & Feature Subsetting:** Each tree is trained on a different subset of data (via **Sampling with Replacement**). |
| **Random Forest (Feature Selection)** | A specific technique within the Random Forest construction. | **Feature Subsetting at Each Node:** At each node, if $n$ features are available, the algorithm picks a random subset of **$k < n$ features** and is only allowed to choose a split from that small subset. This decorrelates the trees. |
| **XGBoost** | **eXtreme Gradient Boosting** (XGBoost) is an open-source implementation of boosted trees. | **Boosting and Regularization:** Trees are built sequentially, with each new tree correcting the errors of the previous ones. It includes **built-in regularization** to prevent overfitting. |

---

## 5. Applicability of Decision Trees

* **Decision Trees (DTs)** work better on **tabular (structured) data** (e.g., spreadsheets, databases).
* **Neural Networks (NNs)** work well on **both structured and unstructured data**.
    * Examples of unstructured data include **images**, video, and free text.
