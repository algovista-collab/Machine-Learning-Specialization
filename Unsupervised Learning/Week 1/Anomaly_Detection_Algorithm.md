# Anomaly Detection Algorithm using Density Estimation

Anomaly Detection is an **unsupervised learning technique** that models a dataset of **normal events** and identifies events that deviate significantly from this model.

## Core Idea: Density Estimation

The algorithm models the probability distribution $P(\mathbf{x})$ of the features in the normal training data. Events with a very low probability are flagged as anomalies.

### Dataset and Feature Representation
* **Dataset:** $\mathcal{D} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(m)} \}$
    * $m$: Number of training examples (normal events).
    * $\mathbf{x}^{(i)}$: The $i^{th}$ training example, which is an $n$-dimensional **feature vector**.
    * $n$: Number of features.
    * $\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}, \dots, x_n^{(i)}]^T$

---

## The Model: Multivariate Gaussian Distribution (Simplified)

The algorithm assumes that each feature $x_j$ is independently distributed according to a **Gaussian (Normal) distribution**.

### 1. Gaussian Probability Density Function (PDF)
The probability of a single feature $x$ is given by:

$$
P(x; \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{(x - \mu)^2}{2 \sigma^2}}
$$

| Symbol | Meaning |
| :--- | :--- |
| $P(x; \mu, \sigma^2)$ | Probability density of $x$, given parameters $\mu$ and $\sigma^2$. |
| $\mu$ | The **mean** of the feature's values. |
| $\sigma^2$ | The **variance** of the feature's values ($\sigma$ is the standard deviation). |
| $e$ | Euler's number (approx. 2.71828). |

### 2. Probability of a Feature Vector $\mathbf{x}$
Assuming **feature independence**, the probability of an entire feature vector $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$ is the **product** of the probabilities of its individual features:

$$
P(\mathbf{x}) = P(x_1; \mu_1, \sigma_1^2) \cdot P(x_2; \mu_2, \sigma_2^2) \cdot \ldots \cdot P(x_n; \mu_n, \sigma_n^2)
$$

This is compactly written using the product operator:

$$
P(\mathbf{x}) = \prod_{j=1}^{n} P(x_j; \mu_j, \sigma_j^2)
$$

| Symbol | Meaning |
| :--- | :--- |
| $P(\mathbf{x})$ | Estimated probability of the vector $\mathbf{x}$. |
| $\prod_{j=1}^{n}$ | The product of terms from $j=1$ to $n$. |
| $\mu_j$ | The mean of the $j^{th}$ feature ($x_j$). |
| $\sigma_j^2$ | The variance of the $j^{th}$ feature ($x_j$). |

---

## Anomaly Detection Algorithm Steps

### Step 1: Feature Selection
**Choose $n$ features** ($\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n$) that are likely to be indicative of an anomalous example.

### Step 2: Parameter Fitting
Use the **Maximum Likelihood Estimation (MLE)** method on the training set $$\mathcal{D}$$ to fit the Gaussian parameters ($$\mu_j$$ and $$\sigma_j^2$$) for **each feature $$j$$ independently**:

* **Mean ($$\mu_j$$):** The average value of feature $j$ across all $m$ examples.

$$
\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}
$$

* **Variance ($$\sigma_j^2$$):** The average squared difference between $$x_j$$ and its mean.
  
$$
\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2
$$

### Step 3: Anomaly Detection
Given a **new example** 

$$\mathbf{x}_{\text{test}}$$

compute its probability 

$$P(\mathbf{x}_{\text{test}})$$ 

using the fitted parameters.

* **Anomaly Threshold ($\varepsilon$):** A predetermined small value (e.g., $10^{-5}$) used to define the boundary between normal and anomalous events.

* **Decision Rule:**

$$P(\mathbf{x}_{\text{test}}) < \varepsilon \implies \mathbf{x}_{\text{test}}$$

is flagged as an **Anomaly** (Red Flag).

$$P(\mathbf{x}_{\text{test}}) \ge \varepsilon \implies \mathbf{x}_{\text{test}}$$ 

is considered **Not an Anomaly**.

| Symbol | Meaning |
| :--- | :--- |
| $\mathbf{x}_{\text{test}}$ | A new, unseen example to be classified. |
| $\varepsilon$ (Epsilon) | The probability threshold for flagging an anomaly. |

## Evaluating Anomaly Detection Systems (Real-Number Evaluation)

While Anomaly Detection is an **unsupervised learning** method (using unlabeled normal data for training), its performance is typically evaluated and its threshold ($\varepsilon$) tuned using a **labeled dataset** of both normal (non-anomalous) and anomalous examples.

## Dataset Split for Evaluation

We assume we have a single dataset that has been partitioned into three labeled subsets:

| Dataset | Purpose | Labeled Examples | Typical Ratio of Anomalies |
| :--- | :--- | :--- | :--- |
| **Training Set** | **Model Fitting** (Learn $P(\mathbf{x})$) | $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(m)}$ | $y=0$ for all or almost all examples |
| **Cross-Validation (CV) Set** | **Threshold Tuning** ($\varepsilon$) | $(\mathbf{x}_{\text{cv}}^{(i)}, y_{\text{cv}}^{(i)})$ for $i=1$ to $m_{\text{cv}}$ | Small number of $y=1$ (anomaly) examples |
| **Test Set** | **Final Evaluation** (Measure performance) | $(\mathbf{x}_{\text{test}}^{(i)}, y_{\text{test}}^{(i)})$ for $i=1$ to $m_{\text{test}}$ | Small number of $y=1$ (anomaly) examples |

**Labels:**
* $y=0$: Non-anomalous (normal) example.
* $y=1$: Anomalous example.

---

## Evaluation Procedure

### 1. Training the Model
* Train the Anomaly Detection algorithm (e.g., Gaussian Distribution) **only** on the **Training Set**.
* **Fit the Parameters:** Calculate the means ($\mu_j$) and variances ($\sigma_j^2$) for each feature $j$.
    * *Note:* The training set is treated as purely non-anomalous ($y=0$), ensuring the model learns the probability distribution of normal events.

### 2. Cross-Validation (CV) for Threshold Tuning ($\varepsilon$)
The CV set is used to select the optimal threshold $\varepsilon$.

* **Iteration:** Iterate through a range of possible values for $\varepsilon$ (e.g., $10^{-1}$ down to $10^{-20}$).
* **Detection:** For each example $\mathbf{x}_{\text{cv}}^{(i)}$ in the CV set, compute $P(\mathbf{x}_{\text{cv}}^{(i)})$ and make a prediction $\hat{y}$ based on the current $\varepsilon$:
    * If $P(\mathbf{x}_{\text{cv}}^{(i)}) < \varepsilon$, then $\hat{y}=1$ (predicted anomaly).
    * If $P(\mathbf{x}_{\text{cv}}^{(i)}) \ge \varepsilon$, then $\hat{y}=0$ (predicted normal).
* **Evaluation Metric:** Because the datasets are highly skewed (many $y=0$, very few $y=1$), standard accuracy is usually insufficient. **F1-Score** or **Precision/Recall** are commonly used metrics to select the best $\varepsilon$.
* **Selection:** Choose the value of $\varepsilon$ that maximizes the desired evaluation metric (e.g., the F1-Score) on the CV set.

### 3. Testing and Final Performance Measurement
* Use the **optimal $\varepsilon$** found in the CV step.
* Compute $P(\mathbf{x}_{\text{test}}^{(i)})$ for every example in the **Test Set** and make predictions $\hat{y}_{\text{test}}$.
* Calculate the final performance (F1-Score, Precision/Recall) on the Test Set.

---

## When to Use Anomaly Detection vs. Supervised Learning

| Situation | Anomaly Detection is Good | Supervised Learning is Good |
| :--- | :--- | :--- |
| **Positive Examples ($y=1$ anomalies)** | **Very Small Number** (e.g., 0-20 examples). | **Large Number** (enough to train a robust classifier). |
| **Type of Anomalies** | **Many Different Types** of anomalies. | Anomalies are **likely to be similar** to those already seen. |
| **Future Anomalies** | Future anomalies **look nothing like** any seen before. | Future anomalies are **similar** to the existing positive examples. |
| **Example Applications** | Fraud detection, Manufacturing finding **new previously unseen defects**, Monitoring machines in a data center. | Email spam classification, Finding **known types of defects**, Weather prediction, Disease classification. |
