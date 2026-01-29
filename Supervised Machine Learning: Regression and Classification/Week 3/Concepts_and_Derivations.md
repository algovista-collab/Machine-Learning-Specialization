# ✍️ Logistic Regression Notes

## 1. Binary Classification

In classification, when the output values can take only **2 values** (like True/False), it is called **Binary Classification**.  

- If the answer is "yes," we call it the **positive class**.  
- The opposite is the **negative class**.  

If we try to predict this using **Linear Regression**, whose output can take values beyond 0 and 1, we set a **threshold**. For example, if the threshold is 0.5:  

- If the predicted value $f_{\mathbf{w},b}(\mathbf{x}) < 0.5$, classify it as **negative class**  
- If $f_{\mathbf{w},b}(\mathbf{x}) \ge 0.5$, classify it as **positive class**  

However, adding new training examples may cause the model to predict poorly. This is why we use **Logistic Regression**, which is designed to output values between **0 and 1**.

---

## 2. Exponential Function and Sigmoid

The **exponential function** $e^x$ behaves as follows:  

- As $x \to 0$, $e^x \to 1$  
- As $x \to \infty$, $e^x \to \infty$  
- As $x \to -\infty$, $e^x \to 0$  

The function $\frac{1}{1+e^x}$ outputs values **only from 0 to 1**, but it is **decreasing**.  

To make it **monotonic increasing**, we use:  

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

This is called the **sigmoid function**, which maps $z \in (-\infty, \infty)$ to $p \in (0, 1)$.  

**Example:**

- $x = -2 \to \frac{1}{1 + e^{-(-2)}} = \frac{1}{1 + e^{2}} \approx 0.119$  
- $x = 0 \to \frac{1}{1 + e^{0}} = 0.5$  
- $x = 2 \to \frac{1}{1 + e^{-2}} \approx 0.881$

---

## 3. Logistic Regression Model

Since $z$ can take any value between $-\infty$ to $+\infty$, we feed the output of a **linear regression model** as $z$:  

$$
z = \vec{w} \cdot \vec{x} + b
$$

This $z$ is then passed through the **sigmoid (activation) function**:  

$$
g(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}} 
$$

The output $g(z)$ represents the **probability** of the positive class.

---

## 4. Decision Boundary

For some models, when $z = 0$, i.e., $g(z) = 0.5$, this acts as a **decision boundary**, separating the target variables 0 and 1.  

**Example:**  

If we have 2 features $x_0$ and $x_1$, with $w_0 = 1$, $w_1 = 1$, and $b = -3$, the linear combination is:

$$
z = w_0 x_0 + w_1 x_1 + b = x_0 + x_1 - 3
$$

For $z = 0$, the **decision boundary line** is:

$$
x_0 + x_1 = 3
$$

- To the right of the line, $y = 1$  
- To the left of the line, $y = 0$

## 5. Cost (Loss) Function Derivation

Before making predictions, we must evaluate **how good our model is**.  
To do this, we measure the **cost function**, which represents the difference between the predicted and actual values.

- A **higher cost** means a **worse model**.  
- Our goal is to **minimize the cost function**, i.e., find its **global minimum**.

---

### Problem with Linear Regression Cost in Logistic Regression

In **linear regression**, we use a **Mean Squared Error (MSE)** cost function, which produces a nice convex curve — easy to optimize.

However, in **logistic regression**, the prediction is:

$$
\hat{y} = \frac{1}{1 + e^{-z}} \quad \text{where } z = w^T x + b
$$

If we use the same MSE cost here, we get a **non-convex (wiggly)** function with **many local minima**, making optimization difficult.

Hence, we use a **different cost function** — one derived from **Maximum Likelihood Estimation (MLE)**.

---

### 🔹 Step 1: Probability of an outcome

For a single training example:

$$
P(y|x) =
\begin{cases}
\hat{y}, & \text{if } y = 1 \\
1 - \hat{y}, & \text{if } y = 0
\end{cases}
$$

This can be combined into a single equation:

$$
P(y|x) = \hat{y}^y (1 - \hat{y})^{(1 - y)}
$$

---

### 🔹 Step 2: Likelihood for the whole dataset

For all training examples:

$$
L(w, b) = \prod_{i=1}^{m} \hat{y}_i^{y_i} (1 - \hat{y}_i)^{(1 - y_i)}
$$

We take the **logarithm** (to simplify multiplication into addition):

$$
\log L(w, b) = \sum_{i=1}^{m} \big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \big]
$$

---

### 🔹 Step 3: Define the Cost Function

Since we want to **minimize** the loss (not maximize likelihood),  
we take the **negative** of the average log-likelihood:

$$
J(w, b) = -\frac{1}{m} \sum_{i=1}^{m}
\Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big]
$$

---

### 🎯 Final Logistic Loss Function

$$
-\boxed{
\Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big]
}
$$

---

### 📊 Behavior of the Loss

- When **y = 1**, loss = $$(-\log(\hat{y})$$)
  - As $$(\hat{y} \to 1\)$$, loss → 0  
  - As $$(\hat{y} \to 0\)$$, loss → ∞

- When **y = 0**, loss = $$(-\log(1 - \hat{y})$$)
  - As $$(\hat{y} \to 0\)$$, loss → 0  
  - As $$(\hat{y} \to 1\)$$, loss → ∞

This forms **two separate curves**, showing how the model is penalized for incorrect predictions.

---

### 🧠 Statistical Insight

This loss function arises naturally from **Maximum Likelihood Estimation (MLE)**,  
which provides a mathematically sound way to estimate parameters that make the observed data most probable.

---

## 6. Gradient Descent – Derivation

## Loss Function (Binary Cross-Entropy)

$$
L(w, b) = -\Big[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})\Big]
$$

---

### Derivative Formulas to know

### Sigmoid Derivative

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\frac{d}{dz} \sigma(z) = \sigma(z) \big(1 - \sigma(z)\big)
$$

---

### Logarithm Derivative

$$
\frac{d}{dx} \log(u(x)) = \frac{u'(x)}{u(x)}
$$

---

We want: 

$$
\frac{\partial L}{\partial w}
$$

### Step 1: Apply chain rule

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

---

### Step 2: Derivative of loss w.r.t $$\hat{y}$$

$$
L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

$$
\frac{\partial L}{\partial \hat{y}} = -\left( \frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}} \right)
= \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}
$$

---

### Step 3: Derivative of sigmoid w.r.t z

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y})
$$

---

### Step 4: Derivative of z w.r.t w

$$
z = wx + b \implies \frac{\partial z}{\partial w} = x
$$

---

### Step 5: Combine using chain rule

$$
\frac{\partial L}{\partial w} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) \cdot x
$$

$$
\frac{\partial L}{\partial w} = (\hat{y} - y) \cdot x
$$

---

### Step 6: Gradient w.r.t b

Similarly, for bias b:

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial b}
$$

$$
\frac{\partial z}{\partial b} = 1
$$

$$
\frac{\partial L}{\partial b} = (\hat{y} - y)
$$

---

## Gradient Descent Update Rule

Let $\eta$ be the learning rate:

$$
w := w - \eta \frac{\partial L}{\partial w} = w - \eta (\sigma(wx+b) - y)x
$$

$$
b := b - \eta \frac{\partial L}{\partial b} = b - \eta (\sigma(wx+b) - y)
$$

---

## 7. Model Performance Theory: Bias-Variance Tradeoff

### **1. Underfitting (High Bias)** 📉

Underfitting occurs when a model is **too simple** to capture the underlying patterns in the training data.

* **Definition:** The model assumes a relationship between input and output that is simpler than the reality (e.g., using a linear model for a quadratic relationship).
* **Result:** The model performs **poorly** on both the **training data** and **new, unseen data**. It fails to learn the complex structure of the training set.
* **Characteristic:** **High Bias** and **Low Variance**.
* **Analogy:** A student who doesn't study enough for the test.
* **Remedies:**
    * Use a **more complex model** (e.g., add polynomial features, increase hidden layers in a neural network).
    * **Decrease the regularization strength** ($\lambda$).

---

### **2. Fits Well / Good Generalization (Optimal Balance)** ✅

A model that fits well has achieved a good balance between bias and variance.

* **Definition:** The model accurately captures the essential relationships in the training data without fitting the noise.
* **Result:** The model performs **well** on both the **training data** and **new, unseen data**.
* **Characteristic:** **Low Bias** and **Low Variance** (relatively speaking).
* **Analogy:** A student who studies the concepts thoroughly and performs well on the test, even with new types of questions.

---

### **3. Overfitting (High Variance)** 📈

Overfitting occurs when a model is **too complex** and fits the training data **too closely**, including the random noise and specific anomalies.

* **Definition:** The model learns a highly intricate, specific relationship that perfectly matches the training set but is irrelevant to the true relationship.
* **Result:** The model performs **exceptionally well** on the **training data** but **very poorly** on **new, unseen data**. It memorizes the data instead of learning to generalize.
* **Characteristic:** **Low Bias** and **High Variance**.
* **Analogy:** A student who memorizes every practice problem's answer, including typos, but fails on a test with slightly different problems.
* **Remedies:**
    * **Regularization** (L1 or L2) to penalize large weights.
    * **Collect more training data**.
    * **Reduce the number of features** (Feature Selection).
    * Use a **simpler model** (e.g., fewer polynomial features).

---

## 8. L2 Regularization Equations (Ridge Regression)

L2 Regularization (Ridge) modifies the cost function by adding a penalty term that is proportional to the **square of the magnitude** of the weights ($\mathbf{w}$). This discourages weights from growing too large, leading to a simpler, more generalized model. The bias term ($b$) is typically **not** regularized.

---

### **1. L2 Regularized Linear Regression (Ridge Regression)**

#### Cost Function ($$J(\mathbf{w}, b)$$)

The cost function combines the Mean Squared Error (MSE) with the L2 penalty:

$$J(\mathbf{w}, b) = \underbrace{\frac{1}{2m} \sum_{i=1}^{m} (f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)})^2}_{\text{Original Cost (MSE)}} + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2}_{\text{L2 Regularization Term}}$$

* $f_{\mathbf{w}, b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$ (Linear Prediction)
* $\lambda$: Regularization parameter ($\lambda \geq 0$)
* $m$: Number of training examples

#### Partial Derivatives for Gradient Descent

The gradient descent rule uses these derivatives to update $\mathbf{w}$ and $b$.

**For the Bias Parameter ($b$):** (No change from standard Linear Regression)
$$\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)})$$

**For the Weight Parameters ($w_j$) for $j=1, \dots, n$:** (Includes the derivative of the L2 penalty, $\frac{\lambda}{m} w_j$)
$$\frac{\partial J(\mathbf{w}, b)}{\partial w_j} = \left( \frac{1}{m} \sum_{i=1}^{m} (f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)} \right) + \frac{\lambda}{m} w_j$$

---

### 2. L2 Regularized Logistic Regression

#### Cost Function ($$J(\mathbf{w}, b)$$)

The cost function combines the Binary Cross-Entropy Loss with the L2 penalty:

$$J(\mathbf{w}, b) = \underbrace{\left[ -\frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} \log(f_{\mathbf{w}, b}(\mathbf{x}^{(i)})) + (1 - y^{(i)}) \log(1 - f_{\mathbf{w}, b}(\mathbf{x}^{(i)})) \right) \right]}_{\text{Original Cost (Binary Cross-Entropy Loss)}} + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2}_{\text{L2 Regularization Term}}$$

* $f_{\mathbf{w}, b}(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$ (Sigmoid Function $$g(z)$$)

#### Partial Derivatives for Gradient Descent

**For the Bias Parameter ($b$):** (No change from standard Logistic Regression)
$$\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)})$$

**For the Weight Parameters ($w_j$) for $j=1, \dots, n$:** (Includes the derivative of the L2 penalty)
$$\frac{\partial J(\mathbf{w}, b)}{\partial w_j} = \left( \frac{1}{m} \sum_{i=1}^{m} (f_{\mathbf{w}, b}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)} \right) + \frac{\lambda}{m} w_j$$

### Comparison: Ridge ($L_2$) vs. Lasso ($L_1$)

| Feature | Ridge Regression ($L_2$) | Lasso Regression ($L_1$) |
| :--- | :--- | :--- |
| **Penalty Term** | Sum of **squared** weights ($\beta^2$) | Sum of **absolute** weights ($|\beta|$) |
| **Feature Selection** | No; shrinks coefficients near zero. | **Yes**; can shrink coefficients to exactly zero. |
| **Model Complexity** | High (retains all features). | Low (creates a sparse model). |
| **Multicollinearity** | Handles it by distributing weights. | Picks one feature and drops the others. |
| **Use Case** | When most features are useful. | When you have a "needle in a haystack" (few useful features). |

---

### The Cost Functions

In both equations, $RSS$ is the **Residual Sum of Squares**. The parameter $\alpha$ (or $\lambda$) controls the regularization strength.

#### Ridge Regression
$$J(\theta) = RSS + \alpha \sum_{j=1}^{n} \theta_j^2$$



#### Lasso Regression
$$J(\theta) = RSS + \alpha \sum_{j=1}^{n} |\theta_j|$$



> **Key Takeaway:** The "Diamond" shape of the Lasso constraint is why it hits the axes, setting coefficients to zero. The "Circular" shape of Ridge prevents coefficients from ever reaching exactly zero.

# Elastic Net Regression

Elastic Net is a regularized regression method that linearly combines the $L_1$ and $L_2$ penalties of the Lasso and Ridge methods.

### The Cost Function
The objective function to minimize is:

$$J(\theta) = RSS + \lambda_1 \sum_{j=1}^{n} |\theta_j| + \lambda_2 \sum_{j=1}^{n} \theta_j^2$$

In software like Scikit-Learn, this is often controlled by two parameters:
1. **$\alpha$ (Alpha):** The overall penalty strength.
2. **L1_ratio ($l1\_ratio$):** The mix between $L_1$ and $L_2$.
   - If `l1_ratio = 1`, it is pure **Lasso**.
   - If `l1_ratio = 0`, it is pure **Ridge**.
   - If `0 < l1_ratio < 1`, it is **Elastic Net**.

---

### Why Use Elastic Net?

| Problem | Lasso ($L_1$) | Ridge ($L_2$) | Elastic Net Solution |
| :--- | :--- | :--- | :--- |
| **Feature Selection** | Excellent (zeros out weights) | Poor (keeps all features) | **Excellent** (performs selection) |
| **Correlated Variables** | Picks one at random | Keeps all of them | **Best** (groups them together) |
| **$n < p$ (More features than data)** | Selects at most $n$ features | Keeps all $p$ features | **Best** (can select more than $n$) |

---

### Visual Comparison of Constraints

Elastic Net's constraint region combines the "corners" of Lasso with the "curves" of Ridge.



---

### Implementation Example (Python)

```python
from sklearn.linear_model import ElasticNet

# l1_ratio=0.5 means 50% Lasso and 50% Ridge
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)

print(f"Coefficients: {enet.coef_}")
```

No Free Lunch Theorem: The theorem states that no single, universally superior optimization or machine learning algorithm exists; all algorithms perform equally well when averaged across every possible problem.
