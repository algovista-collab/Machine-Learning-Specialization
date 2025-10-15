# 🧠 Model Training Steps

---

## 1. Define the Output

### Logistic Regression
$$
z = w \cdot x + b
$$

$$
f(x) = \frac{1}{1 + e^{-z}}
$$

---

### Neural Network

```python
model = Sequential([
    Dense(...),
    Dense(...)
])
```
---

## 2. Specify the Loss and Cost

### Logistic Loss
$$
\text{Loss} = -y \cdot \log(\hat{y}) - (1 - y) \cdot \log(1 - \hat{y})
$$

### Neural Network Compilation
```python
model.compile(loss=BinaryCrossentropy())  
or  
model.compile(loss=MeanSquaredError())
```
---

## 3. Training on Data to Minimize the Cost

### Logistic Regression
$$
w = w - \alpha \cdot \frac{\partial J}{\partial w}
$$

$$
b = b - \alpha \cdot \frac{\partial J}{\partial b}
$$

### Neural Network
```python
model.fit(X, y, epochs=100)
```
---

## ⚙️ Different Activation Functions

| Activation Function | Formula | Notes |
|----------------------|----------|--------|
| **ReLU** | g(z) = max(0, z) | Most common in hidden layers, faster convergence |
| **Linear** | g(z) = w * x + b | Used for regression problems |
| **Sigmoid** | g(z) = 1 / (1 + np.exp(-z)) | Used for binary classification |

> **Note:** Without activation functions, a neural network behaves like simple linear regression.  
> Activation functions introduce non-linearity, allowing the model to handle complex patterns.

---

## 🧩 Output Layer Activations

- Choose **ReLU**, **Sigmoid**, or **Linear** based on the problem type.  
- **Hidden Layers:** ReLU is preferred because it only flattens on one side (faster and easier optimization).  
- **Sigmoid:** Flattens at both ends (0 and 1), leading to slower convergence.

---

## 🔢 Softmax — Generalization of Logistic Regression

- **Logistic Regression** → Binary Classification  
- **Softmax** → Multi-class Classification  

### 🔹 Logistic Regression (Binary Classification)

$$
z = w \cdot x + b
$$

$$
a_1 = g(z) = \frac{1}{1 + e^{-z}} = P(y=1 \mid x)
$$

$$
a_2 = 1 - a_1 = P(y=0 \mid x)
$$

---

### 🔹 Softmax for Multiclass Classification

For classes \( y = 1, 2, 3, 4 \):

$$
z_1 = w_1 \cdot x + b_1
$$

$$
z_2 = w_2 \cdot x + b_2
$$

$$
z_3 = w_3 \cdot x + b_3
$$

$$
z_4 = w_4 \cdot x + b_4
$$

$$
a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}
$$

$$
a_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}
$$

$$
a_3 = \frac{e^{z_3}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}
$$

$$
a_4 = \frac{e^{z_4}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}
$$

---

### 🔹 General Softmax Formula

For $$( y = 1, 2, \dots, N $$):

$$
z_j = w_j \cdot x + b_j \quad \text{for } j = 1, 2, \dots, N
$$

$$
a_j = \frac{e^{z_j}}{\sum_{k=1}^{N} e^{z_k}}
$$

$$
P(y = j \mid x) = a_j
$$

---

## 🧮 Mathematical Connection

Let’s show how **sigmoid = 2-class softmax**.

For binary classes (2 outputs):

$$
a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}}
$$

$$
a_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2}}
$$

Let’s assume $$( z_2 = 0 $$) and rename $$( z_1 = z $$):  
(We can shift logits by a constant — this doesn’t change probabilities.)

$$
a_1 = \frac{e^{z}}{e^{z} + 1}
$$

$$
a_1 = \frac{1}{1 + e^{-z}}
$$

---

### Why exponentials?

- Exponentials $$(e^{z_j}$$) are **always positive**, no matter what $$(z_j$$) is.

- So if we take:

$$
a_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + e^{z_3}}, \quad
a_2 = \frac{e^{z_2}}{e^{z_1} + e^{z_2} + e^{z_3}}, \quad
a_3 = \frac{e^{z_3}}{e^{z_1} + e^{z_2} + e^{z_3}}
$$

we guarantee:  

1. $$(a_j > 0$$) ✅  
2. $$(\sum_{j=1}^{3} a_j = 1$$) ✅  

- Now each $$(a_j$$) can be interpreted as a **probability of class $$(j$$)**.

---

# 🔹 Softmax Cross-Entropy Loss for N Classes

The general formula:

$$
\text{Loss} = - \sum_{j=1}^{N} y_j \cdot \log(a_j)
$$

- $$(y_j = 1$$) for the true class, $$(0$$) otherwise  
- Therefore, for a single example with true class $$(c$$), all terms vanish except the one for the true class:

$$
\text{Loss} = - \log(a_c)
$$

- If you look at all classes individually:

$$
\text{Loss} =
\begin{cases}
-\log(a_1), & \text{if class 1 is true} \\
-\log(a_2), & \text{if class 2 is true} \\
\vdots \\
-\log(a_N), & \text{if class N is true}
\end{cases}
$$

> **Note:** z is called logit as it is the raw score produced by a linear model before the activation function or sigmoid function is applied

## 🔢 Numerical Stability: $\text{Softmax/Sigmoid}$ with Cross-Entropy Loss

The issue arises from potential **numerical underflow or overflow** when calculating probabilities and their logarithms, especially when $z$ (the input to the activation) is very large (positive or negative).

### ❌ The Unstable Approach

In binary classification, the probability $\hat{y}$ is:
$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$
The **Binary Cross-Entropy Loss** ($\mathcal{L}$) for a true label $y$ is:
$$
\mathcal{L}(y, \hat{y}) = - [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

| Scenario | $\mathbf{z}$ | $\mathbf{\hat{y}}$ | $\mathbf{\log(\hat{y})}$ | **Issue** |
| :---: | :---: | :---: | :---: | :--- |
| $\mathbf{z} \rightarrow \infty$ | $\infty$ | $1$ | $\log(1) = 0$ | $\log(1-\hat{y}) = \log(0) = -\infty$ (Numerical Error/`nan` in loss) |
| $\mathbf{z} \rightarrow -\infty$ | $-\infty$ | $0$ | $\log(0) = -\infty$ | $\log(\hat{y}) = -\infty$ (Numerical Error/`nan` in loss) |

> **Example:** If $z = -1000$, $e^{-z}$ is huge (overflow). If $z = 1000$, $\hat{y} \approx 1$ and $1-\hat{y} \approx 0$. $\log(0)$ is undefined, leading to `NaN` in the loss calculation.

### ✅ The Stable Solution: Log-Sum-Exp Trick / Using Logits

Instead of computing  

$$
\hat{y} = \frac{1}{1 + e^{-z}}
$$  

and then  

$$
\log(\hat{y})
$$  

separately, a **numerically stable** way to compute  

$$
\log(\hat{y}) \quad \text{and} \quad \log(1 - \hat{y})
$$  

is used by combining the **Sigmoid** activation and the **Cross-Entropy** loss into a single operation:

$$
\mathcal{L}(y, z) = y \cdot \mathbf{\text{Softplus}(-z)} + (1 - y) \cdot \mathbf{\text{Softplus}(z)}
$$  

where  

$$
\mathbf{\text{Softplus}(x)} = \log(1 + e^x)
$$  

This is derived from substituting $$\hat{y}$$ back into the loss function and using logarithm rules:

* For $$\log(\hat{y})$$:  
  $$
  \log\left(\frac{1}{1 + e^{-z}}\right)
  = -\log(1 + e^{-z})
  = -\mathbf{\text{Softplus}(-z)}
  $$

* For $$\log(1 - \hat{y})$$:  
  $$
  \log\left(1 - \frac{1}{1 + e^{-z}}\right)
  = \log\left(\frac{e^{-z}}{1 + e^{-z}}\right)
  = -z - \log(1 + e^{-z})
  = -z + \mathbf{\text{Softplus}(-z)}
  $$

By performing this combined calculation, large intermediate values like $$e^{-z}$$ (when $z$ is large positive) or $$e^{z}$$ (when $z$ is large negative) are **avoided/handled gracefully** through mathematical reformulations, preventing overflow/underflow.

#### Framework Implementation

* **Previous Layer:** The final layer is a **`Linear`** (or `Dense`) layer, outputting the **logits ($z$)**.
* **Loss Function:** Use `model.compile(BinaryCrossentropy(from_logits=True))`
    * The `from_logits=True` flag tells the loss function to expect the raw $z$ values and internally apply the numerically stable combined computation of $\log(\sigma(z))$ and $\log(1-\sigma(z))$.

| 💡 **Advantage** | **Result** |
| :--- | :--- |
| **Numerical Stability** | Prevents $\log(0)$ and $e^{\pm\text{large number}}$ issues. |
| **Efficiency** | One combined calculation is faster than two separate ones. |
