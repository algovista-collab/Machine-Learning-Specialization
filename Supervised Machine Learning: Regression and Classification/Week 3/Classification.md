# Logistic Regression Notes

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
z = \mathbf{w} \cdot \mathbf{x} + b
$$

This $z$ is then passed through the **sigmoid (activation) function**:  

$$
g(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}} 
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
