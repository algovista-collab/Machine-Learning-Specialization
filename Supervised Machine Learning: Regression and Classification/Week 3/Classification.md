In **Classification**, especially when output values can take only **2 values** like true or false (categories or classes), it is called **Binary Classification**.  

If for a question, the answer is yes, then we can call it a **positive class**, and the opposite is called the **negative class**.  

If we try to predict this using **Linear Regression**, whose output can take values more than just 2, then we set a **threshold**.  

If the threshold is 0.5:

$$
\text{if } f_{wb}(x) < 0.5 \text{, classify as positive class; otherwise, classify as negative class}
$$

But as we add a new training example, the model doesn't predict properly. Hence, we enter into **Logistic Regression**, which is a model designed to output either 0 or 1.  

---

First, let's see the exponential function:

$$
e^x
$$

- As $x \to 0$, $e^x \to 1$  
- As $x \to \infty$, $e^x \to \infty$  
- As $x \to -\infty$, $e^x \to 0$

The function 

$$
\frac{1}{1 + e^x}
$$

outputs values only from 0 to 1. $x$ can be between $-\infty$ to $+\infty$. For any value in this range, the function takes values between 0 and 1.  

**Example:**

- $x = -2 \to \frac{1}{1 + e^{-2}} \approx 0.119$  
- $x = 0 \to \frac{1}{1 + e^{0}} = 0.5$  
- $x = 2 \to \frac{1}{1 + e^{-2}} \approx 0.881$

The model objective to output values between 0 and 1 is satisfied, but it should be **monotonic**, meaning as \(x\) increases, the function should also increase. Hence we consider:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

This maps $z \in (-\infty, \infty)$ to $p \in (0, 1)$.

The functions are mirrored by the property:

$$
\sigma(-z) = 1 - \sigma(z)
$$

This is mostly a **convention**.


Since $z$ can take any value between $-\infty$ to $+\infty$, we feed the output of the linear regression model as $z$, i.e.,

$$
z = \mathbf{w} \cdot \mathbf{x} + b
$$

This will be given as input to the **sigmoid (activation) function**, denoted as:

$$
g(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}
$$

**Decision Boundary:**  

For some models, when $z = 0$, i.e., $g(z) = 0.5$, this can act as a **decision boundary**, i.e., it separates the target variables which are 0 and 1.  

**Example:**  

If we have an equation with 2 features, $x_0$ and $x_1$, with $w_0 = 1$, $w_1 = 1$, and $b = -3$, the linear combination is:

$$
z = w_0 x_0 + w_1 x_1 + b = x_0 + x_1 - 3
$$

For $z = 0$, we get the line:

$$
x_0 + x_1 = 3
$$

This line can be considered the **decision boundary**.  
- To the right of the line, $y = 1$  
- To the left of the line, $y = 0$

