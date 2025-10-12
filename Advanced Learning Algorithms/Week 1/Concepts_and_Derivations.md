# 🧠 Neural Networks: Core Concepts and Backpropagation Derivations

---

## 1. Forward Propagation: Structure of a Neuron

A single neuron in layer $[l]$ processes the activations $\mathbf{a}^{[l-1]}$ from the previous layer in two steps: a linear combination and a non-linear activation.

### A. Linear Combination (The $Z$ Term)

This calculates the weighted sum of inputs plus the bias.

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

| Variable | Description | Shape (Example) |
| :--- | :--- | :--- |
| $\mathbf{z}^{[l]}$ | Linear component for layer $l$ | $(n^{[l]}, m)$ |
| $\mathbf{W}^{[l]}$ | Weight matrix for layer $l$ | $(n^{[l]}, n^{[l-1]})$ |
| $\mathbf{a}^{[l-1]}$ | Activations from previous layer | $(n^{[l-1]}, m)$ |
| $\mathbf{b}^{[l]}$ | Bias vector for layer $l$ | $(n^{[l]}, 1)$ |
| $m$ | Number of training examples | - |
| $n^{[l]}$ | Number of neurons in layer $l$ | - |

### B. Non-Linear Activation (The $A$ Term)

The linear result is passed through a non-linear activation function $g$.

$$\mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$$

| Activation $g(z)$ | Derivative $g'(z)$ (for Backprop) | Use Case |
| :--- | :--- | :--- |
| **Sigmoid** | $g(z)(1-g(z))$ | Output layer for Binary Classification |
| **ReLU** | $1$ if $z > 0$, $0$ otherwise | Hidden layers (most common) |
| **Tanh** | $1 - (g(z))^2$ | Hidden layers (often better than Sigmoid) |

---

## 2. Loss and Cost Functions

### A. Loss Function (Binary Classification)

The standard loss for a single example $i$ is **Binary Cross-Entropy Loss (Log Loss)**.

$$L(y^{(i)}, \hat{y}^{(i)}) = - \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right]$$

### B. Cost Function ($J$)

The overall cost is the average loss across all $m$ training examples.

$$J(\mathbf{W}, \mathbf{b}) = \frac{1}{m} \sum_{i=1}^{m} L(y^{(i)}, \hat{y}^{(i)}) \quad \text{where } \hat{y} = a^{[L]}$$

---

## 3. Backpropagation: The Gradient Derivation

Backpropagation is the efficient calculation of $\frac{\partial J}{\partial \mathbf{W}}$ and $\frac{\partial J}{\partial \mathbf{b}}$ using the **Chain Rule**, working backward from the final layer $L$.

### A. Output Layer ($L$): Simplification

For a network using **Sigmoid** activation and **Cross-Entropy Loss**, the derivative of the cost $J$ with respect to the linear term $\mathbf{Z}^{[L]}$ is remarkably simple:

$$\mathbf{d Z}^{[L]} = \frac{\partial J}{\partial \mathbf{Z}^{[L]}} = \mathbf{A}^{[L]} - \mathbf{Y}$$

* **Intuition:** The complex calculus of $\frac{\partial L}{\partial \mathbf{A}^{[L]}}$ and $\frac{\partial \mathbf{A}^{[L]}}{\partial \mathbf{Z}^{[L]}}$ simplifies entirely to the basic **(Prediction - Actual)** error term. This is the cornerstone of NN learning.

### B. General Backpropagation Step (Layer $l$ to $l-1$)

For any general layer $l$ (where $l < L$):

#### 1. Calculate Error for the Previous Activation ($\mathbf{dA}^{[l-1]}$)

The error signal $d\mathbf{Z}^{[l]}$ is passed to the previous layer via the weights $\mathbf{W}^{[l]}$.

$$\mathbf{d A}^{[l-1]} = (\mathbf{W}^{[l]})^T \mathbf{d Z}^{[l]}$$

#### 2. Calculate Error for the Linear Term ($\mathbf{d Z}^{[l-1]}$)

The error passed from the subsequent layer is scaled by the **derivative of the activation function** $g'$ used in layer $l-1$.

$$\mathbf{d Z}^{[l-1]} = \mathbf{d A}^{[l-1]} \ * \ g'(\mathbf{Z}^{[l-1]})$$

> *(Note: The asterisk $(*)$ here denotes element-wise multiplication, often called the Hadamard product).*

#### 3. Calculate Gradients for Current Layer Parameters ($d\mathbf{W}^{[l]}$ and $d\mathbf{b}^{[l]}$)

These derivatives are averaged over the $m$ training examples and represent the direction and magnitude for the parameter update.

$$\mathbf{d W}^{[l]} = \frac{1}{m} \mathbf{d Z}^{[l]} (\mathbf{A}^{[l-1]})^T$$

$$\mathbf{d b}^{[l]} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{d Z}^{[l]}$$

---

## 4. Gradient Descent Parameter Update

The parameters are updated simultaneously for all layers using the calculated gradients and a learning rate $\alpha$.

$$\mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \alpha \ \mathbf{d W}^{[l]}$$

$$\mathbf{b}^{[l]} := \mathbf{b}^{[l]} - \alpha \ \mathbf{d b}^{[l]}$$
