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
w = w - alpha * dj_dw  
b = b - alpha * dj_db

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

### Logistic Regression
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

For \( y = 1, 2, \dots, N \):

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

Let’s assume \( z_2 = 0 \) and rename \( z_1 = z \):  
(We can shift logits by a constant — this doesn’t change probabilities.)

$$
a_1 = \frac{e^{z}}{e^{z} + 1}
$$

$$
a_1 = \frac{1}{1 + e^{-z}}
$$

---
