In the Linear Regression, we used a model with only single variable: f(x) = w*x + b

In the Multiple Linear Regression: we use a model with multiple variables: $$f_{\vec{w}, b}(\vec{x}) = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + \dots + w_n \cdot x_n + b$$ ---- (1)

We can simply denote this using vector which is nothing but a list of numbers, we usually use row vector: $$\vec{x} = [x_1, x_2, x_3, \dots, x_n] \quad (x_j \text{ is the } j\text{th feature})$$
$$\vec{w}$$ is a vector with multiple weights for the corresponding features. These numbers indicate how important that particular feature is while predicting the output.

Vector Representation of the Model: $$f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$
The dot product from Linear Algebra, denotes each value of $$\vec{w}$$ is multipled with the values in $$\vec{x}$$ and the size of the $$\vec{w}$$ and $$\vec{x}$$ should be the same.

The benefits of using vector representation is to speed up the calculations by using a popular numerical linear algebra library called NumPy (Numerical Python). It has a function called dot which is used to get the product of 2 vectors. The dot function is able to use parallel hardware whether it is a normal computer CPU or a GPU (Graphics Processing Unit) to accelerate the ML jobs especially when the dataset is very large.

Vectorization - The equation 1 in the concise code: $$f = np.dot(\vec{w}, \vec{x}) + b$$

The number of elements in the array is referred to as the dimension or rank of the vector. The arrays in NumPy is used to represent 1-D or more of the same data type called dtype.

Gradient Descent for multiple variables are calculated for different feature parameters. 

$$\text{Update for } w_j: \quad w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

$$\text{Update for } b: \quad b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)$$

Feature Scaling: If the range of any feature varies significantly like size of the house (x1) from 200 to 3000 sq ft and other feature like number of bedrooms (x2) varies only from 0 to 5, we have to scale to bring them to the same range.
Without scaling, any small change in the parameter w1 for x1 will cause the cost function to change significantly because w1 is multiplied by a big number and to make significant change in the cost function, we need to multiply with big number of w2. This results in thin ellipse when we take the contour plot of the cost function w.r.t w1 and w2. To minimize the cost function, we calculate gradient descent which can bounce back nd forth for a long time to reach the minimum. Hence, we need scaling to get circles in the contour plot.

1. Min-Max Scaling: Dividing by maximum value will scale between 0 and 1

$$X_{\text{scaled}} := \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

2. Mean Normalization: Taking mean of the feature values and subtract the mean from the value and dividing by (maximum-minimum)

$$X_{\text{scaled}} := \frac{X - \mu}{X_{\text{max}} - X_{\text{min}}}$$
   
3. Z-Score Normalization: Subtracting the mean from the value and dividing by the standard deviation

$$X_{\text{scaled}} := \frac{X - \mu}{\sigma}$$
why 
The formulas for calculating the mean ($\mu_j$) and variance ($\sigma^2_j$) for feature $j$ are:

$$\mu_j = \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j$$

$$\sigma^2_j = \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2 $$

Note that $\sigma_j$ (standard deviation) is the square root of $\sigma^2_j$ (variance).

When generating the plot, the normalized features were used. Any predictions using the parameters learned from a normalized training set must also be normalized.

In order to know after how many iterations, we should stop - we can either check the learning curve and stop when cost function flattens and does not change much when iterations increases. This is called convergence. 
Another method is called Automatic convergence test where we set a threshold called epsilon (ε) which is a small number and we check whether the change in the cost function between iterations is smaller than the epsilon. If it is, then gradient descent has converged. But if the chosen epsilon is too small, then it might never converge or if its big then it might stop too early before even converging.

How to pick a learning rate? 
1. Typical starting values: 0.001, 0.01, 0.1 (depending on the problem and feature scaling).
2. Observing the cost function J over iterations.
3. Increasing or decreasing based on the plot.

Feature Engineering is using the intuition to design new features by transforming or combining original features. This can include squaring some of the features which might result in non-linear curve. In such cases, Polynomial Regression will fit the curves to the data. The quadratic function is a parabola which decreases eventually, hence it might be better to pick cubic function.

# ✍️ Multiple Linear Regression Notes

## 1. Model Representation and Vectorization

### The Multiple Linear Regression Model
In Multiple Linear Regression, the model uses multiple features ($\vec{x}$) to predict the output.

$$f_{\vec{w}, b}(\vec{x}) = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + \dots + w_n \cdot x_n + b \quad \text{--- (1)}$$

### Vector Representation
Features ($\vec{x}$) and weights ($\vec{w}$) are represented as **vectors**. $\vec{w}$'s elements indicate the importance of corresponding features.

$$\vec{x} = [x_1, x_2, x_3, \dots, x_n] \quad (x_j \text{ is the } j\text{th feature})$$

The model is concisely written using the **dot product** ($\vec{w} \cdot \vec{x}$):
$$f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$

### Vectorization
**Vectorization** is the implementation of vector operations using numerical linear algebra libraries like **NumPy**.

* **Benefit**: NumPy's `np.dot` function utilizes **parallel hardware** (CPU/GPU) to significantly **accelerate** calculations, especially with large datasets.
* **Concise Code**:
    ```python
    f = np.dot(w, x) + b 
    ```
* The number of elements in a vector is its **dimension** or **rank**.

---

## 2. Gradient Descent for Multiple Variables

Gradient Descent is used to find the optimal parameters ($\vec{w}, b$). Updates are calculated for each feature weight ($w_j$) and the bias ($b$) simultaneously.

* $\alpha$ is the **learning rate**.
* $m$ is the number of training examples.

$$\text{Update for } w_j: \quad w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

$$\text{Update for } b: \quad b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w}, b}(\vec{x}^{(i)}) - y^{(i)} \right)$$

---

## 3. Feature Scaling (Normalization) 📏

### Why Scaling is Necessary
When feature ranges vary widely (e.g., house size: 200–3000 sq ft vs. bedrooms: 0–5), the cost function's contour plot forms a **thin, elongated ellipse**. This causes Gradient Descent to waste time "bouncing" back and forth instead of moving directly to the minimum.

**Scaling** brings all features to a similar range, making the cost function contours more circular, which allows Gradient Descent to converge much **faster** and more directly.

### Scaling Methods

1.  **Min-Max Scaling** (Normalization): Scales to the range $[0, 1]$.
    $$X_{\text{scaled}} := \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

2.  **Mean Normalization**: Scales features to have a mean of 0.
    $$X_{\text{scaled}} := \frac{X - \mu}{X_{\text{max}} - X_{\text{min}}}$$

3.  **Z-Score Normalization** (Standardization): Scales features to have a mean ($\mu$) of 0 and a standard deviation ($\sigma$) of 1. This is often the preferred method.
    $$X_{\text{scaled}} := \frac{X - \mu}{\sigma}$$

#### Mean and Variance Calculation
For feature $j$:
$$\mu_j = \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j \quad (\text{Mean})$$

$$\sigma^2_j = \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2 \quad (\text{Variance})$$

> **Important**: Any new data used for prediction **must** be normalized using the $\mu$ and $\sigma$ (or $X_{\text{min}}, X_{\text{max}}$) calculated *from the training set*.

---

## 4. Convergence and Learning Rate

### Detecting Convergence
Convergence is reached when the cost function flattens out and stops decreasing significantly.

1.  **Learning Curve Check**: Plot $J$ (cost function) vs. iterations and stop when the curve **flattens**.
2.  **Automatic Convergence Test**: Define a small threshold $\varepsilon$ (epsilon). Stop if the change in cost function between consecutive iterations is less than $\varepsilon$:
    $$\text{If } |J_{\text{new}} - J_{\text{old}}| < \varepsilon$$
    * **Caution**: Choosing $\varepsilon$ too small may prevent convergence; too large may stop training prematurely.

### Picking the Learning Rate ($\alpha$)
The learning rate is crucial for ensuring efficient convergence.

1.  **Typical Starting Values**: Test powers of 10 and 3, such as: **0.001, 0.01, 0.1** (and potentially 0.003, 0.03, 0.3).
2.  **Observation**: Monitor the cost function $J$ over iterations on a plot.
3.  **Adjustment**:
    * If $J$ **increases** or **oscillates**, $\alpha$ is **too large**. **Decrease $\alpha$**.
    * If $J$ **decreases very slowly**, $\alpha$ is **too small**. **Increase $\alpha$**.

---

## 5. Feature Engineering and Polynomial Regression

### Feature Engineering
This involves using **intuition** and domain knowledge to **design new features** by transforming or combining existing ones.

* **Goal**: To better capture the underlying relationship between features and the target variable.

### Polynomial Regression
When the data does not follow a linear path, we can introduce **polynomial terms** (e.g., $x^2, x^3$) through feature engineering to fit **non-linear curves**.

* Example: $f(x) = w_1 \cdot x + w_2 \cdot x^2 + w_3 \cdot x^3 + b$.
* A **quadratic** ($x^2$) function is a parabola that eventually decreases, which might not be ideal for many problems. A **cubic** ($x^3$) function often provides a more flexible curve fit.
