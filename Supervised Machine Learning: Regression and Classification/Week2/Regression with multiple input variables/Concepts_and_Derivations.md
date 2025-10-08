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
