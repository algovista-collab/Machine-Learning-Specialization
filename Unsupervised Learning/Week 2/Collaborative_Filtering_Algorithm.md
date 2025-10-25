# Collaborative Filtering: Content-Based Approach

In the Content-Based approach to Collaborative Filtering, we assume that features describing each item (e.g., movie genres) are known. We then learn a set of parameters for each user to predict their rating for any given item. It is gathering of data from multiple users to predict the value for future users.

## 1. Notation and Variables

| Symbol | Description |
| :--- | :--- |
| $n_u$ | Number of users. |
| $n_m$ | Number of movies (items). |
| $n$ | Number of features (e.g., genres). |
| $r(i, j)$ | Binary variable: $r(i, j) = 1$ if user $j$ has rated movie $i$, and $0$ otherwise. |
| $y^{(i, j)}$ | The rating given by user $j$ for movie $i$ (only defined when $r(i, j)=1$). |
| $\mathbf{x}^{(i)}$ | Feature vector for movie $i$, where $\mathbf{x}^{(i)} \in \mathbb{R}^n$. (e.g., $[0.9, 0, \dots]^T$ for Romance, Action, etc.) |
| $\mathbf{w}^{(j)}$ | Parameter vector for user $j$, where $\mathbf{w}^{(j)} \in \mathbb{R}^n$. |
| $b^{(j)}$ | Bias term (or intercept) for user $j$, where $b^{(j)} \in \mathbb{R}$. |
| $m_j$ | The total number of movies rated by user $j$ (used for normalization in the cost function). |

## 2. Rating Prediction

For a specific user $j$ and movie $i$ with features $\mathbf{x}^{(i)}$, the predicted rating $\hat{y}^{(i, j)}$ is calculated using a linear model:

$$
\hat{y}^{(i, j)} = (\mathbf{w}^{(j)})^T \mathbf{x}^{(i)} + b^{(j)}
$$

---

## 3. Cost Function for a Single User ($j$)

The objective is to find the parameters $\mathbf{w}^{(j)}$ and $b^{(j)}$ that minimize the squared error between the predicted ratings and the actual ratings for all movies *rated by user j*, plus a regularization term.

$$
J(\mathbf{w}^{(j)}, b^{(j)}) = \frac{1}{2 m_j} \sum_{i: r(i, j)=1} \left( (\mathbf{w}^{(j)})^T \mathbf{x}^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2 m_j} \sum_{k=1}^{n} (w_k^{(j)})^2
$$

**Note:**
* The summation $\sum_{i: r(i, j)=1}$ means we are **summing only over the movies $i$ that user $j$ has actually rated**.
* This cost function is used to learn the parameters $\mathbf{w}^{(j)}$ and $b^{(j)}$ for a *single user j*.
* **Regularization Term:** $\frac{\lambda}{2 m_j} \sum_{k=1}^{n} (w_k^{(j)})^2$ prevents overfitting by penalizing large parameter values. The bias term $b^{(j)}$ is typically **not** included in this regularization, but it is sometimes regularized in practice.

---

## 4. Overall Cost Function

To learn the parameters for **all users** (from $j=1$ to $n_u$), we sum the individual cost functions. We typically drop the $\frac{1}{m_j}$ normalization when combining all terms:

$$
J(\mathbf{w}^{(1)}, \dots, \mathbf{w}^{(n_u)}, b^{(1)}, \dots, b^{(n_u)}) = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i: r(i, j)=1} \left( (\mathbf{w}^{(j)})^T \mathbf{x}^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2
$$

The goal of the overall learning process is to **minimize** this joint cost function $J$ with respect to all user parameters $\mathbf{w}^{(j)}$ and $b^{(j)}$.

## Learning Item Features

In the pure Collaborative Filtering approach, we don't assume the item features (like genre scores) are known. Instead, we assume the user parameters ($\mathbf{w}^{(j)}$ and $b^{(j)}$) have been learned, and the goal is to **learn the feature vector $\mathbf{x}^{(i)}$ for each movie $i$**.

## 1. Rating Prediction

The predicted rating $\hat{y}^{(i, j)}$ uses the same linear model form:

$$
\hat{y}^{(i, j)} = (\mathbf{w}^{(j)})^T \mathbf{x}^{(i)} + b^{(j)}
$$

## 2. Cost Function to Learn Features ($\mathbf{x}^{(i)}$)

The objective is to find the feature vector $\mathbf{x}^{(i)}$ that minimizes the squared error for all users who have rated movie $i$, plus a regularization term.

The cost function $J$ to learn **all movie feature vectors** ($\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n_m)}$) is:

$$
J(\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n_m)}) = \frac{1}{2} \sum_{i=1}^{n_m} \sum_{j: r(i, j)=1} \left( (\mathbf{w}^{(j)})^T \mathbf{x}^{(i)} + b^{(j)} - y^{(i, j)} \right)^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2
$$

**Note:**
* The summation $\sum_{j: r(i, j)=1}$ means we are **summing only over the users $j$ that have rated movie $i$**.
* The overall process typically alternates between learning the user parameters ($\mathbf{w}^{(j)}, b^{(j)}$) and the item features ($\mathbf{x}^{(i)}$) until convergence, a technique known as **Alternating Least Squares**.

---

## 3. Collaborative Filtering: Unified Cost Function

The unified objective is to find the parameters for **all users** ($\mathbf{w}^{(1)}, \dots, \mathbf{w}^{(n_u)}, b^{(1)}, \dots, b^{(n_u)}$) and the feature vectors for **all movies** ($\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n_m)}$) that minimize the total prediction error.

## Unified Cost Function

The cost function $J$ is the sum of the squared prediction errors across all known ratings, plus regularization terms for both the user parameters and the movie features.

$$
J(\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(n_m)}, \mathbf{w}^{(1)}, \dots, \mathbf{w}^{(n_u)}, b^{(1)}, \dots, b^{(n_u)}) =
$$

$$
\frac{1}{2} \sum_{(i, j): r(i, j)=1} \left( (\mathbf{w}^{(j)})^T \mathbf{x}^{(i)} + b^{(j)} - y^{(i, j)} \right)^2
$$

$$
\frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^{n} (w_k^{(j)})^2
\frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^{n} (x_k^{(i)})^2
$$

**Where:**
* $\sum_{(i, j): r(i, j)=1}$: Sums over **all pairs** $(i, j)$ where user $j$ has rated movie $i$. This is the total number of known ratings.
* The first term is the **total squared prediction error**.
* The second term is the **regularization for user parameters** $\mathbf{w}$.
* The third term is the **regularization for movie features** $\mathbf{x}$.

---

## 4. Gradient Descent Formulas

To minimize the unified cost function $J$, we use **Gradient Descent**. The parameters are updated iteratively by moving in the direction opposite to the gradient.

We must compute the partial derivative of $J$ with respect to every single parameter: $w_k^{(j)}$, $b^{(j)}$, and $x_k^{(i)}$.

Let $$\mathbf{e}^{(i, j)}$$ be the prediction error for rating $(i, j)$:

$$
\mathbf{e}^{(i, j)} = (\mathbf{w}^{(j)})^T \mathbf{x}^{(i)} + b^{(j)} - y^{(i, j)}
$$

### 1. Update Rule for Movie Feature $x_k^{(i)}$ (for a specific feature $k$ of movie $i$)
$$
x_k^{(i)} := x_k^{(i)} - \alpha \frac{\partial}{\partial x_k^{(i)}} J
$$

$$
\frac{\partial}{\partial x_k^{(i)}} J = \sum_{j: r(i, j)=1} \left( \mathbf{e}^{(i, j)} \cdot w_k^{(j)} \right) + \lambda x_k^{(i)}
$$

### 2. Update Rule for User Parameter $w_k^{(j)}$ (for a specific feature $k$ of user $j$)
$$
w_k^{(j)} := w_k^{(j)} - \alpha \frac{\partial}{\partial w_k^{(j)}} J
$$

$$
\frac{\partial}{\partial w_k^{(j)}} J = \sum_{i: r(i, j)=1} \left( \mathbf{e}^{(i, j)} \cdot x_k^{(i)} \right) + \lambda w_k^{(j)}
$$

### 3. Update Rule for User Bias $b^{(j)}$ (Bias term for user $j$)
$$
b^{(j)} := b^{(j)} - \alpha \frac{\partial}{\partial b^{(j)}} J
$$

$$
\frac{\partial}{\partial b^{(j)}} J = \sum_{i: r(i, j)=1} \left( \mathbf{e}^{(i, j)} \right)
$$

**Where:**
* $\alpha$ is the **learning rate** (step size).
* The summations only include the terms for which a rating exists ($r(i, j)=1$).
* The terms $+\lambda x_k^{(i)}$ and $+\lambda w_k^{(j)}$ come from the derivative of the regularization terms. **Note:** The bias term $b^{(j)}$ is typically **not** regularized (hence no $\lambda$ term in its derivative).

---

## 5. Binary Classification: Model and Cost Function

## 1. Model Prediction (The Logistic Function)

The model predicts the **probability** that the output is $y=1$. This is achieved by using the **Logistic Function (Sigmoid)**, $\sigma(z)$, which maps any real-valued input $z$ to a probability between 0 and 1.

Given an input $\mathbf{x}$ and learned parameters $\mathbf{w}$ and $b$:

$$
z = \mathbf{w}^T \mathbf{x} + b
$$

The predicted probability $\hat{y}$ is:

$$
\hat{y} = P(y=1 | \mathbf{x}) = g(z) = \frac{1}{1 + e^{-z}}
$$

| Symbol | Description |
| :--- | :--- |
| $\mathbf{x}$ | Input feature vector. |
| $\mathbf{w}$ | Weight vector (parameters). |
| $b$ | Bias term (parameter). |
| $z$ | The linear combination of inputs and weights. |
| $\hat{y}$ | The model's predicted probability that the label is $y=1$. |
| $e$ | Euler's number (base of the natural logarithm). |

---

## 2. Cost Function (Binary Cross-Entropy Loss / Log Loss)

For a single training example $(\mathbf{x}, y)$, the loss $L(\hat{y}, y)$ measures how far the predicted probability $\hat{y}$ is from the true label $y$.

$$
L(\hat{y}, y) = - [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

### How the Loss Function Works:

* **If the true label $y=1$:** The loss simplifies to $-\log(\hat{y})$. To minimize loss, $\hat{y}$ must be close to 1.
* **If the true label $y=0$:** The loss simplifies to $-\log(1 - \hat{y})$. To minimize loss, $\hat{y}$ must be close to 0 (i.e., $1 - \hat{y}$ must be close to 1).

## 3. Overall Cost Function (J)

The overall cost function $J(\mathbf{w}, b)$ is the average of the loss over all $m$ training examples:

$$
J(\mathbf{w}, b) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]
$$

| Symbol | Description |
| :--- | :--- |
| $m$ | Total number of training examples. |
| $y^{(i)}$ | True label for the $i$-th example (either 0 or 1). |
| $\hat{y}^{(i)}$ | Predicted probability for the $i$-th example. |
| $\sum_{i=1}^{m}$ | Summation over all training examples. |

---

## 6. Why We Can't Predict $\mathbf{x}$ in Simple Linear Regression

In standard **Linear Regression** (or the Content-Based approach), when we learn user parameters $\mathbf{w}^{(j)}$, we assume the movie features $\mathbf{x}^{(i)}$ are **fixed and known**.

The reason we **cannot directly learn $\mathbf{x}^{(i)}$** in the Content-Based approach is:

### Dependency on All Users
To learn the movie features $\mathbf{x}^{(i)}$ (i.e., its "Romance score," "Action score," etc.), you need input from **all users $j$** who have rated that movie $i$. The features are universal properties of the movie, not unique to a single user.

### Missing Parameters
In the simple Content-Based approach for a single user $j$:

$$
\text{Minimize } J(\mathbf{w}^{(j)}, b^{(j)}) \text{ given } \mathbf{x}^{(i)}
$$

The problem is set up to only learn $\mathbf{w}^{(j)}$ and $b^{(j)}$. If you were to try to find $\mathbf{x}^{(i)}$ using *only* user $j$'s ratings, the features would be tailored only to user $j$'s preferences, not the movie's true characteristics as perceived by the community.

### Collaborative Filtering Solution
The cost function above solves this by including the ratings and parameters ($\mathbf{w}^{(j)}, b^{(j)}$) from **all relevant users** ($\sum_{j: r(i, j)=1}$). This forces the learned feature vector $\mathbf{x}^{(i)}$ to be a good predictor for **everyone** who has rated the movie, making it a true, community-driven representation of the movie's traits.

## Mean Normalization

Mean normalization is a technique used in **feature scaling** to ensure that all features have a mean value close to zero. This is often applied after or in conjunction with **feature scaling** (like dividing by the range or standard deviation) to help optimization algorithms converge faster.

## 1. Mean Normalization Formula

For a training set with $m$ examples, the mean-normalized value $$x_j^{(i)}$$ of the $$j$$-th feature for the $$i$$-th example is calculated as:

$$
x_j^{(i)} = \frac{x_j^{(i)} - \mu_j}{\sigma_j}
$$

| Symbol | Description |
| :--- | :--- |
| $x_j^{(i)}$ | The **mean-normalized** value of the $j$-th feature for the $i$-th example. |
| $x_j^{(i)}$ | The **original** value of the $j$-th feature for the $i$-th example. |
| $\mu_j$ | The **mean (average)** value of all $m$ training examples for the $j$-th feature. |
| $\sigma_j$ | A measure of feature magnitude, typically the **standard deviation** or the **range** (max - min) of the $j$-th feature. |

## 2. Calculation of Parameters

The parameters $\mu_j$ and $\sigma_j$ are calculated *only* from the **training data**.

### A. Feature Mean ($\mu_j$)

$$
\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}
$$

### B. Scaling Factor ($\sigma_j$)

The scaling factor $\sigma_j$ is typically one of the following:

* **Standard Deviation ($\text{std}(x_j)$):** This is generally preferred when features already have a somewhat normal distribution.

$$
\sigma_j = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2}
$$

* **Range ($R_j$):** The difference between the maximum and minimum values of the feature. This is sometimes used for simplicity.

$$
\sigma_j = \max(x_j) - \min(x_j)
$$

## 3. Effect and Purpose

The resulting normalized feature $x'_j$ will have the following properties:

1.  **Mean of Zero:** The average value of $x'_j$ across the training set will be $\approx 0$.
2.  **Standardized Range:** The values will generally fall within the range of $[-1, 1]$.

**Purpose:** Normalization ensures that all features contribute proportionally to the distance calculations and cost function. This prevents features with large natural scales (e.g., house size) from dominating features with small scales (e.g., number of bedrooms) and **speeds up the convergence of Gradient Descent**.

## Squared Euclidean Distance

Once a model is trained, the **squared distance** between two feature vectors gives an indication of how similar the corresponding items are in the feature space.

The squared Euclidean distance between two vectors $\mathbf{x}^{(k)}$ and $\mathbf{x}^{(i)}$ is calculated as:

$$
\text{distance} = \| \mathbf{x}^{(k)} - \mathbf{x}^{(i)} \|^2 = \sum_{l=1}^{n} (x_l^{(k)} - x_l^{(i)})^2
$$

**Where:**
* $\mathbf{x}^{(k)}$: The feature vector for item $k$.
* $\mathbf{x}^{(i)}$: The feature vector for item $i$.
* $n$: The number of features (or latent factors/dimensions).
* $x_l^{(k)}$: The value of the $l$-th feature for item $k$.
* $\| \cdot \|^2$: Denotes the **squared L2 norm** (Euclidean distance).

<img width="1162" height="563" alt="image" src="https://github.com/user-attachments/assets/5f27936e-fd31-45d3-81ca-8b6cc5bf1159" />
