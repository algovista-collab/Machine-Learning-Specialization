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

## 3. Why We Can't Predict $\mathbf{x}$ in Simple Linear Regression

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
