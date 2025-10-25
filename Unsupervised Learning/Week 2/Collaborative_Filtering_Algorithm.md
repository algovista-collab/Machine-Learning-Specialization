# Collaborative Filtering: Content-Based Approach

In the Content-Based approach to Collaborative Filtering, we assume that features describing each item (e.g., movie genres) are known. We then learn a set of parameters for each user to predict their rating for any given item.

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
