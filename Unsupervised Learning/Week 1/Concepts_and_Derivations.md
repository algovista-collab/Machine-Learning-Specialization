# K-means Clustering Algorithm

K-means is an **unsupervised learning algorithm** used for **clustering** data points. It automatically finds groups of related or similar data points within a dataset.

## Core Idea
The algorithm seeks to partition $m$ data points into $K$ distinct, non-overlapping subsets (clusters), where each data point belongs to the cluster with the nearest **mean** (centroid).

## Algorithm Steps

The K-means algorithm iteratively optimizes the cluster assignments and centroid locations.

### 1. Initialization
1.  Specify the number of clusters, **$K$**, to find.
2.  Randomly select $K$ data points from the dataset to serve as the initial **cluster centroids** ($\mu_1, \mu_2, \dots, \mu_K$).

---

### 2. Iterative Optimization (Repeat until Convergence)
The process iterates between two steps: **Cluster Assignment** and **Centroid Update**.

#### A. Cluster Assignment Step
Assign each data point $x^{(i)}$ to the cluster centroid $\mu_k$ that is closest to it.

For $i = 1$ to $m$:
$$
c^{(i)} := \text{index } (\text{from } 1 \text{ to } K) \text{ of cluster centroid closest to } x^{(i)}
$$

This is mathematically represented as:
$$
c^{(i)} = \arg \min_{k} \| x^{(i)} - \mu_k \|^2
$$
where $\| \cdot \|^2$ is the **squared Euclidean distance** (L2 norm).

---

#### B. Centroid Update Step
Recalculate the position of each cluster centroid $\mu_k$ by taking the average (mean) of all data points currently assigned to that cluster.

For $k = 1$ to $K$:
$$
\mu_k := \text{Average of all points assigned to cluster } k
$$

**Special Case:** If a cluster has **no points** assigned to it, the common practice is to **eliminate that centroid** or **reinitialize it** randomly.

---

### 3. Convergence
The algorithm is considered to have **converged** when the cluster assignments $c^{(i)}$ no longer change, or the change in the cluster centroids $\mu_k$ is below a small threshold.

## K-means Optimization Objective (Distortion Cost Function)

The goal of the K-means algorithm is to **minimize the distortion** (or cost) function, $J$.

### Notation
* $m$: Number of training examples.
* $K$: Number of clusters.
* $x^{(i)}$: The $i^{th}$ training example.
* $c^{(i)}$: The index (1 to $K$) of the cluster to which example $x^{(i)}$ is currently assigned.
* $\mu_k$: The cluster centroid for cluster $k$.
* $\mu_{c^{(i)}}$: The cluster centroid of the cluster to which example $x^{(i)}$ has been assigned.

### Cost Function ($J$)
The distortion cost function measures the **sum of the squared distances** between each data point and its assigned cluster centroid.

$$
J(c^{(1)}, \dots, c^{(m)}, \mu_1, \dots, \mu_K) = \frac{1}{m} \sum_{i=1}^{m} \| x^{(i)} - \mu_{c^{(i)}} \|^2
$$

### Property
A crucial property of the K-means optimization is that the value of the cost function $J$ will **never increase** with every iteration; it will either **stay the same** or **decrease**. The algorithm is guaranteed to converge to a local optimum.
