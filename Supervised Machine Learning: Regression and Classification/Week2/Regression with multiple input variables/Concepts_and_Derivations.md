In the Linear Regression, we used a model with only single variable: f(x) = w*x + b

In the Multiple Linear Regression: we use a model with multiple variables: $$f_{\vec{w}, b}(\vec{x}) = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + \dots + w_n \cdot x_n + b$$ ---- (1)

We can simply denote this using vector which is nothing but a list of numbers, we usually use row vector: $$\vec{x} = [x_1, x_2, x_3, \dots, x_n] \quad (x_j \text{ is the } j\text{th feature})$$
$$\vec{w}$$ is a vector with multiple weights for the corresponding features. These numbers indicate how important that particular feature is while predicting the output.

Vector Representation of the Model: $$f_{\vec{w}, b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$
The dot product from Linear Algebra, denotes each value of $$\vec{w}$$ is multipled with the values in $$\vec{x}$$ and the size of the $$\vec{w}$$ and $$\vec{x}$$ should be the same.

The benefits of using vector representation is to speed up the calculations by using a popular numerical linear algebra library called numpy. It has a function called dot which is used to get the product of 2 vectors. The dot function is able to use parallel hardware whether it is a normal computer CPU or a GPU (Graphics Processing Unit) to accelerate the ML jobs especially when the dataset is very large.

The equation 1 in the concise code: $$f = np.dot(\vec{w}, \vec{x}) + b$$
