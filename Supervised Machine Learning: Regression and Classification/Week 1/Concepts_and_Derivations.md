1. Machine Learning is a field where machines are trained to predict the output without explicitly being programmed.
2. Supervised Learning: Both inputs (feature variables) and outputs (target variables) are given to train and machine predicts the output for a new input.
3. There are 2 types: Linear Regression - Output can be any continuous value or real value (Ex: Housing Price Prediction) Classification - Output can be any finite categories/classes (Ex: Spam filtering)
4. Unsupervised Learning: Only input is given to the machine and it recognized the pattern in the input data and cluster the output with common characteristics. Ex: Clustering Algorithm
5. Linear Regression Model or Univariate Model: This has only one input variable

Linear Regression Model: $$f(x) = wx + b$$

Parameters: w and b

Predicted Variable: $$f(x) or \hat{y}$$

Cost Function: $$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

The model predicts the output based on the weight w and bias b. The weight is multiplied with the input variable to show how important is the feature. Greater the weight, greater the importance of the feature variable in predicting the output. In multi variate regression, size of the plot might have greater weight and age of the house might have negative weight indicating more the age, less is the price. The bias term indicates the basic output value when input is 0.

Cost function is Mean-Squared Error (MSE), to show how different the predicted value is from the actual output. We square it to avoid the cancellation of positive and negative sum. We divide the total error by 2m to average it out. 2 is used to make the partial derivative cleaner when we apply the Gradient Descent.

Since in reality, the models are complex we use Gradient Descent Algorithm to find the values for w and b in order to minimize the cost function. 
We need to keep changing the values of w and b, it can be either increasing or decreasing the values depending on the change of cost function w.r.t to the parameters.

If the cost function is increasing while w is increasing, then we need to reduce the value of w. If the cost function is decreasing while is increasing, then we keep increasing the value of w until we get the minimum value of the cost function or until the parameters converge. This rate of change of the cost function w.r.t to b or w is obtained by using the partial derivative of J w.r.t b or w.

We also choose the learning rate (alpha) to indicate how fast or how slow we need to decrease or increase the value of the parameters. Finally we get the equation for Gradient Descent as follows:

$$\theta := \theta - \alpha \frac{\partial J(w, b)}{\partial \theta}$$

$$w := w - \alpha \frac{1}{m} \sum_{i=1}^{m} ((\hat{y}^{(i)} - y^{(i)}) x^{(i)})$$

$$b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$
