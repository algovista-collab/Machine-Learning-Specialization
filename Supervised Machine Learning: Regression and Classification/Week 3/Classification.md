In **Classification**, especially when output values can take only **2 values** like true or false (categories or classes), it is called **Binary Classification**.  

If for a question, the answer is yes, then we can call it a **positive class**, and the opposite is called the **negative class**.  

If we try to predict this using **Linear Regression**, whose output can take values more than just 2, then we set a **threshold**.  

If the threshold is 0.5:

$$
\text{if } f_{wb}(x) < 0.5 \text{, classify as positive class; otherwise, classify as negative class}
$$

But as we add a new training example, the model doesn't predict properly. Hence, we enter into **Logistic Regression**, which is a model designed to output either 0 or 1.  

---

First, let's see the exponential function:

$$
e^x
$$

- As \(x \to 0\), \(e^x \to 1\)  
- As \(x \to \infty\), \(e^x \to \infty\)  
- As \(x \to -\infty\), \(e^x \to 0\)  

The function 

$$
\frac{1}{1 + e^x}
$$

outputs values only from 0 to 1. \(x\) can be between \(-\infty\) to \(+\infty\). For any value in this range, the function takes values between 0 and 1.  

Example:  
- \(x = -2 \to \frac{1}{1 + e^{-2}} \approx 0.119\)  
- \(x = 0 \to \frac{1}{1 + e^{0}} = 0.5\)  
- \(x = 2 \to \frac{1}{1 + e^{-2}} \approx 0.881\)  

The model objective to output values between 0 and 1 is satisfied, but it should be **monotonic**, meaning as \(x\) increases, the function should also increase. Hence we consider:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

This maps \(z \in (-\infty, \infty)\) to \(p \in (0,1)\).  

The functions are mirrored by the property:

$$
\sigma(-z) = 1 - \sigma(z)
$$

This is mostly a **convention**.
