# Deep-Learning

Https://ut.philkr.net/deeplearning/

## :palm_tree: Binary Cross-Entropy With Logits

In binary classification tasks, there are two common versions of binary cross-entropy (BCE) loss:

### 1 BCE using sigmoid output

Given:
- `y` = target label (0 or 1)
- `z` = logit (raw model output)
- `p = sigmoid(z) = 1 / (1 + exp(-z))`

Then the standard BCE is:
 
BCE(y, p) = -[ y * log(p) + (1 - y) * log(1 - p) ]

### 2 BCE with logits (numerically stable)

This form avoids computing `sigmoid(z)` explicitly:

BCEWithLogits(y, z) = max(z, 0) - z * y + log(1 + exp(-abs(z)))

Both are mathematically equivalent, but the logits version is **more stable** for large values of `z`. This is why deep learning frameworks (like PyTorch’s `BCEWithLogitsLoss`) use it internally.

### It is okay to find local minimal, key is learning rate and initialize model right

## :palm_tree: Backward and Grad function
### we can only call backward once and on sacaler value, and call grad multiple time will sum the gradient up, so need to reset each time.

## :palm_tree: What is a layer?
### Largest computation unit that remains unchanged through out different architectures. And we often only count linear layer. aka maxinum number of linear transformation in case parallel structure exists. one of the reason being most of the case only linear layer carries parameters.

## :palm_tree: How to choose activation function?
### Avoid sigmoid since it saturates on both end and gradient becomes 0. Try ReLU with careful initialization and small learnni rate, if it fails try Leaky Relu. Use GeLU for saphisticated models.

## :palm_tree: Output representation.
### Deep network always output real values, output transformation converts them into what we want, train the network without output transformation.

## 🌴 Reduce SGD variance
### Converge faster with smaller variance
1. Minibatch
2. Momentum
