# Deep-Learning

## Binary Cross-Entropy With Logits

In binary classification tasks, there are two common versions of binary cross-entropy (BCE) loss:

### 1. BCE using sigmoid output

Given:
- `y` = target label (0 or 1)
- `z` = logit (raw model output)
- `p = sigmoid(z) = 1 / (1 + exp(-z))`

Then the standard BCE is:
 
BCE(y, p) = -[ y * log(p) + (1 - y) * log(1 - p) ]

### 2. BCE with logits (numerically stable)

This form avoids computing `sigmoid(z)` explicitly:

BCEWithLogits(y, z) = max(z, 0) - z * y + log(1 + exp(-abs(z)))

Both are mathematically equivalent, but the logits version is **more stable** for large values of `z`. This is why deep learning frameworks (like PyTorch’s `BCEWithLogitsLoss`) use it internally.
