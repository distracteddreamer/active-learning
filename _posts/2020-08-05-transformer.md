---
layout: post
title:  "Another Annotated Transformer"
date:   2020-08-04 19:52:13 +0100
categories: jekyll update
---
## Scaled Dot-Product Attention

>We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension d , and values of dimension d . We compute the dot products of the
k√v
query with all keys, divide each by dk, and apply a softmax function to obtain the weights on the
values.

>In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as:

It is helpful to consider the shapes of the inputs as they get transformed at each step.

>We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections.

One way to handle masking is to set the positions where `mask=False` to a negative value with large magnitude such that its softmax score is almost zero and has negligible effect on the scores of the values where `mask=True`.

<div markdown="0" class="collapse-scaled_dot_product_attention">
<div markdown="1">
```python
    def scaled_dot_product_attention(x, mask, mask_value=-1e9):
        # (B, d', N_kv)
        key_t = tf.transpose(x.key, [0, 2, 1])
        scale_factor = (1 / tf.shape(x.query)[-1])
        # (B, N_q, N_kv)
        alpha_term = scale_factor * (x.query @ key_t)
        # (B, N_q, N_kv)
        alpha = tf.nn.softmax(
            tf.where(mask, mask_value, alpha_term), 
            axis=-1)
        # (B, N_q, N_kv) @ (B, N_kv, d') -> (B, N_q, d')
        return alpha @ x.value
```
</div>
</div>

## Position-wise Feed-Forward Networks

>[E]ach of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.
FFN(x) = max(0, xW1 + b1 )W2 + b2 (2)
While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff =2048.

Let us implement a class `FeedForward`. It should be an instance of `tf.keras.models.Model` and take a single input.

Implementation details:
- Input of of size B x N x D=512
- Position-wise meaning that this is treated like a batch of B*N vectors of dimension D
- A two layer neural network:
    - Hidden dimension of 2048
    - ReLU activation after first layer
    - Output dimension of 512


<div class="collapse-FeedForward" markdown="0">
<div markdown="1">
```python
class FeedForward(tf.keras.models.Model):
    def __init__(self, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_dim)

    def __call__(self, x):
        return self.dense2(self.dense1(x))
```
</div>
</div>
