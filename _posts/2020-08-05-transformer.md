---
layout: post
title:  "Another Annotated Transformer"
date:   2020-08-04 19:52:13 +0100
categories: jekyll update
---
# Scaled Dot-Product Attention

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
def scaled_dot_product_attention(x, mask, inf=1e9):
    # x: (query=(B, H, N_q, d), key=(B, H, N_kv, d), value=(B, H, N_kv, d))
    # mask: (B, 1, N_q, N_kv) or (B, 1, 1, N_kv)
    dim = tf.cast(tf.shape(x.query)[-1], tf.float32)
    # (B, H, N_q, N_kv)
    # The einsum here is the same as QK^T
    alpha_term = tf.einsum('bhqd,bhkd->bhqk', x.query, x.key) / tf.sqrt(dim)
    # (B, H, N_q, N_kv)
    alpha_term = tf.where(mask, alpha_term, -inf)
    alpha = tf.nn.softmax(alpha_term)
    # (B, H, N_q, N_kv) @ (B, H, N_kv, d') -> (B, H, N_q, d')
    return alpha @ x.value
```
</div>
</div>

# Masking
> We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

# Position-wise Feed-Forward Networks

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
        self.dense1 = tf.keras.layers.Dense(hidden_dim,
                                            activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_dim)

    def __call__(self, x):
        x = self.dense2(self.dense1(x))=
        return x
```
</div>
</div>

# Encoder
> The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position- wise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512.

Let us start by building the Sublayer block shown below. One way to implement it is to implement a `ResidualBlock` module, in which the sublayer, Add & Norm and the residual connection are contained in a single block. Here sublayer is some arbitrary module passed in as input. This block will then be used in the `FeedForward` and  `MultiHeadedAttention` blocks. These are the key details:
- Sublayer (FeedForward or MultiHeaded Attention)
- Followed by LayerNorm
- With residual connection
- All layers have same output dimensions of d_model

Hint: sublayers can have had an arbitrary number of inputs for example, query, key and value and mask(s) are required for the multi-headed attention. Think about how to handle that.

<div class="collapse-Sublayer">
<div markdown="1">
```python
class ResidualBlock(tf.keras.models.Model):
    def __init__(self, sublayer, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.sublayer = sublayer
        self.dropout_layer = get_dropout_layer(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def __call__(self, x, sublayer_inputs, training):
        outputs = self.sublayer(*sublayer_inputs)
        outputs = self.dropout_layer(outputs, training=training)
        return self.layer_norm(x + outputs)
```
</div>
</div>
Now we can use this component block to construct a module `EncoderBlock`, which will be the building block of the encoder:

- Multi-Head Attention sublayer block with self-attention so input is used as key, query and value
- Followed by FeedForward sublayer block
- Encoder masking will be used in the attention block

Hint: it might be helpful to use pass in the input as a `QueryKeyValue` data structure defined earlier.

<div class="collapse-EncoderBlock">
<div markdown="1">
```python
class EncoderBlock(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, dropout=0.0):
        super(EncoderBlock, self).__init__()
        self.attn_block = ResidualBlock(
            MultiHeadAttention(dim, num_heads),
            dropout=dropout
        )
        self.ff_block = ResidualBlock(
            FeedForward(hidden_dim=ff_dim, output_dim=dim),
            dropout=dropout
        )

    def __call__(self, x, mask, training):
        out = self.attn_block(x.query, [x, mask], training=training)
        out = self.ff_block(out, [out], training=training)
        return out
```
</div>
</div>

Finally we can put together the `Encoder`, which consists of a stack of N encoder blocks. 

<div markdown="1" class="collapse-Encoder">
```python
class Encoder(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, num_blocks, dropout=0.0):
        super(Encoder, self).__init__()
        self.blocks = [
            EncoderBlock(dim, ff_dim, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ]

    def __call__(self, x, mask, training):
        inputs = QueryKeyValue(x)
        for block in self.blocks:
            x = block(inputs, mask, training=training)
            inputs = QueryKeyValue(x)
        return x
```
</div>

# Decoder

> The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

Let us write a `DecoderBlock`. The decoder block consists of the following:
- The two sublayers in the encoder block.
- An additional attention layer which has key and value inputs from the encoder.

Hint: it will be very similar to `EncoderBlock` but remember that the inputs to the final `AttentionBlock` will be different. Can you reuse `EncoderBlock`?

<div class="collapse-DecoderBlock">
<div markdown="1">
```python
class DecoderBlock(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, dropout=0.0,
                 skip_attn=False):
        super(DecoderBlock, self).__init__()
        self.skip_attn = skip_attn
        if not skip_attn:
            self.self_attn_block = ResidualBlock(
                MultiHeadAttention(dim, num_heads),
                dropout=dropout
            )
        self.memory_block = EncoderBlock(dim, ff_dim, num_heads, dropout=dropout)

    def __call__(self, x, decoder_mask, memory_mask, training):
        if not self.skip_attn:
            y = self.self_attn_block(x.query,
                                     [QueryKeyValue(x.query),
                                      decoder_mask],
                                     training=training)
            x = QueryKeyValue(y, x.key, x.value)
        out = self.memory_block(x, memory_mask, training=training)
        return out
```
</div>
</div>

The decoder network will be very similar to the encoder except that it will receive two different mask inputs, one for self-attention and the other for the encoder outputs. Since intermediate outputs are sometimes used to train auxiliary losses, considering adding the option to return all the outputs.  

<div class="collapse-Decoder">
<div markdown="1">
```python
class Decoder(tf.keras.models.Model):
    def __init__(self, dim, ff_dim, num_heads, num_blocks, dropout=0.0,
                 skip_first_attn=False):
        super(Decoder, self).__init__()
        self.blocks = [
            DecoderBlock(dim, ff_dim, num_heads, dropout=dropout,
                         skip_attn=(i == 0) and skip_first_attn)
            for i in range(num_blocks)
        ]

    def __call__(self, x, memory, decoder_mask, memory_mask,
                 training,
                 return_all=False):
        outputs = [x]
        for block in self.blocks:
            outputs.append(
                block(QueryKeyValue(outputs[-1], memory),
                      decoder_mask, memory_mask,
                      training=training)
            )
        if return_all:
            return outputs
        return [outputs[-1]]
```
</div>
</div>

