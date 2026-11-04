# Lab 02 — Network design and chain-rule derivation

## 1. Architecture and dimensions

For batch size B and hidden width W, the network is:

`X (B,64) → affine → ReLU → affine → ReLU → affine → softmax → P (B,10)`

Both hidden layers have the same width. The two experiments use W=16 and W=64.

| Parameter | General shape | W=16 | W=64 |
|---|---|---|---|
| W1 | (64, W) | (64, 16) | (64, 64) |
| b1 | (W,) | (16,) | (64,) |
| W2 | (W, W) | (16, 16) | (64, 64) |
| b2 | (W,) | (16,) | (64,) |
| W3 | (W, 10) | (16, 10) | (64, 10) |
| b3 | (10,) | (10,) | (10,) |

Weights are sampled independently from a normal distribution with mean 0 and
standard deviation 0.01. All biases start at zero. The total number of trainable
parameters is W² + 76W + 10: **1,482** for W=16 and **8,970** for W=64.

## 2. Forward propagation

With A0 = X:

$$
Z_1 = XW_1+b_1,\quad A_1=\max(0,Z_1)
$$

$$
Z_2 = A_1W_2+b_2,\quad A_2=\max(0,Z_2)
$$

$$
Z_3 = A_2W_3+b_3,\quad
P_{ic}=\frac{\exp(Z_{3,ic}-m_i)}{\sum_{k=1}^{10}\exp(Z_{3,ik}-m_i)},
\quad m_i=\max_k Z_{3,ik}.
$$

Biases broadcast across the batch dimension. Subtracting each row maximum
preserves softmax probabilities and prevents exponential overflow.

## 3. Cross-entropy and the softmax derivative

For one-hot labels Y and a batch of B images:

$$
L=-\frac{1}{B}\sum_{i=1}^{B}\sum_{c=1}^{10}Y_{ic}\log P_{ic}.
$$

An untrained network gives approximately P_ic=1/10, so L≈ln(10)=2.302585.
The implementation calculates log probabilities directly from logits using
log-softmax, rather than adding an epsilon inside the logarithm. This keeps
the loss and its gradient mathematically consistent and handles extreme logits.

For one sample, let z be its output logits and p=softmax(z). The softmax Jacobian is:

$$
\frac{\partial p_c}{\partial z_k}=p_c(\delta_{ck}-p_k).
$$

Since ∂ℓ/∂p_c=−y_c/p_c and ∑_c y_c=1, the chain rule gives:

$$
\frac{\partial\ell}{\partial z_k}
=-\sum_c\frac{y_c}{p_c}\,p_c(\delta_{ck}-p_k)
=-y_k+p_k\sum_c y_c=p_k-y_k.
$$

The mean loss introduces the batch factor once:

$$
D_3=\frac{\partial L}{\partial Z_3}=\frac{P-Y}{B}.
$$

## 4. Backpropagation through affine and ReLU layers

For any affine layer Z=AW+b, a small change satisfies
dZ=dA·W+A·dW+db. Therefore an upstream derivative D=∂L/∂Z gives
∂L/∂W=AᵀD, ∂L/∂b=∑_i D_i and ∂L/∂A=DWᵀ.
ReLU has derivative 1 for positive inputs and 0 for negative inputs;
the implementation chooses 0 at exactly zero.

Applying these rules from the output toward the input:

$$
\nabla_{W_3}L=A_2^TD_3,\quad
\nabla_{b_3}L=\sum_iD_{3,i},\quad
\frac{\partial L}{\partial A_2}=D_3W_3^T
$$

$$
D_2=(D_3W_3^T)\odot\mathbf{1}[Z_2>0],\quad
\nabla_{W_2}L=A_1^TD_2,\quad
\nabla_{b_2}L=\sum_iD_{2,i}
$$

$$
D_1=(D_2W_2^T)\odot\mathbf{1}[Z_1>0],\quad
\nabla_{W_1}L=X^TD_1,\quad
\nabla_{b_1}L=\sum_iD_{1,i}.
$$

Here ⊙ denotes elementwise multiplication. The gradient shapes match their
parameter shapes. No second division by B is needed because D3 already includes it.
The actual batch size must be used, including the final batch of 9 images.

All gradients are calculated using the current parameters before SGD updates:

$$
\theta\leftarrow\theta-\eta\nabla_\theta L.
$$

## 5. Numerical gradient checking

For each selected scalar weight θ_j, central differences estimate the slope:

$$
g_j^{\mathrm{num}}=\frac{L(\theta_j+\epsilon)-L(\theta_j-\epsilon)}{2\epsilon},
\qquad\epsilon=10^{-5}.
$$

The original scalar is restored after both evaluations. The reported relative error is:

$$
e_j=\frac{|g_j^{\mathrm{analytic}}-g_j^{\mathrm{num}}|}
{|g_j^{\mathrm{analytic}}|+|g_j^{\mathrm{num}}|}.
$$

If both slopes are exactly zero, the error is defined as zero. Ten distinct
random weights are checked per width, with four from W1 and three each from
W2 and W3. At least nine must satisfy e_j<10⁻⁶. Zero gradients are retained
and are visible in the CSV; points are never resampled to hide failures.

To avoid roundoff when subtracting losses near ln(10), the numerical check
evaluates the same objective after subtracting that constant. Let
u_ic=z_ic−mean_c(z_ic). Then:

$$
L-\ln(10)=\frac{1}{B}\sum_i\left[
\log\left(1+\frac{1}{10}\sum_c\operatorname{expm1}(u_{ic})\right)
-\sum_cY_{ic}u_{ic}\right].
$$

`expm1(u)` calculates exp(u)−1 accurately near zero; `log1p(v)` calculates
log(1+v) accurately near zero. A constant has zero derivative, so the slopes
and the required tolerance are unchanged. This expression is used for checking
at initialization only; training uses the general stable log-softmax loss.

ReLU is not differentiable at zero. A finite difference that crosses a ReLU
boundary may disagree with the derivative at its center. Both widths use a
fixed batch and random seed, and all measured errors are saved for inspection.

## 6. Prediction

The trained network performs the same forward pass. For each image, the predicted
digit is `argmax(P, axis=1)`. Accuracy is the proportion of predictions equal
to the integer labels recovered from the one-hot rows. No backpropagation or
parameter updates occur during prediction.
