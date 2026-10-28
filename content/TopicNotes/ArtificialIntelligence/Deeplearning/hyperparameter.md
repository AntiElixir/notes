across all deep learning architectures — be it MLPs, CNNs, RNNs, LSTM etc.

These universal deep learning hyperparameters can be grouped into six major categories:
1. Model Design Parameters
	1. Number of hidden layers (depth)
	   $\propto$ data complexity and hierarchy
	2. Number of nodes (or neurons) per layer (width)
	   $\propto$ complex patterns and interactions in the data
2. Internal Computation Parameters
	1. [[#Activation function]]
	2. [[#Weight initialization]]
3. Data Handling and Data Sampling Parameters
	1. Batch size (the amount of data used in each update cycle)
	   $\propto$ stability, computational efficiency, but less generalization
	2. Data shuffling
	   overall good, but never on sequential or time-series data!
4. Optimization and Learning Parameters
	1. Learning rate
	   $\propto$ speed, but may overshoot the optimal point
	2. [[#Optimizer choice]]
5. Training Control and Stopping Parameters
	1. [[#Loss function]]
	2. Number of epochs
	   $\propto$ accuracy, but more training time, computational cost and maybe overfitting
	3. Early stopping (stop the training process early if the model performance does not improve for a set rounds on validation dataset)
	4. [[#Evaluation metrics]]
6. Regularization Parameters
	1. Dropout rate (determines the fraction of neurons randomly dropped or ignored during each training step)
	   $\propto$ less overfitting, but may make the model struggle to capture patterns
	2. [[#L1/L2 Regularization]]


---
---

## Activation function
### 1. ReLU (Rectified Linear Unit)

$$  
f(x)=\max(0,x)  
$$

- **Range:** $[0,+\infty)$
    
- **Advantages**
    
    - Computationally efficient.
		
    - Mitigates vanishing gradients for positive inputs.
        
    - Produces sparse activations because negative inputs become zero.
        
- **Disadvantages**
    
    - **Dying ReLU:** neurons can stop learning if their inputs remain negative, since the gradient is zero.
        
    - Outputs are not zero-centered.
        
- **Typical use:** Hidden layers in deep neural networks.
    

### 2. Leaky ReLU

$$  
f(x)=  
\begin{cases}  
x, & x\geq 0\\ 
\alpha x, & x<0  
\end{cases}  
$$

Here, $\alpha$ is a small positive constant.

- **Range:** $(-\infty,+\infty)$
    
- **Advantages**
    
    - Preserves a nonzero gradient for negative inputs.
        
    - Reduces the risk of dying ReLU.
        
    - Computationally inexpensive.
        
- **Disadvantages**
    
    - Negative activations are no longer zero, reducing activation sparsity.
        
    - Performance depends on the choice of $\alpha$.
        
    - Does not consistently outperform ReLU.
        
- **Typical use:** Hidden layers when dying ReLU is a concern.
    

### 3. ELU (Exponential Linear Unit)

$$  
f(x)=  
\begin{cases}  
x, & x\geq 0\\  
\alpha(e^x-1), & x<0  
\end{cases}  
$$

Here, $\alpha>0$ controls the saturation value for negative inputs.

- **Range:** $(-\alpha,+\infty)$
    
- **Advantages**
    
    - Allows negative outputs, which can make activations closer to zero-centered.
        
    - Provides nonzero gradients for finite negative inputs.
        
    - Reduces the risk of dying ReLU.
        
- **Disadvantages**
    
    - Exponential computation is more expensive than ReLU.
        
    - Introduces an additional hyperparameter, $\alpha$.
        
    - Does not consistently outperform ReLU.
        
- **Typical use:** Hidden layers when negative activations are beneficial.
    

### 4. Sigmoid

$$  
\sigma(x)=\frac{1}{1+e^{-x}}  
$$

- **Range:** $(0,1)$
    
- **Derivative:**
    

$$  
\sigma'(x)=\sigma(x)(1-\sigma(x))  
$$

- **Advantages**
    
    - Smooth and differentiable.
        
    - Maps real-valued inputs to $(0,1)$, making it useful for binary classification probabilities.
        
- **Disadvantages**
    
    - **Vanishing gradients:** when $x$ is very positive or negative, the derivative approaches zero.
        
    - Outputs are not zero-centered.
        
    - Can slow training in deep networks.
        
- **Typical use:** Output layer for binary classification; also used in gating mechanisms.
    

### 5. Tanh (Hyperbolic Tangent)

$$  
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}  
$$

- **Range:** $(-1,1)$
    
- **Derivative:**
    

$$  
\tanh'(x)=1-\tanh^2(x)  
$$

- **Advantages**
    
    - Zero-centered outputs.
        
    - Represents both positive and negative signals.
        
    - Often preferable to Sigmoid in hidden layers when centered activations are useful.
        
- **Disadvantages**
    
    - Suffers from vanishing gradients when inputs have large absolute values.
        
    - More computationally expensive than ReLU.
        
- **Typical use:** Hidden layers in some shallow networks and recurrent neural networks.
    

### 6. Softmax

For a vector of logits $\mathbf{z}=(z_1,\ldots,z_K)$:
$$
\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}  
$$

- **Range:** Each output $p_i\in(0,1)$, with $\sum_{i=1}^{K}p_i=1$.
    
- **Advantages**
    
    - Converts logits into a probability distribution over mutually exclusive classes.
        
    - Works naturally with categorical cross-entropy loss.
        
- **Disadvantages**
    
    - Outputs are coupled: changing one logit affects all class probabilities.
        
    - Direct computation of exponentials can cause numerical overflow or underflow; subtracting $\max_j z_j$ before exponentiation improves numerical stability.
        
    - Generally unsuitable for independent multi-label classification, where classes can co-occur.
        
- **Typical use:** Output layer for multi-class, single-label classification.
    

### 7. Linear (Identity Function)

$$  
f(x)=x  
$$

- **Range:** $(-\infty,+\infty)$
    
- **Derivative:** $f'(x)=1$.
    
- **Advantages**
    
    - Computationally inexpensive.
        
    - Does not introduce activation-induced gradient saturation.
        
    - Allows unrestricted real-valued outputs.
        
- **Disadvantages**
    
    - Introduces no nonlinearity; stacking linear layers alone is equivalent to a single linear transformation.
        
    - Cannot independently model nonlinear relationships.
        
- **Typical use:** Output layer for regression.
    

### Quick Comparison

|Function|Output range|Main advantage|Main disadvantage|
|---|---|---|---|
|ReLU|$[0,\infty)$|Efficient; positive-side gradient does not vanish|Dying ReLU|
|Leaky ReLU|$\mathbb{R}$|Nonzero negative-side gradient|Less sparse|
|ELU|$(-\alpha,\infty)$|Smooth negative branch|More expensive|
|Sigmoid|$(0,1)$|Probability-like output|Vanishing gradients|
|Tanh|$(-1,1)$|Zero-centered output|Vanishing gradients|
|Softmax|Probability simplex|Multi-class probabilities|Coupled outputs|
|Linear|$\mathbb{R}$|Unrestricted output|No nonlinearity|

### Practical Selection

- **Hidden layers:** ReLU as a common starting point; Leaky ReLU or ELU when negative-side gradient flow is important.
    
- **Binary classification:** Sigmoid for a single output probability.
    
- **Multi-class, single-label classification:** Softmax across classes.
    
- **Multi-label classification:** Independent Sigmoid outputs.
    
- **Regression:** Linear output for unrestricted real-valued predictions.
    

**Core idea:** Activation functions introduce nonlinearity or constrain the output range. Without nonlinear activations, a deep network composed only of linear transformations remains equivalent to a single linear transformation.

---
---

## Weight initialization
### 1. Purpose

**Weight Initialization** determines the initial values of a neural network's weights before training.

A good initialization should:

- Break symmetry so that different neurons can learn different features.
    
- Keep activation and gradient magnitudes within a reasonable range.
    
- Reduce vanishing gradients and exploding gradients.
    

For a layer:

$$  
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b}  
$$

where $\mathbf{W}$ is the weight matrix and $\mathbf{b}$ is the bias vector.

**Key distinction:** Weights generally need asymmetric initialization; biases can usually be initialized to zero.

### 2. Constant Initialization

All weights are initialized to the same constant $c$.

$$  
W_{ij}=c  
$$

Examples: zero initialization, one initialization.

- **Algorithm:** Fill the entire weight tensor with $c$.
    
- **Advantages:** Simple and computationally inexpensive.
    
- **Disadvantages:** Identical weights cause neurons in the same layer to receive identical gradients and remain identical during training. This is the _symmetry problem_.
    
- **Use:** Generally avoid for ordinary neural-network weight matrices. Zero initialization is commonly used for biases.
    

### 3. Random Initialization

Weights are sampled independently from a random distribution.

#### Uniform distribution

$$  
W_{ij}\sim\mathcal{U}(a,b)  
$$

#### Normal distribution

$$  
W_{ij}\sim\mathcal{N}(\mu,\sigma^2)  
$$

Here, $\mu$ is the mean and $\sigma^2$ is the variance.

- **Algorithm:** Sample each weight independently from a specified distribution.
    
- **Advantages:** Breaks symmetry and allows neurons to learn different features.
    
- **Disadvantages:** Arbitrary scales can cause activations and gradients to vanish or explode in deep networks.
    
- **Use:** A general initialization strategy, but the distribution and its scale should be chosen carefully.
    

**Note:** Xavier and He initialization are also random initialization methods. Their key difference is that they choose the distribution's scale based on layer dimensions and activation behavior.

### 4. Xavier / Glorot Initialization

Designed to keep the variance of activations and gradients reasonably stable across layers. Commonly used with Sigmoid and Tanh.

Let:

- $n_{\text{in}}$: number of input units (_fan-in_).
    
- $n_{\text{out}}$: number of output units (_fan-out_).
    

#### Xavier Uniform

$$  
W_{ij}\sim\mathcal{U}(-a,a)  
$$

$$  
a=\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}  
$$

#### Xavier Normal

$$  
W_{ij}\sim\mathcal{N}(0,\sigma^2)  
$$

$$  
\sigma=\sqrt{\frac{2}{n_{\text{in}}+n_{\text{out}}}}  
$$

- **Algorithm:** Calculate the scale from fan-in and fan-out, then sample the weights from the corresponding distribution.
    
- **Advantages:** Balances forward activation variance and backward gradient variance; improves training stability.
    
- **Disadvantages:** Usually less suitable for ReLU networks because ReLU suppresses negative pre-activations.
    
- **Use:** A common choice for Tanh and Sigmoid networks.
    

### 5. He / Kaiming Initialization

Designed for ReLU and related activations. Because ReLU sets negative inputs to zero, He initialization uses a larger variance than standard Xavier initialization when fan-in and fan-out are comparable.

#### He Normal

$$  
W_{ij}\sim\mathcal{N}(0,\sigma^2)  
$$

$$  
\sigma=\sqrt{\frac{2}{n_{\text{in}}}}  
$$

#### He Uniform

$$  
W_{ij}\sim\mathcal{U}(-a,a)  
$$

$$  
a=\sqrt{\frac{6}{n_{\text{in}}}}  
$$

- **Algorithm:** Scale the random weights according to fan-in and the activation function.
    
- **Advantages:** Helps preserve forward activation variance in ReLU networks; supports stable training in deep networks.
    
- **Disadvantages:** The scaling depends on activation behavior and the chosen fan mode. It is not universally optimal for every activation function.
    
- **Use:** A standard starting point for ReLU and Leaky ReLU networks.
    

### 6. Orthogonal Initialization

Initializes the weight matrix as an orthogonal or semi-orthogonal matrix.

For a square matrix:

$$  
\mathbf{W}^{\mathsf{T}}\mathbf{W}=\mathbf{I}  
$$

For rectangular matrices, the corresponding rows or columns can be orthonormal.

#### Common algorithm

1. Generate a random matrix $\mathbf{A}$.
    
2. Apply QR decomposition:
    

$$  
\mathbf{A}=\mathbf{Q}\mathbf{R}  
$$

3. Use $\mathbf{Q}$, optionally multiplied by a scaling factor, as the initial weight matrix.
    

- **Advantages:** Helps preserve signal magnitude through linear transformations; can improve gradient propagation in deep or recurrent networks.
    
- **Disadvantages:** More computationally expensive to generate than simple random weights; orthogonality alone does not prevent all gradient problems in nonlinear networks.
    
- **Use:** Particularly useful for recurrent networks and some deep architectures.
    

### 7. Identity Initialization

Initializes the weight matrix to the identity matrix.

$$  
\mathbf{W}=\mathbf{I}  
$$

For example:

$$  
\mathbf{I}=  
\begin{bmatrix}  
1&0&0\\  
0&1&0\\  
0&0&1  
\end{bmatrix}  
$$

Therefore, for a linear transformation without bias:

$$  
\mathbf{z}=\mathbf{I}\mathbf{x}=\mathbf{x}  
$$

The input is initially passed through unchanged.

- **Algorithm:** Set diagonal entries to $1$ and off-diagonal entries to $0$.
    
- **Advantages:** Preserves the input through the initial linear transformation; can help preserve information in suitable recurrent or specially designed networks.
    
- **Disadvantages:** Does not break symmetry between identical neurons; generally unsuitable as a universal initialization for ordinary feedforward layers.
    
- **Use:** Specific recurrent architectures or layers where preserving the initial signal is desirable. The identity matrix requires a square weight matrix in its standard form.
    

### 8. Quick Comparison

|Method|Main principle|Typical use|
|---|---|---|
|Constant|Assign the same value to every weight|Bias initialization; generally avoid for weights|
|Random|Sample weights from a distribution|General baseline with an appropriate scale|
|Xavier / Glorot|Scale using fan-in and fan-out|Tanh, Sigmoid|
|He / Kaiming|Scale using fan-in and ReLU behavior|ReLU, Leaky ReLU|
|Orthogonal|Use orthogonal or semi-orthogonal matrices|RNNs, deep networks|
|Identity|Start with an identity matrix|Specific recurrent or signal-preserving layers|

### 9. Practical Rules

- **ReLU / Leaky ReLU:** Start with He initialization.
    
- **Tanh / Sigmoid:** Start with Xavier initialization.
    
- **RNNs:** Consider orthogonal initialization for recurrent weight matrices.
    
- **Biases:** Zero initialization is usually reasonable.
    
- **Avoid:** Initializing every weight in a standard layer to the same constant.
    

**Core idea:** Weight initialization is not about guessing good individual weights. It is about choosing an initial distribution or structure that allows information and gradients to propagate through the network effectively.

---
---

## Optimizer choice

### 1. Core Concept

**Optimizer** 决定如何利用计算得到的梯度更新模型参数，以最小化 Loss Function。

$$  
\theta_{t+1}=\theta_t-\eta g_t  
$$

- $\theta_t$：当前模型参数
    
- $g_t$：当前梯度估计
    
- $\eta$：Learning Rate
    

不同优化算法的主要区别在于：是否利用历史梯度、是否调整各参数的有效学习率，以及如何计算更新方向。

### 2. Gradient Estimation Strategy

BGD、SGD、MBGD 主要区别在于计算梯度时使用的样本数量，而不是参数更新规则本身。

|Method|Batch Size|Characteristics|
|---|---|---|
|Batch Gradient Descent (BGD)|Entire dataset|梯度估计稳定，但每次更新计算成本高|
|Stochastic Gradient Descent (SGD)|1|更新频繁，梯度噪声大|
|Mini-Batch Gradient Descent (MBGD)|A subset of data|在计算效率与梯度噪声之间折中|

现代深度学习通常采用 Mini-Batch Training。SGD 也常被用作优化器的名称，即使实际 Batch Size 大于 1。

### 3. Optimization Algorithms

#### 3.1 Momentum

**Core idea:** Accumulate past gradients using an exponentially weighted moving average.

$$  
v_t=\beta v_{t-1}+g_t  
$$

$$  
\theta_{t+1}=\theta_t-\eta v_t  
$$

- 优点：沿梯度长期一致的方向加速；减少更新震荡；缓解梯度噪声的影响。
    
- 缺点：增加 Momentum Coefficient $\beta$；设置过大可能导致 Overshooting 或振荡。
    

#### 3.2 Nesterov Accelerated Gradient (NAG)

**Core idea:** Evaluate the gradient at a look-ahead position estimated using momentum.

- 与 Momentum 的区别：Momentum 根据当前位置的梯度更新；NAG 在预估的前进位置计算梯度，再修正更新。
    
- 优点：有助于更及时地修正更新方向，改善部分优化问题中的收敛表现。
    
- 缺点：仍需调整 Momentum Coefficient；效果取决于损失曲面和超参数。
    

#### 3.3 Adagrad (Adaptive Gradient Algorithm)

**Core idea:** Adapt the learning rate of each parameter using its accumulated squared gradients.

Accumulated squared gradients:

$$
G_t = G_{t-1} + g_t^2
$$

Parameter update:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} g_t
$$

- 优点：为不同参数分配不同的有效学习率；适合 Sparse Gradients，例如部分 NLP 和 Embedding 任务。
- 缺点：历史梯度平方持续累积，有效学习率可能不断缩小，导致后期学习停滞。

#### 3.4 RMSprop (Root Mean Square Propagation)

**Core idea:** Replace Adagrad's cumulative squared gradients with an exponential moving average.

Moving average of squared gradients:

$$
s_t = \rho s_{t-1} + (1 - \rho) g_t^2
$$

Parameter update:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} g_t
$$

- 优点：重点考虑近期梯度，避免有效学习率因历史梯度持续累积而过度缩小。
- 缺点：需要调整 Learning Rate 和 Decay Rate $\rho$；并非所有任务上都优于 SGD 或其他优化器。

#### 3.5 Adam (Adaptive Moment Estimation)

**Core idea:** Combine Momentum and adaptive learning rates using estimates of the first and second moments of gradients.

First moment estimate:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

Second raw moment estimate:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

Bias correction:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Parameter update:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

- 优点：结合动量与逐参数自适应步长；通常具有较好的训练效率；适用于多种深度学习任务。
- 缺点：仍需调节 Learning Rate 等参数；在某些任务中，泛化表现可能不如经过充分调优的 SGD。

### 4. Comparison

|Algorithm|Uses Past Gradients|Adaptive Per-Parameter Learning Rate|Key Idea|
|---|---|---|---|
|SGD|No|No|根据当前梯度更新|
|Momentum|Yes|No|累积梯度，形成惯性|
|NAG|Yes|No|利用 Look-ahead Gradient 修正方向|
|Adagrad|Yes|Yes|累积历史梯度平方|
|RMSprop|Yes|Yes|对近期梯度平方进行指数平均|
|Adam|Yes|Yes|结合一阶矩与二阶矩估计|

### 5. Practical Selection

- **General deep learning:** Adam 是常见的起点。
    
- **Sparse Gradients:** 可以考虑 Adagrad 或 Adam。
    
- **Sequential Models:** RMSprop 或 Adam 均可尝试，但应根据具体任务验证。
    
- **Generalization-sensitive tasks:** 可以比较 Adam 与 Momentum SGD 的验证集表现。
    

以上是经验性起点，不是适用于所有任务的固定规则。应根据 Validation Loss、Generalization、Convergence Speed 和计算成本进行选择。

### 6. Key Distinction

- **Batch Size** 决定每次梯度估计使用多少样本。
    
- **Optimizer** 决定如何利用梯度更新参数。
    
- **Learning Rate** 控制参数更新的整体尺度。
    

三者相互影响，但不是同一个概念。Mini-batch training 可以与 Momentum、RMSprop、Adam 等优化器组合使用。


## Loss Functions

### 1. Mean Squared Error ([[MSE]])
$$L_{\mathrm{MSE}}$$
### 2. Mean Absolute Error ([[MAE]])
$$  
L_{\mathrm{MAE}} 
$$
### 3. Binary Cross-Entropy (BCE)

二元交叉熵：用于二分类。

$$  
L_{\mathrm{BCE}}=-\frac{1}{N}\sum_{i=1}^{N}  
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]  
$$

- $y_i\in{0,1}$：真实标签；$p_i$：预测为正类的概率。
    
- **特点：** 通常搭配 Sigmoid，惩罚高置信度的错误预测。
    

### 4. Categorical Cross-Entropy (CCE)

分类交叉熵：用于多分类，标签采用 One-hot Encoding (用一个向量表示类别，只有真实类别对应的位置为 `1`，其他位置均为 `0` )。

$$  
L_{\mathrm{CCE}}=-\frac{1}{N}\sum_{i=1}^{N}  
\sum_{k=1}^{C}y_{ik}\log p_{ik}  
$$

- $C$：类别数；$y_{ik}$：真实标签；$p_{ik}$：预测概率。
    
- **特点：** 通常搭配 Softmax。
    

### 5. Sparse Categorical Cross-Entropy (SCCE)

稀疏分类交叉熵：用于多分类，标签直接使用整数类别索引。

$$  
L_{\mathrm{SCCE}}=-\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}  
$$

- $y_i$：真实类别索引；$p_{i,y_i}$：真实类别的预测概率。
    
- **特点：** 与 CCE 数学本质相同，区别在于标签表示方式。
    

### 6. Huber Loss

结合 MSE 与 MAE：小误差采用平方惩罚，大误差采用线性惩罚。

$$  
L_{\delta}(e)=  
\begin{cases}  
\frac{1}{2}e^2, & |e|\leq\delta\\  
\delta\left(|e|-\frac{1}{2}\delta\right), & |e|>\delta  
\end{cases}  
$$

其中 $e=\hat{y}-y$，$\delta>0$ 为切换阈值。

- **Use:** Regression。
    
- **特点：** 小误差区域梯度平滑，大误差区域对异常值更稳健。
    

### 7. Comparison

|Loss|Task|Key characteristic|
|---|---|---|
|MSE|Regression|大误差惩罚强|
|MAE|Regression|对异常值稳健|
|BCE|Binary classification|二分类概率预测|
|CCE|Multi-class classification|One-hot 标签|
|SCCE|Multi-class classification|整数类别标签|
|Huber|Regression|兼顾平滑性与稳健性|

### 8. Selection

- **Regression:** MSE；异常值较多时考虑 MAE 或 Huber。
    
- **Binary classification:** BCE。
    
- **Multi-class classification:** CCE 或 SCCE，根据标签格式选择。

## Evaluation metrics
### Classification Metrics:
see details in [[ML#Confusion Matrix（混淆矩阵）|confusion matrix]]

- **Accuracy**: It is the percentage of correct predictions by the model. It is best for balanced datasets.
- **Precision**: It measures how confident we can be that a positive prediction by the model is actually correct (true positives among predicted positives). It is important when ***false positives are costly***.
- **Recall**: It is the proportion of positives correctly identified by the model (true positives among actual positives). It is important when ***missing positives is costly***.
- **F1 Score**: It is the harmonic mean of precision and recall. It balances false positives and negatives making it useful for imbalanced datasets.
- **AUC-ROC**: It measures how well a model can separate positive and negative classes, regardless of the decision threshold. Higher AUC means better distinction, which is especially useful for imbalanced binary classification.
### Regression Metrics:

- **Mean Squared Error (MSE)**: It calculates average squared difference between predicted and true values and penalizes large errors.
- **Mean Absolute Error (MAE)**: It measures average absolute difference between predicted and actual values and it is more robust to outliers.

## L1/L2 Regularization

Regularization adds a penalty to the loss function to reduce overfitting.

### 1. L1 Regularization (Lasso)

$$  
L_{\mathrm{total}}=L_{\mathrm{data}}+\lambda\sum_{j=1}^{d}|w_j|  
$$

- **Penalty:** Absolute values of weights.
    
- **Effect:** Encourages sparsity; some weights may become exactly zero.
    
- **Use:** Feature selection and simpler models.
    

### 2. L2 Regularization (Ridge / Weight Decay)

$$  
L_{\mathrm{total}}=L_{\mathrm{data}}+\lambda\sum_{j=1}^{d}w_j^2  
$$

- **Penalty:** Squared weights.
    
- **Effect:** Discourages large weights; generally shrinks weights toward zero without making them exactly zero.
    
- **Use:** Controlling model complexity and reducing overfitting.
    

### 3. Regularization Strength ($\lambda$)

|Strength|Effect|
|---|---|
|High $\lambda$|Stronger penalty; simpler model, but may underfit|
|Low $\lambda$|Weaker penalty; more flexibility, but may overfit|

### 4. Comparison

|             | L1                | L2               |
| ----------- | ----------------- | ---------------- |
| Penalty     | $\|w\|$           | $w^2$            |
| Main effect | Sparse weights    | Small weights    |
| Exact zeros | More likely       | Generally not    |
| Common use  | Feature selection | Weight shrinkage |

**Key idea:** L1 encourages sparsity; L2 discourages excessively large weights. Both balance fitting the training data against model complexity.

Here, $L_{\mathrm{data}}$ is the original task loss, $w_j$ is a model weight, and $\lambda\geq 0$ controls regularization strength.