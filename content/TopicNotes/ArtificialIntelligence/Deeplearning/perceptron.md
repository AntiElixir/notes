the foundation of deep learning ([[TopicNotes/ArtificialIntelligence/Deeplearning/README]]), [[neural_network|neural network]]
## Why?

1. **背景**：1943 年 McCulloch–Pitts 提出 neuron 的数学模型，是一个 threshold unit，$a=f\!\left(\sum_{i=1}^{n} w_i x_i-\theta\right)$，其 weights $w_i$ 与阈值 $\theta$ 需人工设定。1949 年 Hebb 提出：连接强度应随经验改变，即 learning。
2. **要解决的问题**：neuron 模型本身不会 learn。能否让机器直接从带 label 的数据中自动得到 weights，完成分类？
3. **方案**：1958 年 Rosenblatt 提出 Perceptron：保留 neuron 的计算形式，令 $w_1, w_2, \dots, w_n$ 和 $b$（$b=-\theta$）成为可学习参数，并给出 perceptron learning rule 来更新它们。
## What?

Rosenblatt 于 1958 年提出，模仿 biological neuron：接收多个输入，经过变换，产生一个输出。由五部分组成：

1. **Inputs (x₁, x₂, …, xₙ)**
   送入模型的 features。

2. **Weights (w₁, w₂, …, wₙ)**
   每个输入的重要性：越大影响越强，接近 0 则几乎无影响，为负则起反方向作用。

3. **Summation Function (Σ)**
   对 inputs 加权求和，再加 bias $b$，得到中间输出 $z$：

   $$
   z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top\mathbf{x} + b
   $$

   其中 $\mathbf{w}=[w_1, w_2, \dots, w_n]^\top$，$\mathbf{x}=[x_1, x_2, \dots, x_n]^\top$。

4. **Activation Function (f)**
   step function，把 $z$ 变成 0 或 1：

   $$
   f(z)=\begin{cases}1, & z\ge 0\\ 0, & z<0\end{cases}
   $$

5. **Output (a)**
   Perceptron 的最终决策：

   $$
   a = f(z) = f\!\left(\sum_{i=1}^{n} w_i x_i + b\right)
   $$

## Geometric Intuition

1. $z=\mathbf{w}^\top\mathbf{x}+b$ 是 inputs 的 linear combination。
2. $z=0$ 是 feature space 中的 decision boundary。2 个 inputs 时是一条直线：

   $$
   w_1x_1 + w_2x_2 + b = 0 \;\Longleftrightarrow\; x_2 = -\frac{w_1}{w_2}\,x_1 - \frac{b}{w_2}\qquad (w_2\neq 0)
   $$

   n 个 inputs 时是 hyperplane：$w_1x_1 + w_2x_2 + \dots + w_nx_n + b = 0$。
3. 边界把空间分成两侧，activation 判断样本在哪一侧，并给出 class label：

   $$
   a=\begin{cases}1, & \mathbf{w}^\top\mathbf{x}+b \ge 0\\ 0, & \mathbf{w}^\top\mathbf{x}+b < 0\end{cases}
   $$

因此 Perceptron 是 binary linear classifier，与 linear / logistic regression 同类。

## How?

目标：inputs 已知，求 $w_1, w_2, \dots, w_n$ 和 $b$，使 decision boundary 分开两类。

1. 用小随机数初始化 $w_1, w_2, \dots, w_n, b$
2. 取一个训练样本 $(\mathbf{x}, y)$，计算预测 $\hat y = f(\mathbf{w}^\top\mathbf{x}+b)$
3. 比较 $\hat y$ 与真实 label $y$，按 perceptron learning rule 更新（学习率 $\eta$）：

   $$
   \Delta=\eta\,(y-\hat y)=\begin{cases}0, & \hat y=y\ \text{（预测正确，不更新）}\\ +\eta, & y=1,\ \hat y=0\ \text{（输出偏低）}\\ -\eta, & y=0,\ \hat y=1\ \text{（输出偏高）}\end{cases}
   $$

   $$
   w_i \leftarrow w_i + \Delta\, x_i\ \ (i=1,\dots,n),\qquad b \leftarrow b + \Delta
   $$

4. 对所有样本重复 2–3；仍有 misclassification 就再来一轮，直到全部正确或达到最大 iterations

loss 是 $m$ 个训练样本中 misclassification 的个数，不可微，不同于 cross-entropy：

$$
L=\sum_{j=1}^{m}\mathbf{1}\!\left[\hat y^{(j)}\neq y^{(j)}\right]
$$

## Limitations

1. It consisted of only a single computational layer, so 只能解 linearly separable 问题（如 XOR 不行）
2. 只能做 binary classification
3. 数据非 linearly separable 时不收敛：任意 $\mathbf{w},b$ 下都有 $L>0$，权重无限更新
4. 找到的边界不一定 optimal，取决于 initialization 和样本顺序

这些问题催生了 [[MLP]]。
