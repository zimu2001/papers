Deep Residual Learning for Image Recognition

残差网络神作。来自何凯明。只是简单的阅读了一下思想，后续再补充细节问题。

#### 0

CNN网络层数的增加会带来更好的性能，因其每个层都可以提取到不同level的特征，因此越深提取的特征越丰富。

但是简单的增加层数，会导致梯度消失或爆炸（激活函数的值变化很小，其对参数的梯度也很小，导致参数更新会很慢）一种可行的解决方案是BN。

然而随着网络层数的增加，训练集上的准确率会饱和甚至下降，这称为**饱和**。

虽然在数学上，高维网络的解空间向量包括低纬网络的解空间向量，但是SGD往往是找到局部最优解，而高纬网络的解空间更加复杂，这就导致计算上的偏差。

#### -0.5

恒等映射的概念：H(x) = x。

如果深层次的网络的部分层可以被训练为恒等映射，可以转化为浅层网络。但是直接让一些层去拟合一个潜在的恒等映射函数比较困难，这可能就是深层网络难以训练的原因。但是，如果把网络设计为H(x) = F(x) + x,如下图。我们可以转换为学习一个残差函数F(x) = H(x) - x. 只要F(x)=0，就构成了一个恒等映射H(x) = x. 而且，拟合残差肯定更加容易。

![image-20240928235327185](D:\wzm\paper\papers\Deep Residual Learning for Image Recognition\image-20240928235327185.png)

#### 0

ResNet提出了两种mapping：一种是identity mapping，指的就是上图中”弯弯的曲线”，另一种residual mapping，指的就是除了”弯弯的曲线“那部分，所以最后的输出是 y=F(x)+x

identity mapping顾名思义，就是指本身，也就是公式中的x，而residual mapping指的是“差”，也就是y−x，所以残差指的就是F(x)部分。

#### 公式推导

来自知乎。

![img](https://pic3.zhimg.com/v2-c6d32cf9e9d1480367973991bb37379c_r.jpg)



