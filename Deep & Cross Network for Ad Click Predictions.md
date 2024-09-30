Deep & Cross Network for Ad Click Predictions

#### 0

DNN能够有效地学习features interactions。但是，1. 交互是隐式的，2. DNN不一定学习到所有类型。

FM是二阶的特征交叉，可以显示的学习特征，但是在高级交叉时，FM参数会指数增长。

本文使用DNN+cross network，可以更有效地学习有限度数的特征交互。（通过cross net的层数控制）

DCN由多个层组成，其中最高程度的交互由层深度决定。每一层都会基于现有的交互产生更高阶的交互，并保留来自先前层的交互。

#### DCN

![image-20240929235535798](D:\wzm\paper\papers\Deep & Cross Network for Ad Click Predictions\image-20240929235535798.png)

##### embedding

密集特征直接作为向量，稀疏特征通过嵌入矩阵（学习得到）转化为低纬嵌入向量，

![image-20240930000558004](D:\wzm\paper\papers\Deep & Cross Network for Ad Click Predictions\image-20240930000558004.png)

二者首尾相接得到输入向量

![image-20240930000631064](D:\wzm\paper\papers\Deep & Cross Network for Ad Click Predictions\image-20240930000631064.png)

##### cross net

cross net关在在于以有效的方式应用显示特征交叉，每层有

![image-20240930085221436](D:\wzm\paper\papers\Deep & Cross Network for Ad Click Predictions\image-20240930085221436.png)

xi 是层间的传递向量，wi,bi 是第 i 层的权重和偏置。f 即特征交叉函数，其拟合了xi+1和 xi 的残差。上式的可视化如下。

![image-20240930090001935](D:\wzm\paper\papers\Deep & Cross Network for Ad Click Predictions\image-20240930090001935.png)

可见，我们通过矩阵乘法实现了特征的组合，L 层cross net可以表示 L+1 阶的特征交互。在计算feature crossing时，可以先计算后两个向量乘积，可以大幅降低运算量。

cross net类的代码实现如下。

```python
class CrossNet(nn.Module):
    //初始化层，laynum即croosnet总层数
    def __init__(self, in_features, layer_num=2, seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        //权重初始化，因为是一维向量使用了parameterList
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        //偏置初始化
        self.to(device)
 
    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        //前向传播，x0是batch*d,变为batch*d*1
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            //先xl * w
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = torch.squeeze(x_l, dim=2)
        //batch*d*1,变为batch*d
        return x_l
```

每层cross network的学习参数只有两个向量，因此相比于DNN其复杂度可以忽略不计，这得益于使用了向量叉积避免了存储大型矩阵。但也正因为其参数较少，模型的表达受限，还需要添加额外的DNN层。

##### DNN

仍然是常规的全连接层

![image-20240930093739032](D:\wzm\paper\papers\Deep & Cross Network for Ad Click Predictions\image-20240930093739032.png)

激活函数使用ReLU

##### 组合层

最后一层组合CN和DNN的输出，

![image-20240930094922731](D:\wzm\paper\papers\Deep & Cross Network for Ad Click Predictions\image-20240930094922731.png)

其中 xL1 ∈ Rd 、 hL2 ∈ Rm 分别是交叉网络和深度网络的输出，wlogits ∈ R(d+m) 是组合层的权重向量，σ (x) 为sigmoid函数。

##### loss

![image-20240930095012988](D:\wzm\paper\papers\Deep & Cross Network for Ad Click Predictions\image-20240930095012988.png)

loss设计为交叉熵+L2正则，其中 pi 是根据公式 5 计算的概率，yi 是真实标签，N 是输入总数，λ 是 L2 正则化参数。

联合训练两个网络，因为这使得每个单独的网络在训练过程中都能了解其他网络。

#### 实验

略
