#### 1 Hermite矩阵及正定性

设Hermite矩阵![img](https://rain-oplat.xuetangx.com/ue_i/20230728/3029dbc9-c24f-4960-929f-e8b94322ec7c.jpg) ,则矩阵A 为(  )Hermite矩阵。



![image-20240918112105298](C:\Users\G304\AppData\Roaming\Typora\typora-user-images\image-20240918112105298.png)

![image-20240918112112088](C:\Users\G304\AppData\Roaming\Typora\typora-user-images\image-20240918112112088.png)

Hermite矩阵的性质：

- Hermite矩阵的对角元素必须是实数；
- Hermite矩阵的特征值都是实数； 
- Hermite矩阵任意两个不同特征值所对应的特征向量正交；
- 对正整数 k，A^k 也是Hermite矩阵；
- 若A可逆，则 A^-1 也是Hermite矩阵；

讨论正定：

- ![image-20240918112837442](C:\Users\G304\AppData\Roaming\Typora\typora-user-images\image-20240918112837442.png)

判断正定

- 定义
- n个特征值均为正数
- 顺序主子式均为正数
- 所有主子式全部大于0
- 存在 n 阶非奇异下三角矩阵 L，使得A =  LL^H（该分解称为Cholesky分解）
- 存在 n 阶非奇异矩阵，使得 A=B^HB
- 存在n nn阶非奇异Hermite矩阵 A=S^2

本题用顺序主子式判断。

#### 2 秩的判断

已知![img](https://rain-oplat.xuetangx.com/ue_i/20230728/dead6587-a12a-4bb6-ac43-724758d9fc2e.jpg) 和集合![img](https://rain-oplat.xuetangx.com/ue_i/20230728/f4c4e47c-1d2b-4612-ab7d-7d6c54671242.jpg) ,则SA (   )  R 上的线性空间,若SA 是R 上的线性空间,则![img](https://rain-oplat.xuetangx.com/ue_i/20230728/d17c01ab-b7ea-4ccc-8fc8-a96d4ad66c40.jpg) (   )



A的0，1次幂可以表示任何的幂次。秩为2。

#### 3 秩的判断

已知![img](https://rain-oplat.xuetangx.com/ue_i/20230728/6c3cbf4a-a27a-4e49-8506-15dea2c25ad8.jpg) ,![img](https://rain-oplat.xuetangx.com/ue_i/20230728/c1f8c1a3-0e54-43d2-8e85-f315b527117e.jpg) ,则子空间SA={B∈C5×5|AB=0} 的维数是()

AB=0，B的每列其实都是AX=0的解，假设A的秩=r. 那么AX=0最多有n-r个线性无关的解。所以B的秩≤n-r. r(A)+r(B)≤r+n-r=n. 

B是零矩阵的时候秩是0.不是零矩阵的时候，只要B保证最少有一列向量是A的Ax=0的解就行，其他列可以为零向量，但是最多只能有n-r个非零列向量是Ax=0的解。

矩阵B的每一列都是AX=0的解，需要两个无关的向量，一个任意的矩阵B，需要2*5=**10**个基来表示。

要注意题干是矩阵组成的集合构成的维数，并不是矩阵的维数。

#### 4 秩的判断

设V={X∈R10×10|Xa=0,a=[1,1,…,1]T} ,则![img](https://rain-oplat.xuetangx.com/ue_i/20230728/3eadc1ca-ec61-433b-8218-5dffb3194800.jpg) 的维数是()

与上题雷同

#### 5 秩和核？

若矩阵![img](https://rain-oplat.xuetangx.com/ue_i/20230728/f67423f1-fb39-41d9-9a64-1599132f8a01.jpg) ,则下列说法错误的是()

N（A）是Ax=0的解集，R（A）是列向量线性组合的集合（Column，疑似ppt有误）

- 行空间 C(AT)和零空间N(A)正交且为正交补
- 列空间C(A)和左零空间N(AT)正交且为正交补

#### 6 秩的判断？

#### 7 矩阵补空间？

#### 8 线性空间？

a。对

b。不满足数乘

c。x0 取 高位零向量。高维矩阵数乘可以为0？

d。不满足数乘和加法。

#### 9 ?

#### 10 ？

零向量

不可以

对

对

#### 11 秩的判断

#### 12 正交投影？

#### 13 

#### 54 

 设![img](https://rain-oplat.xuetangx.com/ue_i/20230728/3efa4b46-b537-4aff-8679-d06fd3987298.jpg) ,![img](https://rain-oplat.xuetangx.com/ue_i/20230728/1d760e68-fdbb-4bc9-959b-eee71b91c162.jpg) ,若非齐次线性方程组![img](https://rain-oplat.xuetangx.com/ue_i/20230728/8ecb4dfa-0f8b-4482-9f2a-0325870e0243.jpg) 有解,则 在![img](https://rain-oplat.xuetangx.com/ue_i/20230728/1e76c7e3-4638-41a2-8882-343cb5642fa8.jpg) 只有![img](https://rain-oplat.xuetangx.com/ue_i/20230728/4b3f7fa3-6f22-46f6-8f9c-626915a0910d.jpg) 的一个解向量。



#### 55 维度公式？

看作九维列向量，dim即变量个数-独立约束个数。

dimu=9-1

dim w = 9-6=3

九维列向量求极大无关组吗？

#### 56 维度公式

设 R4的两个子空间为 ![img](https://rain-oplat.xuetangx.com/ue_i/20230728/7e9630de-5253-4c1e-a84d-48b5b86b7815.jpg) ![img](https://rain-oplat.xuetangx.com/ue_i/20230728/7ff4cba3-3924-4fc9-a364-3b8b8bb9e8fe.jpg) 则![img](https://rain-oplat.xuetangx.com/ue_i/20230728/68743d35-1f58-4415-9b65-cea4ad007459.jpg) ,dim![img](https://rain-oplat.xuetangx.com/ue_i/20230728/e2e529fa-75b4-4ff6-a222-3313eef8b352.jpg) 。

同59。答案正确。

#### 57 Hermite

设![img](https://rain-oplat.xuetangx.com/ue_i/20230728/4885e716-9bf5-4919-9ff8-38d1a3248a2c.jpg) 是Hermite矩阵,则a=2i ,b=i 

hermite矩阵定义见题1，

共轭对称有，a=0，b=-i

#### 59 维度公式

设 ![img](https://rain-oplat.xuetangx.com/ue_i/20230728/f7130c8b-74d0-47fd-ab39-bf0e6bf5095a.jpg) 则dim![img](https://rain-oplat.xuetangx.com/ue_i/20230728/d9a64aa2-d968-4236-814e-fbfa7b66a326.jpg) ,dim![img](https://rain-oplat.xuetangx.com/ue_i/20230728/39626d66-3992-4532-b4b1-0ac822de769f.jpg) 。

把两个矩阵空间看作是四维的列向量空间，分别求基向量。

和空间的维度=基向量组的极大线性无关组向量个数

dim（w1+w2）=3，

dimW1=dimW2=2,

交空间维度使用维度公式计算。

dim(w1 ∩ w2) = 2+2-3=1