Personalized top-N sequential recommendation via convolutional sequence embedding.

#### 0

caser,CNN+序列推荐，即利用滤波器捕捉item嵌入矩阵的特征。这篇文章很喜欢举例子，令人感动。

```
绪论中有一个形象的比喻，uer 的general behavier 可以是更喜欢苹果产品相比于三星产品，与之相对的 sequential patterns代表了用户短期和动态的行为，比如说他刚买了iphone。基于general behavier 的推荐不会推荐user手机配件，但是关注时序后可以推荐手机配件。
```

序列推荐旨在给定Su （item序列），考虑sequential patterns和 general perferences，为每个user推荐一个item list来最大化满足他的需求，

现有的模型并未考虑顺序模式中的跳跃行为，如c

![image-20241011191733456](Personalized top-N sequential recommendation via convolutional sequence embedding/image-20241011191733456.png)

```
例如，一个游客依次在机场、酒店、餐厅、酒吧和景点进行签到。虽然机场和酒店的签到并不紧邻景点的签到，但它们与后者有很强的关联性。另一方面，餐厅或酒吧的签到对景点的签到影响较小（因为它们不一定会发生）。
```

#### 1

为了验证跳跃行为的影响，使用一种规规则来衡量序列强度，

对于序列

![image-20241011205622097](Personalized top-N sequential recommendation via convolutional sequence embedding/image-20241011205622097.png)

属于规则X→Y，

定义支持度计数sup(XY)，是指在序列中按规则顺序出现X和Y的次数，置信度sup(XY)，是指在出现X的序列中，Y在X之后发生的百分比。通过将右侧改为Sut+1或Sut+2，该规则也可以捕捉到一或两步跳跃的影响。

在 Movielens和Gowalla 进行筛选置信度大于50的规则

![image-20241011211019356](Personalized top-N sequential recommendation via convolutional sequence embedding/image-20241011211019356.png)

可见考虑跳跃行为是合理的。

#### model

