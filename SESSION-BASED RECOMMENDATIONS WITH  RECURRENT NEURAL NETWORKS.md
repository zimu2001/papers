SESSION-BASED RECOMMENDATIONS WITH  RECURRENT NEURAL NETWORKS

#### 0

首篇RNN + RS，即GRU4Rec。

具体来说，将用户进入网站时点击的第一个项目视为 RNN 的初始输入，然后希望根据该初始输入查询模型以获取推荐。用户的每次连续点击都会产生一个输出（推荐），该输出取决于之前的所有点击。

与传统 nlp 任务相比，序列推荐有两个主要区别。一是序列稀疏，二是loss的设计。

#### -1

传统会话推荐方法是基于项目相似度的，即根据已有信息