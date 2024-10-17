Self-Attentive Sequential Recommendation

transformer + 序列推荐。具体来说，采用自注意力机制来对用户的历史行为信息建模，提取更为有价值的信息。最后将得到的信息分别与所有的物品embedding内容做内积，根据相关性的大小排序、筛选，得到Top-k个推荐。

具体来说，在每个时间步骤，SASRec 都会尝试识别哪些项目与用户的操作历史记录“相关”，并使用它们来预测下一个项目。

![image-20241017212859322](./Self-Attentive Sequential Recommendation/image-20241017212859322.png)

#### model



