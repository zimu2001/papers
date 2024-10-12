Self-Attentive Sequential Recommendation

类 transformer + 序列推荐。具体来说，采用自注意力机制来对用户的历史行为信息建模，提取更为有价值的信息。最后将得到的信息分别与所有的物品embedding内容做内积，根据相关性的大小排序、筛选，得到Top-k个推荐。

