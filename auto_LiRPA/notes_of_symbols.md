`ndim`: (int) 一个张量的总维数
`root_name`: (list) 计算图每层命名，其中第 0 个为输入层
`root`: (list) 同上，长度为 checkpoint 的长度
`_modules`: (orderDict) 节点类型，如输入层是 BoundInput
`ub`: (tensor) 在 forward propagation 中是偏置的上界，对应文章 17 页的 d 上界
`node.linear`: (object) 保存 LinearBound 对象
`prev_dim_in`: (int) 为了处理一般计算图中的分块输入，对于 DNN 等于输入层维数
`x.uw`: (tensor) 对应文章 17 页的 diag(alpha)W 上界
`upper_k`: (tensor) ReLU 抽象的上界斜率
`_add_linear()`: 更新 `w_out` 和 `b_out`，即更新 `lw` 和 `lb` 等
`_clear_and_set_new()`: 中的 l 是网络一层
`forward()`: 同网络结构定义中的 `forward`，定义了邻层间计算方式
`w_out = self.lw`: (w_out = (tensor)) 是引用了 `lw`，而不是复制值
`mask_pos`: (tensor) 线性化时遮掉下界大于 0 的部分
`node`: (object) 指一层节点而非单个节点
