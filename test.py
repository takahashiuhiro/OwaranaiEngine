import torch
import torch.nn.functional as F

# 三个“cost”样本：1 < 2 < 3
cost = torch.tensor([1.0, 2.0, 3.0])

print("原始 cost:", cost.tolist())

# ① 直接对 +cost 做 softmax
w_pos = F.softmax(cost, dim=0)
print("\nSoftmax(+cost)  ==> ", w_pos.tolist())

# ② 先取负号，再 softmax
w_neg = F.softmax(-cost, dim=0)
print("Softmax(-cost) ==> ", w_neg.tolist())
