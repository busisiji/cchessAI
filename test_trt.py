# 测试 TensorRT 模型是否可用

import torch
from torch2trt import TRTModule

trt_model = TRTModule()
trt_model.load_state_dict(torch.load('models/current_policy_batch300_2025-06-27_14-03-44.trt'))

dummy_input = torch.randn(1, 15, 10, 9).cuda()
with torch.no_grad():
    policy, value = trt_model(dummy_input)

print("Policy Output Shape:", policy.shape)
print("Value Output Shape:", value.shape)
