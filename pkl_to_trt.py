import torch
from torch2trt import torch2trt
from net import Net

# 加载原始 PyTorch 模型
model = Net().eval().cuda()
model.load_state_dict(torch.load('models/current_policy_batch300_2025-06-27_14-03-44.pkl'))

# 创建 dummy input
dummy_input = torch.randn(1, 15, 10, 9).cuda()

# 转换为 TensorRT 模型
print("🔄 正在将模型转换为 TensorRT 格式...")
trt_model = torch2trt(model, [dummy_input], fp16_mode=True)

# 保存 TensorRT 模型
torch.save(trt_model.state_dict(), 'models/current_policy_batch300_2025-06-27_14-03-44.trt')
print("✅ TensorRT 模型已保存")
