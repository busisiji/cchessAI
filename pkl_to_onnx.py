import torch
from net import Net

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化并加载模型
model = Net().eval().to(device)
model.load_state_dict(torch.load('models/current_policy_batch300_2025-06-27_14-03-44.pkl'))

# 创建 dummy input
dummy_input = torch.randn(1, 15, 10, 9).to(device)

# 导出 ONNX 模型
onnx_path = 'models/current_policy.onnx'
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,  # 存储训练参数
    opset_version=13,    # ONNX 算子集版本
    do_constant_folding=True,  # 优化常量
    input_names=['input'],     # 输入名
    output_names=['policy', 'value'],  # 输出名
    dynamic_axes={
        'input': {0: 'batch_size'},  # 动态维度
        'policy': {0: 'batch_size'},
        'value': {0: 'batch_size'}
    }
)

print(f"✅ ONNX 模型已导出至 {onnx_path}")
