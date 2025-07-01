import torch
import cchess
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast
from tools import move_action2move_id, decode_board

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")


# 构建残差块
class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_filters, num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.conv1_bn = nn.BatchNorm2d(
            num_filters,
        )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.conv2_bn = nn.BatchNorm2d(
            num_filters,
        )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)


# 构建骨干网络, 输入: N, 15, 10, 9 --> N, C, H, W
class Net(nn.Module):
    def __init__(
        self, num_channels=256, num_res_blocks=40
    ):  # 40 ResBlock为AlphaZero的默认值
        super(Net, self).__init__()
        # 全局特征
        # self.global_conv = nn.Conv2D(in_channels=15, out_channels=512, kernel_size=(10, 9))
        # self.global_bn = nn.BatchNorm2D(512)
        # 初始化特征
        self.conv_block = nn.Conv2d(
            15, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.conv_block_bn = nn.BatchNorm2d(
            num_channels,
        )
        self.conv_block_act = nn.ReLU()
        # 残差块抽取特征
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)]
        )
        # 策略头
        self.policy_conv = nn.Conv2d(
            num_channels, 16, kernel_size=(1, 1), stride=(1, 1)
        )
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * 10 * 9, 2086)
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 10 * 9, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 公共头
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        # policy = torch.reshape(policy, [-1, 16 * 10 * 9])
        batch_size = x.size(0)
        policy = torch.reshape(policy, [batch_size, 16 * 10 * 9])

        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        # value = torch.reshape(value, [-1, 8 * 10 * 9])
        value = torch.reshape(value, [batch_size, 8 * 10 * 9])


        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return policy, value


class PolicyValueNet(object):
    def __init__(self, model_file=None, use_gpu=True, device="cuda"):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3
        self.device = device
        self.stream = (
            torch.cuda.Stream() if self.use_gpu and torch.cuda.is_available() else None
        )

        # 初始化 TensorRT 模型
        self.trt_model = None
        self.onnx_session = None

        if model_file:
            if model_file.endswith('.trt'):
                # 加载 TensorRT 模型
                from torch2trt import TRTModule
                self.trt_model = TRTModule()
                self.trt_model.load_state_dict(torch.load(model_file))
                print("✅ 成功加载 TensorRT 模型")
            elif model_file.endswith('.pkl'):
                # 加载 PyTorch 模型（兼容旧代码）
                self.policy_value_net = Net().to(self.device)
                self.policy_value_net.load_state_dict(torch.load(model_file))
                print("✅ 成功加载 PyTorch 模型")
            elif model_file.endswith('.onnx'):
                # 加载 ONNX 模型（兼容旧代码）
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(model_file)
                print("✅ 成功加载 ONNX 模型")
            else:
                raise ValueError("❌ 不支持的模型格式，请提供 .pkl 或 .trt 文件")

    def policy_value_fn(self, board):
        legal_positions = [
            move_action2move_id[cchess.Move.uci(move)]
            for move in list(board.legal_moves)
        ]

        current_state = decode_board(board)
        current_state = np.ascontiguousarray(current_state.reshape(-1, 15, 10, 9)).astype("float32")

        if self.trt_model is not None:
            # print("使用 TensorRT 模型进行预测")
            input_tensor = torch.from_numpy(current_state).to(self.device)
            with torch.inference_mode():
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        log_act_probs, value = self.trt_model(input_tensor)
                    torch.cuda.current_stream().wait_stream(self.stream)
                else:
                    log_act_probs, value = self.trt_model(input_tensor)
                log_act_probs, value = log_act_probs.cpu(), value.cpu()
        elif self.onnx_session is not None:
            # print("使用 ONNX 模型进行预测")
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: current_state})
            log_act_probs, value = outputs[0], outputs[1]
        else:
            # print("使用 PyTorch 模型进行预测")
            self.policy_value_net.eval()
            with torch.no_grad():
                input_tensor = torch.from_numpy(current_state).to(self.device)
                log_act_probs, value = self.policy_value_net(input_tensor)
            log_act_probs, value = log_act_probs.cpu(), value.cpu()

        act_probs = np.exp(log_act_probs.flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])

        return act_probs, value.item()
