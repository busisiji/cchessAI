# 子进程采集数据
import pickle
from threading import Thread, Lock

import cchess
import time
import os
import copy
import argparse
from collections import deque
from net import PolicyValueNet
from mcts import MCTS_AI
from game_multi_thread import Game
from parameters import C_PUCT, PLAYOUT, BUFFER_SIZE, DATA_BUFFER_PATH, MODEL_PATH
from tools import (
    move_id2move_action,
    move_action2move_id,
    zip_state_mcts_prob,
    flip,
)
from multiprocessing import Manager


# 定义整个对弈收集数据流程
class CollectPipeline:
    def __init__(self, init_model=None, pid=0, port=8000, log_pipe=None, enable_logging=True):
        self.board = cchess.Board()
        self.game = Game(self.board, port=port)
        self.pid = pid if pid != 0 else os.getpid()

        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = PLAYOUT  # 每次移动的模拟次数
        self.c_puct = C_PUCT
        self.buffer_size = BUFFER_SIZE  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.mcts_ai = None
        self.policy_value_net = None  # 延迟初始化

        # 数据保存路径
        self.sub_data_path = f"{DATA_BUFFER_PATH}_{self.pid}"

    def log(self, message):
        """公共日志接口：直接打印"""
        print(f"[{time.strftime('%H:%M:%S')}] [PID={self.pid}] {message}")

    def load_model(self):
        if self.policy_value_net is None:  # 仅初始化一次
            try:
                self.policy_value_net = PolicyValueNet(model_file=MODEL_PATH)
                print(f"[{time.strftime('%H:%M:%S')}] 已加载最新模型")
            except:
                self.policy_value_net = PolicyValueNet()
                print(f"[{time.strftime('%H:%M:%S')}] 已加载初始模型")
            self.mcts_ai = MCTS_AI(
                self.policy_value_net.policy_value_fn,
                c_puct=self.c_puct,
                n_playout=self.n_playout,
                is_selfplay=True,
            )

    def mirror_data(self, play_data):
        """左右对称变换，扩充数据集一倍，加速一倍训练速度"""
        mirror_data = []
        # 棋盘形状 [15, 10, 9], 走子概率，赢家
        for state, mcts_prob, winner in play_data:
            # 原始数据
            mirror_data.append(zip_state_mcts_prob((state, mcts_prob, winner)))
            # 水平翻转后的数据
            state_flip = state.transpose([1, 2, 0])[:, ::-1, :].transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                # 水平翻转后，走子概率也需要翻转
                mcts_prob_flip[i] = mcts_prob[
                    move_action2move_id[flip(move_id2move_action[i])]
                ]
            mirror_data.append(
                zip_state_mcts_prob((state_flip, mcts_prob_flip, winner))
            )
        return mirror_data

    def collect_data(self, n_games=1, is_shown=False):
        """收集自我对弈的数据"""
        while True:
            start_time = time.time()
            self.load_model()  # 从本体处加载最新模型
            winner, play_data = self.game.start_self_play(
                self.policy_value_net, is_shown=is_shown, pid=self.pid
            )  # 开始自我对弈

            play_data = list(play_data)  # 转换为列表
            self.episode_len = len(play_data)  # 记录每盘对局长度
            # 增加数据
            play_data = self.mirror_data(play_data)
            self.data_buffer.extend(play_data)

            # 将数据写入子文件中
            self.iters += 1
            self.save_to_sub_file()

            # 打印日志信息
            self.log(f"第 {self.iters} 局完成")
            self.log(f"总步数：{len(play_data)}")
            self.log(f"总耗时：{time.time() - start_time:.2f} 秒")
            self.log(f"胜负方：{'红方' if winner == 1 else '黑方' if winner == -1 else '和棋'}")

        return self.iters

    def save_to_sub_file(self):
        """将当前 data_buffer 写入对应的子文件"""
        with open(self.sub_data_path, "ab") as f:
            pickle.dump({"data_buffer": self.data_buffer, "iters": self.iters}, f)

    def run(self, is_shown=False):
        """开始收集数据"""
        try:
            while True:
                iters = self.collect_data(is_shown=is_shown)
                print(
                    f"[{time.strftime('%H:%M:%S')}] batch i: {iters}, episode_len: {self.episode_len}"
                )
        except KeyboardInterrupt:
            print(f"\n\r[{time.strftime('%H:%M:%S')}] quit")


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="收集中国象棋自对弈数据")
    parser.add_argument(
        "--show", action="store_true", default=True, help="是否显示棋盘对弈过程"
    )
    args = parser.parse_args()
    # 创建数据收集管道实例
    collecting_pipeline = CollectPipeline(init_model="current_policy.pkl")
    collecting_pipeline.run(is_shown=args.show)
