import pickle
import cchess
import time
import os
import copy
import argparse
from collections import deque
from multiprocessing import Pool, cpu_count
from net import PolicyValueNet
from mcts import MCTS_AI
from game import Game
from parameters import C_PUCT, PLAYOUT, BUFFER_SIZE, DATA_BUFFER_PATH, MODEL_PATH
from tools import (
    move_id2move_action,
    move_action2move_id,
    zip_state_mcts_prob,
    flip,
)


# 定义整个对弈收集数据流程
class CollectPipeline:
    def __init__(self, init_model=None):
        self.board = cchess.Board()
        self.game = Game(self.board)
        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = PLAYOUT  # 每次移动的模拟次数
        self.c_puct = C_PUCT
        self.buffer_size = BUFFER_SIZE  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.mcts_ai = None
        self.policy_value_net = None  # 延迟初始化

    # 从主体加载模型
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

    def collect_single_game(self, is_shown=False):
        """单个进程执行一次对弈"""
        self.load_model()  # 从本体处加载最新模型
        winner, play_data = self.game.start_self_play(
            self.mcts_ai, is_shown=is_shown
        )  # 开始自我对弈
        play_data = list(play_data)  # 转换为列表
        play_data = self.mirror_data(play_data)  # 扩充数据
        return play_data

    def collect_data(self, n_games=1, is_shown=False):
        """使用多进程收集自我对弈的数据"""
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(
                self.collect_single_game, [(is_shown,) for _ in range(n_games)]
            )
            for play_data in results:
                self.data_buffer.extend(play_data)

        self.iters += 1
        data_dict = {"data_buffer": self.data_buffer, "iters": self.iters}
        with open(DATA_BUFFER_PATH, "wb") as data_file:
            pickle.dump(data_dict, data_file)

        print(f"[{time.strftime('%H:%M:%S')}] batch i: {self.iters}, 数据已保存")
        return self.iters

    def run(self, n_games_per_iter=1, is_shown=False):
        """开始收集数据"""
        try:
            while True:
                iters = self.collect_data(n_games=n_games_per_iter, is_shown=is_shown)
                print(
                    f"[{time.strftime('%H:%M:%S')}] batch i: {iters}, 当前共收集了 {len(self.data_buffer)} 条数据"
                )
        except KeyboardInterrupt:
            print(f"\n\r[{time.strftime('%H:%M:%S')}] quit")


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="收集中国象棋自对弈数据")
    parser.add_argument(
        "--show", action="store_true", default=False, help="是否显示棋盘对弈过程"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=4,
        help="每次迭代中并行运行的对局数量（默认：CPU核心数）",
    )
    args = parser.parse_args()

    # 创建数据收集管道实例
    collecting_pipeline = CollectPipeline(init_model="current_policy.pkl")

    # 动态设置对局数量，默认为 CPU 核心数
    n_games = args.games if args.games > 0 else cpu_count()

    collecting_pipeline.run(n_games_per_iter=n_games, is_shown=args.show)
