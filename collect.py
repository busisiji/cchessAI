import csv
import pickle

import numpy as np

import cchess
import time
import os
import copy
import argparse
from collections import deque
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

class CollectPipeline:
    def __init__(self, init_model=None, pid=0, port=8000):
        self.board = cchess.Board()
        self.game = Game(self.board, port=port)
        self.pid = pid if pid != 0 else os.getpid()  # 使用传入的 pid 或当前进程 pid

        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = 400  # 每次移动的模拟次数
        self.c_puct = 5
        self.buffer_size = 10000  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.completed_games = 0  # 完整棋局计数器
        self.move_times = []  # 所有对弈步骤的耗时记录

        # 加载模型
        try:
            self.policy_value_net = PolicyValueNet(model_file=MODEL_PATH)
            self.log("已加载最新模型")
        except Exception as e:
            self.policy_value_net = PolicyValueNet()
            self.log(f"已加载初始模型")

        self.mcts_ai = MCTS_AI(
            self.policy_value_net.policy_value_fn,
            c_puct=5,
            n_playout=400,
            is_selfplay=True,
        )

        # 自动恢复断点
        self.load_checkpoint()

        # 创建日志目录
        self.log_dir = "game_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def load_checkpoint(self):
        """尝试从磁盘加载已有的 data_buffer"""
        if os.path.exists(DATA_BUFFER_PATH):
            try:
                with open(DATA_BUFFER_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.data_buffer = deque(data["data_buffer"], maxlen=self.buffer_size)
                    self.completed_games = data.get("completed_games", 0)
                    self.log(f"成功加载历史数据，已有 {len(self.data_buffer)} 条样本，完成 {self.completed_games} 局")
            except Exception as e:
                self.log(f"加载断点失败：{e}")

    def save_checkpoint(self):
        """将当前数据缓冲区和计数器保存到磁盘"""
        data = {
            "data_buffer": list(self.data_buffer),
            "completed_games": self.completed_games,
        }
        with open(DATA_BUFFER_PATH, "wb") as f:
            pickle.dump(data, f)

    def collect_data(self, n_games=1, is_shown=False):
        for i in range(n_games):
            try:
                start_time = time.time()
                winner, play_data = self.game.start_self_play(self.mcts_ai, is_shown=is_shown)
                play_data = list(play_data)
                self.episode_len = len(play_data)

                # 只有当棋局正常结束时才保存
                if winner is not None and len(play_data) > 0:
                    move_times = []
                    moves = []

                    # 开始记录每一步
                    log_file = os.path.join(self.log_dir, f"game_{self.completed_games + 1}.csv")
                    fieldnames = ["step", "from_pos", "to_pos", "uci", "elapsed_sec"]

                    with open(log_file, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        board = cchess.Board()
                        prev_time = time.time()
                        for idx, (state, mcts_prob, _) in enumerate(play_data):
                            curr_time = time.time()
                            elapsed = curr_time - prev_time
                            prev_time = curr_time
                            move_times.append(elapsed)

                            move = board.last_move
                            if move:
                                ucci = move.to_ucci()
                                from_pos = cchess.parse_ucci(ucci)[0]
                                to_pos = cchess.parse_ucci(ucci)[1]
                                moves.append((from_pos, to_pos))

                                # 写入CSV
                                writer.writerow({
                                    "step": idx + 1,
                                    "from_pos": from_pos,
                                    "to_pos": to_pos,
                                    "uci": ucci,
                                    "elapsed_sec": round(elapsed, 3)
                                })

                                # 分析慢速步骤
                                if elapsed > 2.0:
                                    self.log(f"⚠️ 第 {idx + 1} 步耗时 {elapsed:.2f}s，可能较慢")

                        self.log(f"棋局记录已保存至 {log_file}")

                    # 统计本局信息
                    total_time = time.time() - start_time
                    num_moves = len(play_data)
                    avg_time = np.mean(move_times)
                    max_time = max(move_times) if move_times else 0

                    # 显示总结
                    self.log(f"\n第 {self.completed_games + 1} 局完成")
                    self.log(f"总步数：{num_moves}")
                    self.log(f"总耗时：{total_time:.2f} 秒")
                    self.log(f"平均每步耗时：{avg_time:.2f} 秒")
                    self.log(f"最大单步耗时：{max_time:.2f} 秒")
                    self.log(f"胜负方：{'红方' if winner == 1 else '黑方' if winner == -1 else '和棋'}")
                    self.log(f"前三步移动坐标：{moves[:3]}\n")

                    # 数据处理与保存
                    play_data = self.mirror_data(play_data)
                    self.data_buffer.extend(play_data)
                    self.completed_games += 1
                    self.save_checkpoint()  # 实时保存

                else:
                    self.log("检测到非完整棋局，已跳过")
            except KeyboardInterrupt:
                self.log("棋局被中断，未保存该局数据")
                continue

        return self.completed_games

    def run(self, queue=None, is_shown=False):
        try:
            while True:
                iters = self.collect_data(is_shown=is_shown)
                data_to_save = list(self.data_buffer)
                if queue:
                    queue.put(data_to_save)  # 异步写入队列
                self.log(f"batch i: {iters}, 总完整局数: {self.completed_games}")
        except KeyboardInterrupt:
            self.log("程序退出，最终完整局数: %d" % self.completed_games)

    # 新增：封装日志函数，支持 PID 显示
    def log(self, message):
        print(f"[{time.strftime('%H:%M:%S')}][PID={self.pid}] {message}")



if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="收集中国象棋自对弈数据")
    parser.add_argument(
        "--show", action="store_true", default=False, help="是否显示棋盘对弈过程"
    )
    args = parser.parse_args()
    # 创建数据收集管道实例
    collecting_pipeline = CollectPipeline(init_model="current_policy.pkl")
    collecting_pipeline.run(is_shown=args.show)
