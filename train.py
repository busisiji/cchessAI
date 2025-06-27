# 训练模型
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cchess
import random
import pickle
import time
import numpy as np
from game import Game
from collections import deque
from net import PolicyValueNet
from tools import recovery_state_mcts_prob

# 关键参数
# MODEL_PATH = models/current_policy.pkl # 模型路径
# GAME_BATCH_NUM = 30000 #训练轮数
# CHECK_FREQ=300  # 保存模型的频率

# 持续强化学习参数
# IS_CRL # 是否持续强化学习 （边下棋边训练）
# UPDATE_INTERVAL = 10 # 持续强化学习时每隔多少秒检查并更新模型

# 其他参数
# BUFFER_SIZE = 100000  # 经验池大小
# KL_TARG = 0.02 kl散度控制
# BATCH_SIZE = 1024 # 批次大小

# from torch.utils.data import DataLoader, Dataset
from parameters import (
    PLAYOUT,
    C_PUCT,
    BATCH_SIZE,
    EPOCHS,
    KL_TARG,
    BUFFER_SIZE,
    GAME_BATCH_NUM,
    UPDATE_INTERVAL,
    DATA_BUFFER_PATH,
    MODEL_PATH,
    CHECK_FREQ,
)


class TrainPipeline:
    def __init__(self, init_model=None):
        # 训练参数
        self.init_model = init_model
        self.board = cchess.Board()
        self.game = Game(self.board)
        self.n_playout = PLAYOUT  # 每次移动的模拟次数
        self.c_puct = C_PUCT
        self.learning_rate = 1e-3  # 学习率
        self.lr_multiplier = 1  # 基于KL自适应的调整学习率
        self.temp = 1.0  # 温度
        self.batch_size = BATCH_SIZE  # 批次大小
        self.epochs = EPOCHS  # 每次训练的轮数
        self.kl_targ = KL_TARG  # kl散度控制
        self.check_freq = CHECK_FREQ  # 保存模型的频率
        self.game_batch_num = GAME_BATCH_NUM  # 训练次数
        self.names = [] # 此次训练的模型
        self.train_num = 0
        self.best_win_ratio = 0.0 # 最佳胜率
        # self.best_win_ratio = 0.0
        # self.pure_mcts_playout_num = 500
        self.buffer_size = BUFFER_SIZE  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)

        os.makedirs("models", exist_ok=True)

        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=self.init_model)
                self.train_num = self.extract_batch_number(self.init_model)
                print(f"[{time.strftime('%H:%M:%S')}] 已加载上次最终模型 {self.train_num}批次训练")
            except:
                # 从零开始训练
                print(f"[{time.strftime('%H:%M:%S')}] 模型路径不存在，从零开始训练")
                self.policy_value_net = PolicyValueNet()
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 从零开始训练")
            self.policy_value_net = PolicyValueNet()
    def extract_batch_number(self,model_path):
        """
        从模型路径中提取 batch 轮数。

        参数:
            model_path (str): 模型文件路径，如 "models/current_policy_batch456_2024-04-05_10-20-30.pkl"

        返回:
            int: 提取出的轮数，如 456；若未找到则返回 None
        """
        filename = os.path.basename(model_path)
        match = re.search(r"batch(\d+)_", filename)
        if match:
            return int(match.group(1))
        else:
            return 0

    def cleanup_models(self):
        """保留 self.names 中最后十个模型，其余删除"""
        if len(self.names) <= 3:
            return

        for model_path in self.names[:-3]:
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print(f"[{time.strftime('%H:%M:%S')}] 已删除模型: {model_path}")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] 删除失败: {model_path}, 错误: {e}")

    def policy_evaluate(self, num_games=5):
        """
        使用当前策略网络与纯MCTS玩家进行对局，计算胜率 win_ratio
        num_games: 每次评估的对局数
        return: 胜率 win_ratio
        """
        from mcts import MCTS_AI
        from game import Game

        # 初始化MCTS AI玩家
        current_player = MCTS_AI(policy_value_fn=self.policy_value_net.policy_value_fn, n_playout=400)
        opponent_player = MCTS_AI(policy_value_fn=self.policy_value_net.policy_value_fn, n_playout=200)

        # 单个对局函数
        def play_game(game):
            winner, _ = game.start_play(current_player, opponent_player, is_shown=False)
            return 1 if winner == cchess.RED else 0

        win_count = 0
        with ThreadPoolExecutor(max_workers=num_games) as executor:
            futures = []
            for _ in range(num_games):
                game_copy = Game(cchess.Board())  # 创建新的棋盘实例避免冲突
                futures.append(executor.submit(play_game, game_copy))

            for future in futures:
                win_count += future.result()

        win_ratio = win_count / num_games
        return win_ratio

    def policy_update(self,i=None):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        # print(mini_batch[0][1],mini_batch[1][1])
        mini_batch = [recovery_state_mcts_prob(data) for data in mini_batch]
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype("float32")

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype("float32")

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype("float32")

        # 旧的策略，旧的价值函数
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learning_rate * self.lr_multiplier,
            )
            # 新的策略，新的价值函数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.kl_targ * 4:  # 如果KL散度很差，则提前终止
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # print(old_v.flatten(),new_v.flatten())
        explained_var_old = 1 - np.var(
            np.array(winner_batch) - old_v.flatten()
        ) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(
            np.array(winner_batch) - new_v.flatten()
        ) / np.var(np.array(winner_batch))
        self.train_num += 1
        print(
            (
                f"[{time.strftime('%H:%M:%S')}] kl:{kl:.5f},"
                f"第{self.train_num}轮"
                f"lr_multiplier:{self.lr_multiplier:.3f},"
                f"loss:{loss:.3f},"
                f"entropy:{entropy:.3f},"
                f"explained_var_old:{explained_var_old:.9f},"
                f"explained_var_new:{explained_var_new:.9f}"
            )
        )
        return loss, entropy

    def run(self):
        """开始训练"""
        try:
            while True:
                try:
                    with open(DATA_BUFFER_PATH, "rb") as data_dict:
                        data_file = pickle.load(data_dict)
                        self.data_buffer = data_file["data_buffer"]
                        self.iters = data_file["iters"]
                        del data_file
                    print(
                        f"[{time.strftime('%H:%M:%S')}] 已载入数据，缓冲区大小: {len(self.data_buffer)}, batch_size: {self.batch_size}"
                    )
                    break
                except:
                    time.sleep(10)
            for i in range(self.game_batch_num):
                print(f"[{time.strftime('%H:%M:%S')}] step i {self.iters}: ")
                win_ratio = 0.0
                if len(self.data_buffer) > self.batch_size:
                    try:
                        loss, entropy= self.policy_update(i)
                        print("current self-play batch: {},win_ratio: {}".format(i + 1, win_ratio))
                        # # 保存模型
                        # self.policy_value_net.save_model(MODEL_PATH)
                        # 清理此批次之外的模型
                        self.cleanup_models()
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] 训练失败: {e}")

                time.sleep(UPDATE_INTERVAL)  # 每10s更新一次模型

                if (i + 1) % self.check_freq == 0:
                    # win_ratio = self.policy_evaluate()
                    # self.policy_value_net.save_model('./current_policy.model')
                    # if win_ratio > self.best_win_ratio:
                    #     print(f"[{time.strftime('%H:%M:%S')}] New best policy!!!!!!!!")
                    #     self.best_win_ratio = win_ratio
                    #     # update the best_policy
                    #     self.policy_value_net.save_model('./best_policy.model')
                    #     if (self.best_win_ratio == 1.0 and
                    #             self.pure_mcts_playout_num < 5000):
                    #         self.pure_mcts_playout_num += 1000
                    #         self.best_win_ratio = 0.0
                    # print(
                    #     f"[{time.strftime('%H:%M:%S')}] current self-play batch: {i + 1}"
                    # )
                    name = f"models/current_policy_batch{self.train_num}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"  # 添加日期信息
                    os.makedirs(os.path.dirname(name), exist_ok=True)  # 确保目录存在
                    self.policy_value_net.save_model(name)
                    self.names.append(name)
        except KeyboardInterrupt:
            # 清理最后一个之外的模型
            # self.cleanup_models()
            print(f"\n\r[{time.strftime('%H:%M:%S')}] quit")


if __name__ == "__main__":
    training_pipeline = TrainPipeline(init_model="models/current_policy_batch300_2025-06-27_14-03-44.pkl")
    training_pipeline.run()
