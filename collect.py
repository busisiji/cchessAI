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
from game import Game
from parameters import C_PUCT, PLAYOUT, BUFFER_SIZE, DATA_BUFFER_PATH, MODEL_PATH
from tools import move_id2move_action, move_action2move_id, zip_state_mcts_prob, flip


class CollectPipeline:
    def __init__(self, init_model=None, pid=0, port=8000):
        self.board = cchess.Board()
        self.game = Game(self.board, port=port)
        self.pid = pid if pid != 0 else os.getpid()

        self.buffer_size = BUFFER_SIZE
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.completed_games = 0
        self.move_times = []

        try:
            self.policy_value_net = PolicyValueNet(model_file=MODEL_PATH)
            self.log("已加载最新模型")
        except Exception as e:
            self.policy_value_net = PolicyValueNet()
            self.log(f"已加载初始模型")

        self.load_checkpoint()

        self.log_dir = "game_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


    def load_checkpoint(self):
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
        data = {
            "data_buffer": list(self.data_buffer),
            "completed_games": self.completed_games,
        }
        with open(DATA_BUFFER_PATH, "wb") as f:
            pickle.dump(data, f)



    def collect_data(self, n_games=1, is_shown=False, batch_write_interval=5):
        games_collected = 0
        batch_data = []

        for i in range(n_games):
            try:
                start_time = time.time()

                winner, play_data = self.game.start_self_play(self.policy_value_net, is_shown=is_shown and (i % 10 == 0),pid=self.pid)
                play_data = list(play_data)
                self.episode_len = len(play_data)

                if winner is not None and len(play_data) > 0:
                    move_times = []
                    moves = []

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

                            last_move = board.peek()
                            if last_move:
                                ucci = last_move.to_ucci()
                                from_pos = cchess.parse_ucci(ucci)[0]
                                to_pos = cchess.parse_ucci(ucci)[1]
                                moves.append((from_pos, to_pos))



                                if idx % 20 == 0 or idx == len(play_data) - 1:
                                    writer.writerow({
                                        "step": idx + 1,
                                        "from_pos": from_pos,
                                        "to_pos": to_pos,
                                        "uci": ucci,
                                        "elapsed_sec": round(elapsed, 3)
                                    })

                                if elapsed > 2.0 and idx % 20 == 0:
                                    self.log(f"⚠️ 第 {idx + 1} 步耗时 {elapsed:.2f}s，可能较慢")

                        self.log(f"棋局记录已保存至 {log_file}")

                    total_time = time.time() - start_time
                    num_moves = len(play_data)
                    avg_time = np.mean(move_times)
                    max_time = max(move_times) if move_times else 0

                    self.log(f"\n第 {self.completed_games + 1} 局完成")
                    self.log(f"总步数：{num_moves}")
                    self.log(f"总耗时：{total_time:.2f} 秒")
                    self.log(f"平均每步耗时：{avg_time:.2f} 秒")
                    self.log(f"最大单步耗时：{max_time:.2f} 秒")
                    self.log(f"胜负方：{'红方' if winner == 1 else '黑方' if winner == -1 else '和棋'}")
                    self.log(f"前三步移动坐标：{moves[:3]}\n")

                    play_data = self.mirror_data(play_data)
                    batch_data.extend(play_data)
                    games_collected += 1
                    self.completed_games += 1

                    if games_collected % batch_write_interval == 0:
                        self.data_buffer.extend(batch_data)
                        self.save_checkpoint()
                        batch_data.clear()
                        self.log(f"已批量写入 {games_collected} 局数据到磁盘")

                else:
                    self.log("检测到非完整棋局，已跳过")

            except KeyboardInterrupt:
                self.log("棋局被中断，未保存该局数据")
                continue

        if batch_data:
            self.data_buffer.extend(batch_data)
            self.save_checkpoint()
            self.log(f"剩余 {len(batch_data)} 局数据已写入磁盘")

        return self.completed_games

    def mirror_data(self, play_data):
        mirrored = []
        for state, mcts_prob, winner_z in play_data:
            mirrored_state = flip(state)
            mirrored_mcts_prob = np.flipud(mcts_prob.reshape(10, 9)).flatten()
            mirrored_winner_z = -winner_z
            mirrored.append((mirrored_state, mirrored_mcts_prob, mirrored_winner_z))
        return mirrored + play_data

    def run(self, queue=None, is_shown=False):
        try:
            while True:
                iters = self.collect_data(is_shown=is_shown)
                if queue:
                    queue.put(list(self.data_buffer))
                self.log(f"batch i: {iters}, 总完整局数: {self.completed_games}")
        except KeyboardInterrupt:
            self.log("程序退出，最终完整局数: %d" % self.completed_games)

    def log(self, message):
        print(f"[{time.strftime('%H:%M:%S')}][PID={self.pid}] {message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收集中国象棋自对弈数据")
    parser.add_argument("--show", action="store_true", default=False, help="是否显示棋盘对弈过程")
    args = parser.parse_args()

    collecting_pipeline = CollectPipeline(init_model="current_policy.pkl")
    collecting_pipeline.run(is_shown=args.show)
