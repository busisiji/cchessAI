import pickle
import cchess
import time
import os
import copy
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# 日志打印函数，带PID标识
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [PID {os.getpid()}] {msg}")

# 定义整个对弈收集数据流程
class CollectPipeline:
    def __init__(self, init_model=None):
        self.board = cchess.Board()
        self.game = Game(self.board)
        # 对弈参数
        self.n_playout = PLAYOUT
        self.c_puct = C_PUCT
        self.buffer_size = BUFFER_SIZE
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0

        # 提前加载模型
        try:
            self.policy_value_net = PolicyValueNet(model_file=MODEL_PATH)
            log("已加载最新模型")
        except Exception as e:
            log(f"模型加载失败: {e}，使用初始模型")
            self.policy_value_net = PolicyValueNet()

        # 初始化 MCTS AI
        self.mcts_ai = MCTS_AI(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=True,
        )

    def mirror_data(self, play_data):
        """左右对称变换，扩充数据集一倍"""
        mirror_data = []
        for state, mcts_prob, winner in play_data:
            mirror_data.append(zip_state_mcts_prob((state, mcts_prob, winner)))
            state_flip = state.transpose([1, 2, 0])[:, ::-1, :].transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                mcts_prob_flip[i] = mcts_prob[
                    move_action2move_id[flip(move_id2move_action[i])]
                ]
            mirror_data.append(zip_state_mcts_prob((state_flip, mcts_prob_flip, winner)))
        return mirror_data

    def collect_single_game(self, is_shown=False, max_retry=3):
        """单个进程执行一次对弈，最多重试 max_retry 次"""
        pid = os.getpid()
        for retry in range(max_retry):
            try:
                log("开始对弈")
                winner, play_data = self.game.start_self_play(self.mcts_ai, is_shown=is_shown)
                play_data = list(play_data)
                play_data = self.mirror_data(play_data)
                log(f"完成对弈，获得 {len(play_data)} 条数据")
                return play_data
            except Exception as e:
                log(f"第 {retry+1}/{max_retry} 次对弈失败: {e}")
                time.sleep(1)
        log("多次失败，跳过本局")
        return []

    def collect_data(self, n_games=1, is_shown=False):
        start_time = time.time()
        log(f"开始并行采集 {n_games} 局数据")

        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_tasks = [executor.submit(self.collect_single_game, is_shown) for _ in range(n_games)]
            for future in as_completed(future_tasks):
                try:
                    play_data = future.result()
                    results.append(play_data)
                except Exception as e:
                    log(f"某个对弈任务异常完成: {e}")

        for play_data in results:
            self.data_buffer.extend(play_data)

        self.iters += 1
        data_dict = {"data_buffer": self.data_buffer, "iters": self.iters}
        with open(DATA_BUFFER_PATH, "wb") as data_file:
            pickle.dump(data_dict, data_file)

        duration = round(time.time() - start_time, 2)
        log(f"批量采集完成，共 {len(self.data_buffer)} 条数据，耗时 {duration} 秒")
        return self.iters

    def run(self, n_games_per_iter=1, is_shown=False):
        """开始收集数据"""
        try:
            while True:
                start_total = time.time()
                iters = self.collect_data(n_games=n_games_per_iter, is_shown=is_shown)
                total_duration = round(time.time() - start_total, 2)
                print(
                    f"[{time.strftime('%H:%M:%S')}] batch i: {iters}, "
                    f"总耗时 {total_duration}s, 当前共收集了 {len(self.data_buffer)} 条数据"
                )
        except KeyboardInterrupt:
            print(f"\n\r[{time.strftime('%H:%M:%S')}] 用户中断")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收集中国象棋自对弈数据")
    parser.add_argument("--show", action="store_true", default=False, help="是否显示棋盘过程")
    parser.add_argument("--games", type=int, default=int(os.cpu_count() * 0.75), help="每次并行运行的对局数（默认：CPU核心数）")
    args = parser.parse_args()

    collecting_pipeline = CollectPipeline(init_model="current_policy.pkl")
    n_games = args.games if args.games > 0 else os.cpu_count()
    collecting_pipeline.run(n_games_per_iter=n_games, is_shown=args.show)
