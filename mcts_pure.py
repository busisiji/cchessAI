# mcts_pure.py

import threading
import queue
import copy
import time

import numpy as np
from mcts import MCTS, Node


class ParallelMCTSWorker(threading.Thread):
    """
    并行 MCTS 工作线程类，用于在单独的线程中执行一次 MCTS 模拟
    """

    def __init__(self, request_queue, result_queue, policy_value_fn, c_puct, n_playout):
        super().__init__()
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.mcts = MCTS(self.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

    def run(self):
        while True:
            task = self.request_queue.get()
            if task is None:
                break
            board_copy, temp = task
            acts, probs = self.mcts.get_move_probs(board_copy, temp)
            self.result_queue.put((acts, probs))
            self.request_queue.task_done()


class ParallelMCTS:
    """
    并行 MCTS 类，使用多个线程并行执行 MCTS 模拟以提高效率
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400, num_workers=4):
        """
        :param policy_value_fn: 输入棋盘返回 (action_probs, value) 的函数
        :param c_puct: 探索常数
        :param n_playout: 每次搜索的模拟次数
        :param num_workers: 使用的线程数量
        """
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.num_workers = num_workers

        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # 启动工作线程
        self.workers = [
            ParallelMCTSWorker(
                self.request_queue,
                self.result_queue,
                self.policy_value_fn,
                self.c_puct,
                self.n_playout
            ) for _ in range(num_workers)
        ]
        for worker in self.workers:
            worker.daemon = True
            worker.start()

        # 当前根节点
        self.root = Node(prob=1.0)

    def get_move_probs(self, board, temp=1e-3):
        """
        获取当前棋盘下每个动作的概率分布（并行版）
        :param board: 棋盘对象
        :param temp: 温度参数，控制探索程度
        :return: (动作列表, 概率列表)
        """

        # 清空队列中的旧任务
        while not self.request_queue.empty():
            self.request_queue.get_nowait()
        while not self.result_queue.empty():
            self.result_queue.get_nowait()

        # 提交多个并行任务
        for _ in range(self.num_workers):
            self.request_queue.put((board.copy(), temp))

        # 收集结果
        results = []
        for _ in range(self.num_workers):
            results.append(self.result_queue.get())

        # 合并所有结果
        combined_acts = {}
        for acts, probs in results:
            for a, p in zip(acts, probs):
                combined_acts[a] = combined_acts.get(a, 0) + p

        combined_acts = {k: v / len(results) for k, v in combined_acts.items()}
        acts, probs = zip(*combined_acts.items())
        return list(acts), list(probs)

    def update_with_move(self, last_move):
        """
        更新树结构以反映最新一步落子
        :param last_move: 最后一步的动作
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(prob=1.0)


class MCTS_AI:
    """
    基于并行 MCTS 的 AI 玩家接口
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400, num_workers=4, is_selfplay=False):
        """
        :param policy_value_fn: 输入棋盘返回 (action_probs, value) 的函数
        :param c_puct: 探索常数
        :param n_playout: 每次搜索的模拟次数
        :param num_workers: 并行线程数
        :param is_selfplay: 是否是自我对弈模式
        """
        self.mcts = ParallelMCTS(policy_value_fn, c_puct=c_puct, n_playout=n_playout, num_workers=num_workers)
        self.is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_idx(self, idx):
        self.player = idx

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=False):
        """
        获取 AI 动作
        :param board: 棋盘对象
        :param temp: 温度参数
        :param return_prob: 是否返回概率
        :return: 动作 or (动作, 概率列表)
        """
        print(f"[AI] 开始思考第 {len(board.move_stack)} 步...")

        start_time = time.time()
        acts, probs = self.mcts.get_move_probs(board, temp)
        print(f"[AI] 思考结束，耗时 {time.time() - start_time:.2f} 秒")

        move_probs = np.zeros(2086)
        move_probs[list(acts)] = probs

        if self.is_selfplay:
            # 自我对弈时添加 Dirichlet 噪声增强探索
            from parameters import EPS, ALPHA
            move = np.random.choice(acts, p=(1 - EPS) * probs + EPS * np.random.dirichlet(ALPHA * np.ones(len(probs))))
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)

        if return_prob:
            return move, move_probs
        else:
            return move
