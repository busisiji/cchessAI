# mcts.py

import time
import numpy as np
import cchess

from parameters import C_PUCT, EPS, ALPHA, IS_DYNAMIC_PLAYOUT
from tools import is_tie, move_id2move_action, softmax


class Node:
    """
    蒙特卡罗树中的节点，表示一个游戏状态。
    """

    def __init__(self, parent=None, prob=None):
        self.parent = parent  # 父节点
        self.children = {}   # 子节点字典 {action: Node}
        self.value = 0       # 当前节点的价值估计 Q
        self.visits = 0      # 访问次数 N
        self.prob = prob     # 先验概率 P（来自策略网络）

    def is_leaf(self):
        """判断是否是叶子节点"""
        return self.children == {}

    def is_root(self):
        """判断是否是根节点"""
        return self.parent is None

    def expand(self, action_priors):
        """
        根据策略网络的输出扩展子节点
        :param action_priors: (action, prior_probability) 列表
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(parent=self, prob=prob)

    def puct_value(self, c_puct=C_PUCT):
        """
        使用 PUCT 公式计算当前节点的价值
        """
        if self.visits == 0:
            return float("inf")
        q_value = self.value
        u_value = c_puct * self.prob * np.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value

    def select(self, c_puct):
        """
        选择最优子节点
        :return: (action, next_node)
        """
        return max(self.children.items(), key=lambda node: node[1].puct_value(c_puct))

    def update(self, leaf_value):
        """
        更新当前节点的访问次数和价值估计
        """
        self.visits += 1
        self.value += (leaf_value - self.value) / self.visits

    def update_recursive(self, leaf_value):
        """
        从叶节点反向更新所有祖先节点
        """
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)


class MCTS:
    """
    蒙特卡罗树搜索主体类
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        """
        :param policy_value_fn: 输入棋盘返回 (action_probs, value) 的函数
        :param c_puct: 探索常数
        :param n_playout: 每次搜索的模拟次数
        """
        self.root = Node(prob=1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.cache = {}  # FEN 缓存，避免重复计算相同局面

    def playout(self, board):
        """
        执行一次模拟，从根节点开始直到叶节点，并进行评估和回溯更新
        """
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            board.push(cchess.Move.from_uci(move_id2move_action[action]))

        fen = board.fen()
        if fen in self.cache:
            leaf_value = self.cache[fen]
            action_probs, _ = self.policy(board)
        else:
            action_probs, leaf_value = self.policy(board)
            self.cache[fen] = leaf_value

        if not board.is_game_over() and not is_tie(board):
            node.expand(action_probs)

        elif board.is_game_over():
            winner = cchess.RED if board.outcome().winner else cchess.BLACK
            leaf_value = 1.0 if winner == board.turn else -1.0
        else:
            leaf_value = 0.0

        node.update_recursive(-leaf_value)

    def get_move_probs(self, board, temp=1e-3):
        """
        获取当前棋盘下每个动作的概率分布
        :param board: 棋盘对象
        :param temp: 温度参数，控制探索程度
        :return: (动作列表, 概率列表)
        """
        for _ in range(self.n_playout):
            board_copy = board.copy()
            self.playout(board_copy)

        act_visits = [(act, node.visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        更新树结构以反映最新一步落子
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(prob=1.0)


class MCTS_AI():
    """
    基于 MCTS 的 AI 玩家接口
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=400, is_selfplay=False):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_idx(self, idx):
        self.player = idx

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=False):
        """
        获取 AI 动作
        """
        print(f"[AI] 开始思考第 {len(board.move_stack)} 步...")

        start_time = time.time()
        acts, probs = self.mcts.get_move_probs(board, temp)
        print(f"[AI] 思考结束，耗时 {time.time() - start_time:.2f} 秒")

        move_probs = np.zeros(2086)
        move_probs[list(acts)] = probs

        if self.is_selfplay:
            # 自我对弈时添加 Dirichlet 噪声增强探索
            move = np.random.choice(acts, p=(1 - EPS) * probs + EPS * np.random.dirichlet(ALPHA * np.ones(len(probs))))
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)

        if return_prob:
            return move, move_probs
        else:
            return move
