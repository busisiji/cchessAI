import numpy as np
import cchess
from parameters import C_PUCT, EPS, ALPHA
from tools import is_tie, move_id2move_action, softmax, decode_board


class Node(object):
    """
    蒙特卡罗树搜索的游戏状态节点
    """

    def __init__(self, parent=None, prob=None):
        self.parent = parent
        self.children = {}
        self.value = 0
        self.visits = 0
        self.prob = prob

    def is_leaf(self):
        return not self.children

    def is_root(self):
        return self.parent is None

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def puct_value(self, c_puct=C_PUCT):
        if self.visits == 0:
            return float("inf")
        return self.value + c_puct * self.prob * np.sqrt(self.parent.visits) / (1 + self.visits)

    def select(self, c_puct):
        return max(
            self.children.items(),
            key=lambda act_node: act_node[1].puct_value(c_puct)
        )

    def update(self, leaf_value):
        self.visits += 1
        self.value += 1.0 * (leaf_value - self.value) / self.visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)


class MCTS(object):
    """
    蒙特卡罗树搜索算法
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        self.root = Node(None, 1.0)
        self.policy = policy_value_fn  # 可能是 PolicyValueNet 的 batch_policy_value
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.batch_size = 64  # 批量大小
        self.batch_states = []
        self.batch_nodes = []

    def playout(self, board):
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            board.push(cchess.Move.from_uci(move_id2move_action[action]))

        action_probs, leaf_value = self.policy([decode_board(board)])
        end = board.is_game_over()

        if end and is_tie(board):
            leaf_value = 0.0
        elif not end and not is_tie(board):
            node.expand(action_probs)

        node.update_recursive(-leaf_value)
    def get_move_probs(self, board, temp=1e-3):
        for _ in range(self.n_playout):
            board_copy = board.copy()
            self.playout(board_copy)

        act_visits = [(act, node.visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)
    def batch_playout(self, board):
        """用于批量推理"""
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            board.push(cchess.Move.from_uci(move_id2move_action[action]))

        # 缓存状态和对应节点
        state = decode_board(board)
        self.batch_states.append(state)
        self.batch_nodes.append(node)

        if len(self.batch_states) >= self.batch_size:
            self.process_batch()

    def process_batch(self):
        """执行批量推理"""
        if not self.batch_states:
            return

        states_batch = np.array(self.batch_states)
        act_probs_list, value_list = self.policy(states_batch)

        for node, act_probs, value in zip(self.batch_nodes, act_probs_list, value_list):
            node.expand(list(enumerate(act_probs)))
            node.update_recursive(-value)

        self.batch_states.clear()
        self.batch_nodes.clear()

class MCTS_AI(object):
    """
    基于MCTS的AI玩家
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=False):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.is_selfplay = is_selfplay

    def set_player_idx(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=False):
        acts, probs = self.mcts.get_move_probs(board, temp=temp)
        move_probs = np.zeros(2086)
        move_probs[list(acts)] = probs

        if self.is_selfplay:
            move = np.random.choice(acts, p=(1 - EPS) * probs + EPS * np.random.dirichlet(ALPHA * np.ones(len(probs))))
            self.mcts.update_with_move(move)
        else:
            move = np.random.choice(acts, p=probs)
            self.mcts.update_with_move(-1)

        return (move, move_probs) if return_prob else move
