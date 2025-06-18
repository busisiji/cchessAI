import cchess
import cchess.svg
import time
import numpy as np
from IPython.display import display, SVG
from mcts import MCTS_AI
from parameters import PLAYOUT, C_PUCT
from tools import decode_board, move_id2move_action, is_tie
from frontend import get_chess_window, create_window_visualization


class Game(object):
    """
    在cchess.Board类基础上定义Game类, 用于启动并控制一整局对局的完整流程,
    收集对局过程中的数据，以及进行棋盘的展示
    """

    def __init__(self, board, port=8000):
        self.board = board
        self.port = port
        self.chess_window = create_window_visualization(port=self.port)

        # 动态 playout 参数
        self.c_puct_schedule = [
            ((0, 30), (PLAYOUT-400 if PLAYOUT > 500 else 200, C_PUCT if C_PUCT > 0 else 3)),
            ((30, 80), (PLAYOUT-200 if PLAYOUT > 400 else 400, C_PUCT-1 if C_PUCT > 1 else 2)),
            ((80, float('inf')), (PLAYOUT if PLAYOUT > 600 else 800, C_PUCT-2 if C_PUCT > 2 else 1))
        ]

    @staticmethod
    def get_last_move(board):
        """获取最后一步移动"""
        return board.peek() if board.move_stack else None

    def graphic(self, board):
        """可视化棋盘"""
        svg = cchess.svg.board(
            board,
            size=600,
            coordinates=True,
            axes_type=0,
            checkers=board.checkers(),
            lastmove=board.peek() if board.move_stack else None,
            orientation=cchess.RED,
        )

        current_player = "红方" if board.turn == cchess.RED else "黑方"
        status_text = f"当前走子: {current_player} - 步数: {len(board.move_stack)}"

        try:
            window = self.chess_window()
            if window:
                window.update_board(svg, status_text)
                time.sleep(0.1)
        except ImportError:
            display(SVG(svg))

    def start_play(self, player1, player0, is_shown=True):
        """启动人机/人人对局"""
        self.board = cchess.Board()
        player1.set_player_idx(1)
        player0.set_player_idx(0)
        players = {cchess.RED: player1, cchess.BLACK: player0}

        if is_shown:
            self.graphic(self.board)

        while True:
            player_in_turn = players[self.board.turn]
            move = player_in_turn.get_action(self.board)
            self.board.push(move)

            if is_shown:
                self.graphic(self.board)

            if self.board.is_game_over():
                outcome = self.board.outcome()
                winner = outcome.winner if outcome else None
                if winner is not None:
                    winner_name = "RED" if winner == cchess.RED else "BLACK"
                    print(f"[{time.strftime('%H:%M:%S')}] 游戏结束. 赢家是: {winner_name}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] 游戏结束. 平局")
                return winner

    def get_c_puct_by_step(self, step):
        """根据当前步数获取对应的搜索参数"""
        return next(((n_playout, c_puct) for (low, high), (n_playout, c_puct) in self.c_puct_schedule if low <= step < high), (PLAYOUT, C_PUCT))

    def reset_mcts(self, policy_value_net, n_playout=None, c_puct=None):
        """重置MCTS"""
        n_playout = n_playout or PLAYOUT
        c_puct = c_puct or C_PUCT
        return MCTS_AI(
            policy_value_net.policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=True
        )

    def start_self_play(self, policy_value_net, is_shown=False, temp=1e-3, pid=None, board=None):
        """
        开始自我对弈
        :param board: 可选参数，如果传入则使用该棋盘实例
        """
        # 如果未传入棋盘，则新建一个
        if board is None:
            self.board = cchess.Board()
        else:
            self.board = board
            board.reset()  # 复用时清空棋盘状态

        states, mcts_probs, current_players = [], [], []
        move_count = 0

        # 初始化玩家
        player = self.reset_mcts(policy_value_net, PLAYOUT, C_PUCT)

        while True:
            move_count += 1
            n_playout, c_puct = self.get_c_puct_by_step(move_count)

            # 只在必要时更新参数
            if n_playout != player.mcts.n_playout or abs(c_puct - player.mcts.c_puct) > 1e-6:
                player.mcts.n_playout = n_playout
                player.mcts.c_puct = c_puct

            # 获取动作
            if move_count % 20 == 0 or move_count == 1:
                start_time = time.time()
                move, move_probs = player.get_action(self.board, temp=temp, return_prob=True)
                print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 第{move_count}步耗时: {time.time() - start_time:.2f}秒")
            else:
                move, move_probs = player.get_action(self.board, temp=temp, return_prob=True)

            # 收集数据
            states.append(decode_board(self.board))
            mcts_probs.append(move_probs)
            current_players.append(self.board.turn)
            self.board.push(cchess.Move.from_uci(move_id2move_action[move]))

            if is_shown:
                self.graphic(self.board)

            if self.board.is_game_over() or is_tie(self.board):
                outcome = self.board.outcome() if self.board.is_game_over() else None
                winner = -1
                winner_z = np.zeros(len(current_players))

                if outcome and outcome.winner is not None:
                    winner = outcome.winner
                    for i, player_id in enumerate(current_players):
                        winner_z[i] = 1.0 if player_id == winner else -1.0
                    if is_shown:
                        print(f"[{time.strftime('%H:%M:%S')}] 游戏结束. {'红方' if winner == cchess.RED else '黑方'} 获胜")
                else:
                    if is_shown:
                        print(f"[{time.strftime('%H:%M:%S')}] 游戏结束. 平局")

                player.reset_player()
                return winner, zip(states, mcts_probs, winner_z)
