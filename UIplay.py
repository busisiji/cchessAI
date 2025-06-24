# UIplay.py - 改进版，支持人机对战

from game import Game
import cchess
from mcts import MCTS_AI
from tools import move_action2move_id, move_id2move_action  # 注意变量名是否一致

from net import PolicyValueNet


class Human:
    """人类玩家类"""
    def get_action(self, board):
        try:
            user_input = input('请输入走法 (例如: e6e9): ')
            uci_move = move_action2move_id.get(user_input)
            if uci_move is None:
                print("无效的走法，请重新输入。")
                return self.get_action(board)
            uci_str = move_id2move_action[uci_move]
            move = cchess.Move.from_uci(uci_str)

            # 检查是否是合法走法
            if move not in board.legal_moves:
                print(f"非法走法: {user_input}，请重试！")
                return self.get_action(board)

            return uci_move

        except Exception as e:
            print(f"输入错误，请重试。错误: {e}")
            return self.get_action(board)


    def set_player_idx(self, idx):
        """设置玩家标识（红方/黑方）"""
        self.player_idx = idx



def run():
    # 加载模型
    policy_value_net = PolicyValueNet(model_file='current_policy.pkl')

    # 创建 MCTS 玩家
    mcts_player = MCTS_AI(
        policy_value_net.policy_value_fn,
        c_puct=5,
        n_playout=1200,
        is_selfplay=False
    )

    # 创建人类玩家
    human = Human()

    # 初始化棋盘和游戏
    board = cchess.Board()  # 使用 cchess 的 Board 类
    game = Game(board)

    # 开始人机对战（AI 先手）
    game.start_play(player1=human, player0=mcts_player, is_shown=1)


if __name__ == '__main__':
    run()
