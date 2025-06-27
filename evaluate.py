# 加载模型并进行自我对弈评估
import os
import time
import cchess
from mcts import MCTS_AI
from game import Game
from net import PolicyValueNet
from parameters import PLAYOUT, MODEL_PATH


def evaluate_model(model_path, num_games=10):
    """
    加载模型并进行自我对弈评估
    :param model_path: 模型路径
    :param num_games: 对弈次数
    :return: 胜率、平局率
    """
    print(f"[{time.strftime('%H:%M:%S')}] 正在加载模型：{model_path}")
    policy_value_net = PolicyValueNet(model_file=model_path)

    # 创建两个 AI 玩家，使用相同的模型和不同模拟次数
    player1 = MCTS_AI(policy_value_fn=policy_value_net.policy_value_fn, n_playout=400)
    player2 = MCTS_AI(policy_value_fn=policy_value_net.policy_value_fn, n_playout=400)

    game = Game(cchess.Board())

    win_count = 0
    tie_count = 0

    for i in range(num_games):
        print(f"[{time.strftime('%H:%M:%S')}] 开始第 {i+1} 局对弈")
        winner = game.start_play(player1, player2, is_shown=False)  # 设置为 False 不显示棋盘

        if winner == cchess.RED:
            win_count += 1
        elif winner == -1:
            tie_count += 1

        print(f"[{time.strftime('%H:%M:%S')}] 第 {i+1} 局结果：{'红方胜' if winner == cchess.RED else '黑方胜' if winner == cchess.BLACK else '平局'}")

    win_rate = win_count / num_games
    tie_rate = tie_count / num_games

    print(f"\n[评估完成] 胜率: {win_rate:.2%}, 平局率: {tie_rate:.2%}\n")
    return win_rate, tie_rate


if __name__ == "__main__":
    # 指定要评估的模型路径
    model_path = "models/current_policy_batch300_2025-06-27_14-03-44.pkl"

    # 运行评估
    win_rate, tie_rate = evaluate_model(model_path, num_games=10)

    # 输出最终结果
    print(f"模型路径: {model_path}")
    print(f"胜率: {win_rate:.2%}")
    print(f"平局率: {tie_rate:.2%}")
