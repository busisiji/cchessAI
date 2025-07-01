# csv_to_pkl.py
import copy
import os
import pickle

import pandas as pd
import cchess
from collections import defaultdict

from tools import zip_state_mcts_prob, move_action2move_id, flip, move_id2move_action

# UCI 格式棋盘索引对照表
"""
81:a9 82:b9 83:c9 84:d9 85:e9 86:f9 87:g9 88:h9 89:i9
72:a8 73:b8 74:c8 75:d8 76:e8 77:f8 78:g8 79:h8 80:i8
63:a7 64:b7 65:c7 66:d7 67:e7 68:f7 69:g7 70:h7 71:i7
54:a6 55:b6 56:c6 57:d6 58:e6 59:f6 60:g6 61:h6 62:i6
45:a5 46:b5 47:c5 48:d5 49:e5 50:f5 51:g5 52:h5 53:i5
36:a4 37:b4 38:c4 39:d4 40:e4 41:f4 42:g4 43:h4 44:i4
27:a3 28:b3 29:c3 30:d3 31:e3 32:f3 33:g3 34:h3 35:i3
18:a2 19:b2 20:c2 21:d2 22:e2 23:f2 24:g2 25:h2 26:i2
 9:a1 10:b1 11:c1 12:d1 13:e1 14:f1 15:g1 16:h1 17:i1
 0:a0  1:b0  2:c0  3:d0  4:e0  5:f0  6:g0  7:h0  8:i0
"""

"""
红方从右到左算列数
C2+5 第8列的炮向上移动5行
k6+1 第6列的将向下移动1行
"""

class ChessDataLoader():
    def __init__(self):
        self.ele_red = "B" # B
        self.ele_blacl = 'b' # e
        self.horse_red = 'N' # H
        self.horse_black = 'n'  # h
        # self.ele_red = "E"
        # self.ele_blacl =  "e"
        # self.horse_red =  "H"
        # self.horse_black =  "h"
        self.piece_map = {
            'r': 'r', 'R': 'R',  # 车
            self.horse_black : 'n', self.horse_red: 'N',  # 马
            self.ele_blacl: 'b', self.ele_red: 'B',  # 象
            'a': 'a', 'A': 'A',  # 士
            'k': 'k', 'K': 'K',  # 将
            'c': 'c', 'C': 'C',  # 炮
            'p': 'p', 'P': 'P',  # 兵
        }
        self.to_row_sq = 0
        self.to_col_sq = 0
    def get_red_col(self,col):
        return 8 - col
    def find_pieces_in_same_column(self,board: cchess.Board, piece_char: str):
        """
        查找指定棋子类型在棋盘上是否出现在同一列。

        Args:
            board (cchess.Board): 当前的棋盘状态。
            piece_char (str): 要查找的棋子符号（如 'r', 'n', 'k' 等）。

        Returns:
            dict: 键为列号，值为该列上的此棋子所在位置列表。
        """
        from collections import defaultdict

        column_pieces = defaultdict(list)

        for square in range(90):  # 中国象棋棋盘总共有 90 个格子（10行9列）
            piece = board.piece_at(square)
            if piece and piece.symbol() == piece_char:
                col = cchess.square_column(square)  # 获取列号
                column_pieces[col].append(square)

        # 过滤出同一列有多个棋子的情况
        same_col_pieces = {col: squares for col, squares in column_pieces.items() if len(squares) > 1}

        return same_col_pieces

    def notation_to_uci(self,notation: str,side : str, board: cchess.Board) -> str:
        """
        将中文象棋记谱法动作字符串转换为 UCI 格式的动作字符串。

        Args:
            notation (str): 如 "C2.5", "H2+3"
            board (cchess.Board): 当前棋盘状态用于推断棋子位置

        Returns:
            str: UCI 格式的动作字符串（如 "h2h9"）
        """
        same_col = None # 相同棋子是否出现同列
        same_col_piece = []
        values = []

        piece_char = notation[0]
        if piece_char not in self.piece_map:
            raise ValueError(f"未知棋子类型: {piece_char}")

        wxf_move = notation[1:]

        # 获取当前棋盘上的所有合法动作
        legal_moves = list(board.legal_moves)

        # if notation == 'N3-4':
        #     print("调试用")
        # 相同棋子出现同列
        if notation[1] in ['-', '+']:
            same_col = notation[1]
            same_columns = self.find_pieces_in_same_column(board, self.piece_map[piece_char])
            # 获取所有棋子位置，并按行号从小到大排序
            sorted_squares = sorted(
                list(same_columns.values())[0],  # 假设只有一列有重复棋子
                key=lambda sq: cchess.square_row(sq)  # 按行号排序
            )
            if notation[2] == '-':
                selected_square = sorted_squares[0]  # 取出行号最小的棋子位置
            else:
                selected_square = sorted_squares[-1]
            from_col = cchess.square_column(selected_square)
            wxf_move = str(from_col) + wxf_move[1:]
        else:
            from_col = int(wxf_move[0]) - 1

        if '.' in wxf_move or piece_char.upper() in [self.horse_red, self.ele_red, 'A']:
            to_col = int(wxf_move[-1]) - 1
        else:
            to_col = from_col

        # 红方棋子列数从右算起
        if side == 'red':
            if not same_col:
                from_col = self.get_red_col(from_col)
            if '.' in wxf_move or piece_char.upper() in [self.horse_red, self.ele_red, 'A']:
                to_col = self.get_red_col(to_col)
                wxf_move = str(from_col) + wxf_move[1] + str(to_col)
            else:
                to_col = from_col
                wxf_move = str(from_col) + wxf_move[1:]

        # 处理平移（如 C2.5）
        if '.' in wxf_move:

            for move in legal_moves:
                from_square = move.from_square
                to_square = move.to_square
                from_name = cchess.SQUARE_NAMES[from_square]
                to_name = cchess.SQUARE_NAMES[to_square]

                from_row_sq = cchess.square_row(from_square)
                from_col_sq = cchess.square_column(from_square)
                self.to_col_sq = cchess.square_column(to_square)

                # if from_name == 'e6' and to_name == 'a6':
                #     print(f"[INFO] 找到平移: {from_name}{to_name}")
                #     print("调试用")

                # 检查起始列和目标列是否匹配
                if (
                    from_col_sq == from_col and
                    self.to_col_sq == to_col and
                    self.piece_map[piece_char] == board.piece_at(from_square).symbol()
                ):
                    if not same_col:
                        return f"{from_name}{to_name}",move
                    else:
                        same_col_piece.append({from_row_sq:["{from_name}{to_name}",move]})


        # 处理进/退（如 H2+3）
        elif '+' in wxf_move or '-' in wxf_move:
            if piece_char.upper() in [self.horse_red, self.ele_red, 'A']: # 象 马 士 不走直线 +为移动后的位置
                for move in legal_moves:
                    from_square = move.from_square
                    to_square = move.to_square
                    from_name = cchess.SQUARE_NAMES[from_square]
                    to_name = cchess.SQUARE_NAMES[to_square]

                    from_col_sq = cchess.square_column(from_square)
                    self.to_col_sq = cchess.square_column(to_square)
                    from_row_sq = cchess.square_row(from_square)
                    self.to_row_sq = cchess.square_row(to_square)

                    # if from_name == 'e8' and to_name == 'f7':
                    #     print(f"[INFO] 找到进退: {from_name}{to_name}")
                    #     print("调试用")

                    if side == 'red':
                        if '+' in wxf_move and from_row_sq > self.to_row_sq:
                            continue
                        if '-' in wxf_move and from_row_sq < self.to_row_sq:
                            continue
                    else:
                        if '+' in wxf_move and from_row_sq < self.to_row_sq:
                            continue
                        if '-' in wxf_move and from_row_sq > self.to_row_sq:
                            continue

                    # 检查行是否在合理范围内
                    if from_col_sq == from_col and to_col == self.to_col_sq:
                        if self.piece_map[piece_char] == board.piece_at(from_square).symbol():
                            if not same_col:
                                return f"{from_name}{to_name}", move
                            else:
                                same_col_piece.append({from_row_sq: (f"{from_name}{to_name}", move)})
            else:
                for move in legal_moves:
                    from_square = move.from_square
                    to_square = move.to_square
                    from_name = cchess.SQUARE_NAMES[from_square]
                    to_name = cchess.SQUARE_NAMES[to_square]

                    from_row_sq = cchess.square_row(from_square)
                    from_col_sq = cchess.square_column(from_square)
                    self.to_row_sq = cchess.square_row(to_square)
                    self.to_col_sq = cchess.square_column(to_square)

                    # if from_name == 'g6' and to_name == 'g5':
                    #     print(f"[INFO] 找到进退: {from_name}{to_name}")
                    #     print("调试用")

                    if from_col == from_col_sq and to_col == self.to_col_sq and self.piece_map[piece_char] == board.piece_at(from_square).symbol():
                        if side == 'red':
                            if ('+' in wxf_move and self.to_row_sq == from_row_sq + int(wxf_move[-1]))\
                                    or ('-' in wxf_move and self.to_row_sq == from_row_sq - int(wxf_move[-1])):
                                if not same_col:
                                    return f"{from_name}{to_name}", move
                                else:
                                    same_col_piece.append({from_row_sq: (f"{from_name}{to_name}", move)})
                        else:
                            if ('+' in wxf_move and self.to_row_sq == from_row_sq - int(wxf_move[-1]))\
                                    or ('-' in wxf_move and self.to_row_sq == from_row_sq + int(wxf_move[-1])):
                                if not same_col:
                                    return f"{from_name}{to_name}", move
                                else:
                                    same_col_piece.append({from_row_sq: (f"{from_name}{to_name}", move)})

        if same_col_piece:
            if (side == 'red' and same_col == '-') or (side == 'black' and same_col == '+'):
                if same_col_piece[0].keys() <= same_col_piece[-1].keys():
                    values = same_col_piece[0].values()
                else:
                    values = same_col_piece[-1].values()
            else:
                if same_col_piece[0].keys() <= same_col_piece[-1].keys():
                    values = same_col_piece[-1].values()
                else:
                    values = same_col_piece[0].values()
        for key, move in values:
            return key, move
    def mirror_data(self, play_data):
        """左右对称变换，扩充数据集一倍，加速一倍训练速度"""
        mirror_data = []
        # 棋盘形状 [15, 10, 9], 走子概率，赢家
        for state, mcts_prob, winner in play_data:
            # 原始数据
            mirror_data.append(zip_state_mcts_prob((state, mcts_prob, winner)))
            # 水平翻转后的数据
            state_flip = state.transpose([1, 2, 0])[:, ::-1, :].transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                # 水平翻转后，走子概率也需要翻转
                mcts_prob_flip[i] = mcts_prob[
                    move_action2move_id[flip(move_id2move_action[i])]
                ]
            mirror_data.append(
                zip_state_mcts_prob((state_flip, mcts_prob_flip, winner))
            )
        return mirror_data

    # def load_game_data(self,moves_file, gameinfo_file):
    #     """
    #     加载并解析棋局数据，生成用于训练的数据集。
    #
    #     Args:
    #         moves_file (str): 棋局步骤文件路径。
    #         gameinfo_file (str): 棋局胜负信息文件路径。
    #
    #     Returns:
    #         list: 包含(gameID, board_state, outcome) 的元组列表。
    #     """
    #     moves_df = pd.read_csv(moves_file)
    #     gameinfo_df = pd.read_csv(gameinfo_file)
    #
    #     game_results = dict(zip(gameinfo_df['gameID'], gameinfo_df['winner']))
    #
    #     data_buffer = []
    #
    #     grouped_moves = moves_df.groupby('gameID')
    #     num = 0
    #
    #     for game_id, group in grouped_moves:
    #         self.last_moves = []
    #         group = group.sort_values(by=['turn', 'side'], ascending=[True, False])
    #
    #         board = cchess.Board()
    #         winner = game_results.get(game_id, None)
    #
    #         print("\n")
    #         print("\n")
    #         print("\n")
    #         print(f"[INFO] 开始解析棋局: {game_id}, 胜方: {winner}")
    #
    #         for i, row in group.iterrows():
    #             move_str = row['move']
    #             side = row['side']
    #             turn = row['turn']
    #
    #             try:
    #                 print(board)
    #                 move_uci,move = self.notation_to_uci(move_str, side, board)
    #                 if move is None:
    #                     print(f"[警告] 无法找到合法动作: {move_uci}")
    #                     continue
    #             except Exception as e:
    #                 print(f"[警告] 非法动作格式: {move_str}, 错误: {e}")
    #                 continue
    #
    #             if not board.is_legal(move):
    #                 print(f"[警告] 非法动作: {move_uci}")
    #                 continue
    #             print(f"{game_id}第{turn}回合---{board.turn}----{[self.piece_map[move_str[0]]+move_str[1:]]}{[move_uci]}----")
    #
    #             board.push(move)
    #
    #             self.last_moves.append({move_str[0]:cchess.square_column(move.to_square)})
    #
    #         outcome = board.outcome()
    #         result = "未完成"
    #         if outcome is not None:
    #             if outcome.winner is not None:
    #                 num += 1
    #                 result = "红方胜利" if outcome.winner == cchess.RED else "黑方胜利"
    #             else:
    #                 num += 1
    #                 result = "和棋"
    #
    #         print(f"[INFO] 棋局 {game_id} 结束，结果: {result},已完成{ num}局")
    #
    #         data_buffer.append({
    #             'gameID': game_id,
    #             'board': board.copy(),
    #             'outcome': outcome
    #         })
    #
    #     return data_buffer
    def load_game_data(self, moves_file, gameinfo_file):
        """
        加载并解析棋局数据，按自我对弈格式保存到 DATA_BUFFER_PATH。

        Args:
            moves_file (str): 棋局步骤文件路径。
            gameinfo_file (str): 棋局胜负信息文件路径。
        """
        import numpy as np
        from tools import decode_board
        from parameters import DATA_BUFFER_PATH

        moves_df = pd.read_csv(moves_file)
        gameinfo_df = pd.read_csv(gameinfo_file)

        game_results = dict(zip(gameinfo_df['gameID'], gameinfo_df['winner']))
        grouped_moves = moves_df.groupby('gameID')
        num = 0
        print(f"[INFO] 开始加载棋局数据，目标文件: {DATA_BUFFER_PATH}")

        data_buffer = []
        iters = 0
        # 加载已有数据（如果存在）
        if os.path.exists(DATA_BUFFER_PATH):
            try:
                with open(DATA_BUFFER_PATH, "rb") as f:
                    data = pickle.load(f)
                    loaded_data = data.get("data_buffer", [])
                    data_buffer.extend(loaded_data)
                    iters += data.get("iters", 0)
            except Exception as e:
                print(f"加载旧数据失败：{e}")

        for game_id, group in grouped_moves:
            self.last_moves = []
            board = cchess.Board()
            winner = game_results.get(game_id, None)
            group = group.sort_values(by=['turn', 'side'], ascending=[True, False])

            print(f"[INFO] 解析棋局: {game_id}, 胜方: {winner}")

            states, mcts_probs, current_players = [], [], []

            for i, row in group.iterrows():
                move_str = row['move']
                side = row['side']
                turn = row['turn']
                if side == 'red':
                    move_str = move_str[0].upper() + move_str[1:]
                else:
                    move_str = move_str[0].lower() + move_str[1:]

                try:
                    if move_str == 'n7-5':
                        print("调试用")
                    print(board)
                    print(f"{game_id}第{turn}回合---{move_str}--------")
                    move_uci, move = self.notation_to_uci(move_str, side, board)
                    if move is None:
                        print(f"[警告] 无法找到合法动作: {move_uci}")
                        break
                except Exception as e:
                    print(f"[警告] 非法动作格式: {move_str}, 错误: {e}")
                    break

                if not board.is_legal(move):
                    print(f"[警告] 非法动作: {move_uci}")
                    break

                # 记录当前状态
                current_state = decode_board(board)  # 获取当前棋盘状态
                states.append(current_state)
                # 创建全零数组
                prob = np.zeros(2086, dtype=np.float32)

                # 获取当前 move 对应的 move_id
                uci_move = move.uci()  # 如 'e6e9'
                move_idx = move_action2move_id.get(uci_move, -1)

                if move_idx != -1:
                    prob[move_idx] = 1.0  # 设置对应动作概率为 1.0
                else:
                    print(f"[警告] 动作 {uci_move} 未找到对应的 move_id")

                mcts_probs.append(prob)
                current_players.append(board.turn)

                # 执行移动
                board.push(move)
                self.last_moves.append({move_str[0]:cchess.square_column(move.to_square)})


            # 处理游戏结束情况
            outcome = board.outcome()
            if outcome is None:
                continue
            num += 1
            winner_z = np.zeros(len(current_players))

            if outcome.winner is not None:
                winner_color = outcome.winner
                for i, player_id in enumerate(current_players):
                    winner_z[i] = 1.0 if player_id == winner_color else -1.0

            # 将(state, mcts_prob, winner)添加到play_data
            play_data = list(zip(states, mcts_probs, winner_z))
            play_data = self.mirror_data(play_data)
            data_buffer.extend(play_data)
        # 写入或追加写入 pickle 文件
        with open(DATA_BUFFER_PATH, "wb") as f:
            pickle.dump({"data_buffer": data_buffer,
                         "iters": num}, f)

        print(f"[INFO] 棋局 {game_id} 已写入文件，共 {len(data_buffer)} 步")

        print(f"[INFO] 所有棋局已成功追加写入至 {DATA_BUFFER_PATH}")



# 示例运行
if __name__ == "__main__":
    chess = ChessDataLoader()
    data_buffer = chess.load_game_data("archive/moves.csv", "archive/gameinfo.csv")

