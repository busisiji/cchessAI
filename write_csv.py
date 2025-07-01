import time
from collections import defaultdict

import pandas as pd
import os
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen, Request
import csv
import os
import re
import requests
import webbrowser
import socket
from jiema import jiema

def getOwnerPage(owner, page):
    page = str(page)
    url = "http://www.dpxq.com/hldcg/search/list.asp?owner=" + owner + "&page=" + page
    user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0"
    req = Request(url, headers={'User-Agent': user_agent})
    try:
        html = urlopen(req).read().decode('gb2312')
    except Exception:
        try:
            html = urlopen(req).read().decode('gbk')
        except Exception:
            html = urlopen(req).read().decode('GB18030')
        finally:
            return [['', '']]
    soup = BeautifulSoup(html, features='lxml')
    boardData = soup.find_all('a', {"href": re.compile(r"javascript:view\(\'owner=(.*).*id=\d*\'\)")})
    url_list = []
    for bo in boardData:
        url_action = bo['href']
        url_result = re.findall(r'javascript:view\(\'owner=(.*)&id=(.*)\'', url_action)[0]
        url_list.append([url_result[0], url_result[1]])
    return url_list
def getChessManual(owner, id_name):  # 获取游戏棋谱，返回谁获胜，棋谱
    url = "http://www.dpxq.com/hldcg/search/view.asp?owner=" + owner + "&id=" + id_name
    user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0"
    req = Request(url, headers={'User-Agent': user_agent})
    try:
        html = urlopen(req).read().decode('gb2312')
    except Exception:
        try:
            html = urlopen(req).read().decode('gbk')
        except Exception:
            html = urlopen(req).read().decode('GB18030')
        finally:
            return ("error", '')
    soup = BeautifulSoup(html, features='lxml')
    boardData = str(soup.find_all('div', id='dhtmlxq_view')[0])
    try:
        a = re.findall(r"var DhtmlXQ_movelist.*\[DhtmlXQ_movelist\](.*)\[/DhtmlXQ_movelist\]", html)[0]
        b = re.findall(r"\[DhtmlXQ_binit\](.*)\[/DhtmlXQ_binit\]", boardData)[0]
        if b == '':
            b = '8979695949392919097717866646260600102030405060708012720323436383'
        chess_type = eval('\"' + re.findall(r"\[DhtmlXQ_type\](.*)\[/DhtmlXQ_type\]", boardData)[0] + '\"')

        # 只要全局
        if chess_type != '全局':
            return

        if chess_type == '':
            chess_type = "残局"
        winType = "\"" + re.findall(r"\[DhtmlXQ_result\](.*)\[/DhtmlXQ_result\]", boardData)[0] + "\""
        result_list = jiema().getMoveListString(a, b)
        if result_list[0][1] == 0:
            nextType = 'r'
        else:
            nextType = 'b'
        fen = binitToFen(b) + ' ' + nextType
        return fen, chess_type, eval(winType), result_list
    except Exception as e:
        raise  e

def binitToFen(binit):
    chess_board = [[" " for i in range(9)] for i in range(10)]
    chess_type = ['R', 'N', 'B', 'A', 'K', 'A', 'B', 'N', 'R', 'C', 'C', 'P', 'P', 'P', 'P', 'P']
    for i in range(32):
        x = int(binit[i * 2])
        y = int(binit[i * 2 + 1])
        if x == 9:
            continue
        if i <= 15:
            ct = chess_type[i]
        else:
            ct = str.lower(chess_type[i - 16])
        chess_board[y][x] = ct
    fen = ""
    for line in chess_board:
        black = 0
        for index, char in enumerate(line):
            if char == ' ':
                black += 1
            else:
                if black != 0:
                    fen += str(black)
                black = 0
                fen += char
            if index == 8:
                if char == ' ':
                    fen += str(black)
                fen += '/'
    return fen[:-1]



def getOwnerList(owner, page_down=100):
    re_set = set()
    for i in range(page_down):
        t = getOwnerPage(owner, i)
        for uname, id_name in t:
            if id_name == '':
                continue
            re_set.add(id_name)
    return re_set

def save_file(re_list, bh):
    if re_list[0] == 'error':
        return
    win_type = re_list[1]
    win_who = re_list[2]
    if win_who == '和棋':
        return
    elif win_who == '':
        return
    if win_who == '红胜':
        win_who = 'red'
    elif win_who == '黑胜':
        win_who = 'black'

    if not os.path.isdir('棋局库'):
        os.mkdir('棋局库')
    dir_path = '棋局库/' + win_type + '/'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    file_path = dir_path + win_who + '.csv'

    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # 如果文件不存在，则写入表头
        if not file_exists:
            writer.writerow(['gameID', 'turn', 'side', 'move', 'winner'])

        turn = 0
        for index, (move_str, side_flag) in enumerate(re_list[3], start=1):
            side = 'red' if side_flag == 0 else 'black'
            converted_move = (
                move_str.replace('j', '+')
                        .replace('Z', '.')
                        .replace('t', '-')
                        .replace('一', '1')
                        .replace('二', '2')
                        .replace('三', '3')
                        .replace('四', '4')
                        .replace('五', '5')
                        .replace('六', '6')
                        .replace('七', '7')
                        .replace('八', '8')
                        .replace('九', '9')
            )
            if side == 'red':
                converted_move = converted_move[0].upper() + converted_move[1:]
                turn += 1
            else:
                converted_move = converted_move[0].lower() + converted_move[1:]
            writer.writerow([bh, str(turn), side, converted_move, win_who])
        print(f"{bh}保存成功")

def merge_csv_files_recursively(directory, output_file='merged_output.csv'):
    """
    递归合并指定目录及其所有子目录下的CSV文件。

    参数:
        directory (str): 要搜索的根目录路径。
        output_file (str): 输出文件的路径，默认为'merged_output.csv'。

    返回:
        None: 结果保存到指定的输出文件中。
    """
    all_data = []

    # 递归遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                try:
                    df = pd.read_csv(file_path)
                    all_data.append(df)
                    print(f"[INFO] 已读取 {file_path}")
                except Exception as e:
                    print(f"[ERROR] 无法读取文件 {file_path}: {e}")

    # 合并数据
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        print(f"[INFO] 所有文件已合并并保存至 {output_file}")
    else:
        print("[WARNING] 没有找到可以合并的CSV文件。")


# 读取 moves.csv 文件
def read_moves(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        moves = list(reader)
    return moves

# 提取每个 gameID 的基本信息
def extract_game_info(moves_data):
    games = defaultdict(list)

    # 按 gameID 分组
    for move in moves_data:
        game_id = move['gameID']
        games[game_id].append(move)

    gameinfos = []
    for game_id, moves in games.items():
        first_side = moves[0]['side']  # 第一步是谁下的
        blackID = ''
        redID = ''
        winner = moves[0]['winner']

        # 构造 gameinfo 数据
        gameinfo = {
            'gameID': game_id,
            'winner': winner
        }
        gameinfos.append(gameinfo)
    return gameinfos

# 写入 gameinfo.csv 文件
def write_game_info(gameinfos, output_file):
    fieldnames = ['gameID',  'winner']
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gameinfos)

# 主函数
def moves_add_gameinfo(moves_path='moves.csv',gameinfo_path='gameinfo.csv'):
    moves_data = read_moves(moves_path)
    gameinfos = extract_game_info(moves_data)
    write_game_info(gameinfos, gameinfo_path)
    print("✅ gameinfo.csv 已成功生成！")



if __name__ == '__main__':
    # 37535, 1489888 棋友上传 u
    # 121608,131607 大师棋谱 m
    # 1313710 顶级棋谱 t

    now_label = 0
    try:
        for i in range(121610,131607):
            # index = str(int("10000000") + i)[1:]
            index = str(i)
            try:
                save_file(getChessManual('m', index), index)
            except Exception as e:
                print(f"[ERROR] 处理编号 {index} 时出错：{e}")
                time.sleep(30)
                continue
            # if i % 5 == 0:
            #     print(i)
        merge_csv_files_recursively('棋局库', '棋局库/moves.csv')
        moves_add_gameinfo('棋局库/moves.csv', '棋局库/gameinfo.csv')
    except KeyboardInterrupt:
        print("\n[INFO] 检测到中断，程序即将退出。")
        # 可选：在这里执行额外的清理操作
        exit()
