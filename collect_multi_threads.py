# 多进程采集数据
import os
import pickle
import time
import argparse
from multiprocessing import Process, Manager
import numpy as np
from collections import deque
import hashlib
import torch

from parameters import DATA_BUFFER_PATH, DATA_BUFFER_PATH_2


# 获取可用 GPU 列表
def get_available_gpus():
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    else:
        return []


def worker(save_interval, gpu_id, pid, is_shown=False):
    """
    子进程执行的任务函数：创建 CollectPipeline 实例并收集数据，通过共享列表传出。
    参数：
        save_interval: 每多少局保存一次数据
        gpu_id: 使用的 GPU 编号（None 表示使用 CPU）
        pid: 当前子进程 PID
        is_shown: 是否显示棋盘对弈过程
    """
    print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 子进程启动，使用 {'CPU' if gpu_id is not None else f'GPU:{gpu_id}'}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id is not None else "-1"

    from collect_multi_thread import CollectPipeline

    # 创建CollectPipeline实例并传入数据队列
    pipeline = CollectPipeline(pid=pid, port=pid)

    try:
        while True:
            iters = pipeline.collect_data(n_games=save_interval, is_shown=is_shown)
            print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 完成 {save_interval} 局，总完整局数: {iters}")
    except KeyboardInterrupt:
        print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 子进程中止，提交完成数据")


def main():
    parser = argparse.ArgumentParser(description="多进程采集中国象棋自对弈数据")
    parser.add_argument("--save-interval", type=int, default=1, help="每多少局保存一次数据")
    parser.add_argument("--workers", type=int, default=int(os.cpu_count() * 0.75), help="使用的进程数")
    parser.add_argument("--show", action="store_true", default=False, help="是否显示棋盘对弈过程")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="是否使用 GPU 进行推理（默认为 True）")
    args = parser.parse_args()

    print(f"使用 {args.workers} 个进程进行数据采集")

    manager = Manager()

    # 设置环境变量控制线程数（适用于 TensorFlow/PyTorch）
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # 获取可用 GPU
    available_gpus = get_available_gpus()
    num_gpus = len(available_gpus)
    print(f"检测到可用 GPU 数量: {num_gpus}")

    # 起始端口
    base_port = 8000

    # 启动多个 worker 进程
    workers = []
    for i in range(args.workers):
        if args.use_gpu and num_gpus > 0:
            gpu_id = available_gpus[i % num_gpus]
        else:
            gpu_id = None

        port = base_port + i  # 每个进程分配不同端口

        p = Process(
            target=worker,
            args=(args.save_interval, gpu_id, port, args.show)
        )
        p.start()
        workers.append(p)

    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("主进程收到中断信号，等待所有子进程结束...")

    print(f"[{time.strftime('%H:%M:%S')}] 所有子进程已完成，正在合并并保存数据...")

    # 收集所有子进程的数据
    merged_data = []
    iters = 0

    # 遍历所有子文件并读取数据
    for i in range(args.workers):
        pid = base_port + i
        sub_data_path = f"{DATA_BUFFER_PATH}_{pid}"

        if os.path.exists(sub_data_path):
            try:
                with open(sub_data_path, "rb") as f:
                    while True:
                        try:
                            item = pickle.load(f)
                            merged_data.extend(item.get("data_buffer", []))
                            iters += item.get("iters", 0)
                        except EOFError:
                            break
                print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 已读取子文件数据，当前数据量: {len(merged_data)},局数{iters}")
                os.remove(sub_data_path)  # 删除子文件
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 读取子文件失败：{e}")

    # 加载已有数据（如果存在）
    data_path = DATA_BUFFER_PATH_2
    if os.path.exists(data_path):
        try:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                loaded_data = data.get("data_buffer", [])
                merged_data.extend(loaded_data)
                iters += data.get("iters", 0)
                print(f"[{time.strftime('%H:%M:%S')}] 已加载历史数据，共 {len(loaded_data)} 条样本,局数{iters}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 加载旧数据失败：{e}")
    # 保存最终数据
    with open(data_path, "wb") as f:
        pickle.dump({
            "data_buffer": merged_data,
            "iters": iters
        }, f)

    print(f"[{time.strftime('%H:%M:%S')}] 最终数据已合并至 {data_path},共 {len(merged_data)} 条样本,局数{iters}")


if __name__ == "__main__":
    import torch
    main()
