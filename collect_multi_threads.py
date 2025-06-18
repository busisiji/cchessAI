import hashlib
import os
import pickle
import time
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue, Process, Pipe, cpu_count


def get_available_gpus():
    """获取可用 GPU 列表"""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    else:
        return []


def worker(args):
    """工作进程"""
    from collect import CollectPipeline

    save_interval = args["save_interval"]
    shared_queue = args["shared_queue"]
    gpu_id = args["gpu_id"]
    port = args["port"]
    is_shown = args["is_shown"]
    pid = args["pid"]
    log_pipe = args["log_pipe"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id is not None else "-1"

    pipeline = CollectPipeline(pid=pid, port=port, log_pipe=log_pipe)

    try:
        while True:
            iters = pipeline.collect_data(n_games=save_interval, is_shown=is_shown)
            log_pipe.send(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 完成 {save_interval} 局")
            shared_queue.put(list(pipeline.data_buffer))
    except KeyboardInterrupt:
        log_pipe.send(f"[{time.strftime('%H:%M:%S')}][PID={pid}] 子进程中止")


def generate_fingerprint(state):
    """生成状态指纹用于去重"""
    return hashlib.md5(state.tobytes()).hexdigest()


def deduplicate(data_list, seen_hashes):
    """去除重复数据"""
    deduped = []
    for item in data_list:
        state = item[0]
        fp = generate_fingerprint(state)
        if fp not in seen_hashes:
            seen_hashes[fp] = True
            deduped.append(item)
    return deduped

def log_writer(conn):
    """单独进程处理日志写入"""
    while True:
        message = conn.recv()
        if message == "STOP":
            break
        print(message)


def writer_process(shared_queue, output_path, buffer_size, seen_hashes):
    """数据写入进程"""
    merged_data = deque([], maxlen=buffer_size)

    if os.path.exists(output_path):
        try:
            with open(output_path, "rb") as f:
                data = pickle.load(f)
                loaded_data = data.get("data_buffer", [])
                merged_data.extend(loaded_data)
                print(f"[{time.strftime('%H:%M:%S')}][WRITER] 已加载历史数据")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}][WRITER] 加载旧数据失败：{e}")

    try:
        while True:
            try:
                data = shared_queue.get(timeout=5)
                deduped_data = deduplicate(data, seen_hashes)
                merged_data.extend(deduped_data)
                print(f"[{time.strftime('%H:%M:%S')}][WRITER] 接收到新数据，新增 {len(deduped_data)} 条")
            except Exception as e:
                continue
    except KeyboardInterrupt:
        print("[WRITER] 写入最终数据...")
        final_deduped = deduplicate(list(merged_data), set())
        with open(output_path, "wb") as f:
            pickle.dump({"data_buffer": final_deduped}, f)
        print(f"[WRITER] 数据已保存至 {output_path}")
        merged_data.clear()


def main():
    parser = argparse.ArgumentParser(description="多进程采集中国象棋自对弈数据")
    parser.add_argument("--save-interval", type=int, default=1, help="每多少局保存一次数据")
    parser.add_argument("--workers", type=int, default=int(os.cpu_count() * 0.75), help="使用的进程数")
    parser.add_argument("--show", action="store_true", default=False, help="是否显示棋盘对弈过程")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="是否使用 GPU 推理")
    args = parser.parse_args()

    manager = Manager()
    shared_queue = manager.Queue()
    seen_hashes = manager.dict()
    output_path = "data_buffer.pkl"

    # 创建日志管道
    parent_conn, child_conn = Pipe()
    writer_proc = Process(target=log_writer, args=(child_conn,))
    writer_proc.start()

    # 启动写入进程
    writer_process_handle = Process(
        target=writer_process,
        args=(shared_queue, output_path, 10000, seen_hashes)
    )
    writer_process_handle.start()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    available_gpus = get_available_gpus()
    num_gpus = len(available_gpus)
    print(f"检测到可用 GPU 数量: {num_gpus}")

    base_port = 8000

    # 使用进程池
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i in range(args.workers):
            if args.use_gpu and num_gpus > 0:
                gpu_id = available_gpus[i % num_gpus]
            else:
                gpu_id = None

            port = base_port + i

            worker_args = {
                "save_interval": args.save_interval,
                "shared_queue": shared_queue,
                "gpu_id": gpu_id,
                "port": port,
                "is_shown": args.show,
                "pid": port,
                "log_pipe": parent_conn
            }

            future = executor.submit(worker, worker_args)
            futures.append(future)

        try:
            for future in as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            print("主进程收到中断信号，等待所有子进程结束...")

    writer_process_handle.terminate()
    writer_process_handle.join()

    parent_conn.send("STOP")
    writer_proc.join()


if __name__ == "__main__":
    import torch
    main()
