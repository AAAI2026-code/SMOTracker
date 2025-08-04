import multiprocessing
import torch
import time
import random
import os
from datetime import datetime
from sam2_run import process_run

# 工作进程函数
def worker(gpu_id, task_queue, result_queue):
    """工作进程函数，使用指定的GPU执行任务"""
    # 设置进程名称以便识别
    current_process = multiprocessing.current_process()
    current_process.name = f"Worker-GPU{gpu_id}"

    try:
        # 初始化GPU设备
        if torch.cuda.is_available():
            # 打印GPU信息
            device_name = torch.cuda.get_device_name(gpu_id)
            print(f"\n{current_process.name} initialized on GPU {gpu_id}: {device_name}")

            # 执行一个小的预热操作
            warm_up_tensor = torch.rand(10, 10, device=torch.device(f"cuda:{gpu_id}"))
            _ = torch.mm(warm_up_tensor, warm_up_tensor.T)
        else:
            print(f"Warning: {current_process.name} - CUDA not available")
            result_queue.put((f"GPU {gpu_id} not available", []))
            return

        # 处理任务队列
        while not task_queue.empty():
            try:
                task = task_queue.get_nowait()
                result = process_run(task)
                result_queue.put((gpu_id, task, result))
            except Exception as e:
                print(f"Task processing error in {current_process.name}: {str(e)}")
                break

    except Exception as e:
        print(f"Initialization error in {current_process.name}: {str(e)}")
    finally:
        # 清理GPU内存
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        print(f"{current_process.name} shutting down")


# 主函数
def main():
    # 检查可用GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print(f"Error: This script requires at least 2 GPUs. Found {num_gpus}.")
        return

    print(f"Available GPUs: {num_gpus}")
    print(f"Using GPU 0 and GPU 1 for parallel inference\n")

    # 显示GPU信息
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # 创建任务队列和结果队列
    manager = multiprocessing.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # 添加任务数据
    tasks = [
        "MOT17_Anti-UAV(DUT)" ,
        "MOT17_AntiUAV410",
        "MOT17_AntiUAV_infrared",
        "MOT17_AntiUAV_visible"
    ]

    for task in tasks:
        task_queue.put(task)

    print(f"\nStarting inference on {len(tasks)} tasks using GPU 0 and GPU 1...")

    # 创建并启动工作进程
    processes = []

    # 进程1: 专门使用GPU 0
    p1 = multiprocessing.Process(
        target=worker,
        args=(0, task_queue, result_queue),
        name="Worker-GPU0"
    )
    processes.append(p1)

    # 进程2: 专门使用GPU 1
    p2 = multiprocessing.Process(
        target=worker,
        args=(1, task_queue, result_queue),
        name="Worker-GPU1"
    )
    processes.append(p2)

    # 启动所有进程
    start_time = time.time()
    for p in processes:
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # 计算总耗时
    total_time = time.time() - start_time

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("Inference Results Summary:")
    print("=" * 60)

    # 按GPU分组显示结果
    gpu0_results = [r for r in results if r[0] == 0]
    gpu1_results = [r for r in results if r[0] == 1]

    print(f"\nGPU 0 processed {len(gpu0_results)} tasks:")
    for _, task, result in gpu0_results:
        print(f"  - {task}: {result}")

    print(f"\nGPU 1 processed {len(gpu1_results)} tasks:")
    for _, task, result in gpu1_results:
        print(f"  - {task}: {result}")

    print("\n" + "=" * 60)
    print(f"Total tasks processed: {len(results)}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn')

    # 运行主程序
    main()