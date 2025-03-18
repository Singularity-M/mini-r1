import json
import time
import re

from multiprocessing import Process, Value, Lock, Queue
from tqdm import tqdm
from transformers import AutoTokenizer



def get_total_lines(input_file):
    """计算输入文件的行数"""
    count_sum = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            count_sum += 1
    
    return count_sum

def add_special_tokens(text, tokenizer):
    """在每个句子的开头和末尾加入开始和结束标记符号"""
    return tokenizer.bos_token + text + tokenizer.eos_token

def sliding_window_split(tokens, max_length=512, min_length=490):
    """使用滑动窗口切分tokens"""
    chunks = []
    start = 0
    res_chunks = []
    
    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunk = tokens[start:end]
        if len(chunk) == max_length:
            start = end - 20
            chunks.append(chunk)
        elif len(chunk) >= min_length:
            chunks.append(chunk)
            start = len(tokens)
        else:
            res_chunks = chunk
            start = len(tokens)
    
    return chunks, res_chunks


def worker_process(input_queue, output_queue, processed_count, lock):
    """工作进程：从队列获取数据并处理"""
    tokenizer = AutoTokenizer.from_pretrained('./model/minir1_tokenizer')
    while True:
        data_block = input_queue.get()
        if data_block is None:
            break
        
        pretrain_data = []
        current_tokens = []
        
        for line in data_block:
            data = json.loads(line.strip())
            q = re.sub(r'\s+', ' ', data['conversations'][0]['content'].strip().replace("\n", ""))
            a = re.sub(r'\s+', ' ', data['conversations'][1]['content'].strip().replace("\n", ""))
            
            text = q + a
            tokens = tokenizer.encode(add_special_tokens(text, tokenizer))
            with lock:
                processed_count.value += len(data_block)

            current_tokens.extend(tokens)
            
            if len(current_tokens) >= 512:
                chunks, res_chunks = sliding_window_split(current_tokens)
                for chunk in chunks:
                    pretrain_data.append({'tokens': chunk})
                current_tokens = res_chunks
            
            if current_tokens:
                if len(current_tokens) <= 512:
                    continue
                else:
                    chunks, res_chunks = sliding_window_split(current_tokens)
                    for chunk in chunks:
                        pretrain_data.append({'tokens': chunk})
                    if res_chunks:
                        pretrain_data.append({'tokens': res_chunks})
            if len(pretrain_data) >= 100:
                with lock:
                    output_queue.put(pretrain_data)
                    pretrain_data = []
        
        if pretrain_data:
            with lock:
                output_queue.put(pretrain_data)
        


def writer_process(output_queue, output_file, lock):
    """写入进程：从队列获取处理结果并写入文件"""
    with open(output_file, 'a', encoding='utf-8') as f:
        while True:
            data = output_queue.get()
            if data is None:
                break
            with lock:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def progress_monitor(processed_count, total):
    """进度监控：显示处理进度"""
    with tqdm(total=total, desc="Processing") as pbar:
        while True:
            current = processed_count.value
            pbar.update(current - pbar.n)
            if current >= total:
                break
            time.sleep(0.05)

def main(input_file, output_file, num_workers=1, block_size=1000):
    """主进程：初始化进程并管理队列和进度"""
    # 共享变量初始化
    processed_count = Value('i', 0)
    lock = Lock()

    # 初始化队列
    input_queue = Queue(maxsize=num_workers*2)
    output_queue = Queue()
    
    # 启动写入进程
    writer = Process(target=writer_process, args=(output_queue, output_file, lock))
    writer.start()
    
    # 启动工作进程
    workers = []
    for _ in range(num_workers):
        p = Process(target=worker_process, args=(input_queue, output_queue, processed_count, lock))
        p.start()
        workers.append(p)
    
    # 假设总行数是已知的
    total = get_total_lines(input_file)  # 这个值可以通过实际读取文件来计算

    # 启动进度监控进程
    monitor = Process(target=progress_monitor, args=(processed_count, total))
    monitor.start()
    
    # 读取输入文件并将数据块放入队列
    data_block = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data_block.append(line)
                # 每10000行放入一次队列
            if len(data_block) >= block_size:
                input_queue.put(data_block)
                data_block = []  # 清空数据块

            # 如果文件剩余数据小于10000行，依然要放入队列
        if data_block:
            input_queue.put(data_block)
    
    # 发送结束信号
    for _ in range(num_workers):
        input_queue.put(None)
    
    # 等待工作进程结束
    for p in workers:
        p.join()
    monitor.terminate()
    
    # 发送结束信号给写入进程
    output_queue.put(None)
    writer.join()

if __name__ == "__main__":
    input_path = ".jsonl"
    output_file = ".jsonl"
    main(input_path, output_file, num_workers=100)
