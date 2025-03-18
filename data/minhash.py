import jieba
import re
import json
import os
import time
from tqdm import tqdm
from multiprocessing import Process, Queue, Value, Lock, Manager
from datasketch import MinHashLSH, MinHash
from collections import defaultdict

current_directory = os.getcwd()

def get_total_lines(input_files):
    """计算输入文件的行数"""
    count_sum = 0
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            for _ in f:
                count_sum += 1
    return count_sum

def load_stopwords(stopwords_file):
    """加载停用词"""
    stopwords = set()
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords

def tokenize(text, punctuation_pattern, combined_stopwords):
    """文本预处理"""
    text = re.sub(punctuation_pattern, '', text)
    words = jieba.lcut(text)
    return [word for word in words if word not in combined_stopwords and word.strip()]

def deduplication(input_queue, output_queue, stopwords_files, lsh, lock, deplication_count, processed_count):
    """去重操作"""
    jieba.initialize()
    chinese_stopwords = load_stopwords(stopwords_files['chinese'])
    english_stopwords = load_stopwords(stopwords_files['english'])
    combined_stopwords = chinese_stopwords.union(english_stopwords)

    while True:
        line = input_queue.get()
        if line is None:
            break

        with lock:
            processed_count.value += 1

        try:
            data = json.loads(line.strip())
            text = data['conversations'][0]['content'] + data['conversations'][1]['content']
        except json.JSONDecodeError:
            with lock:
                deplication_count.value += 1  # 增加重复计数
            continue

        punctuation_pattern = re.compile(r'[，。！？；：“”‘’（）《》〈〉【】、~@#￥%……&*——+=|"<>?/\\\[\]!?;:"()<>{}\']')
        words = tokenize(text, punctuation_pattern, combined_stopwords)

        if len(words) <= 20:
            with lock:
                deplication_count.value += 1  # 增加重复计数
            continue

        # 使用 MinHash 计算文本的哈希值
        minhash = MinHash()
        for word in words:
            minhash.update(word.encode('utf8'))

        # 查询是否有相似的句子
        with lock:  # 确保对 LSH 的操作是线程安全的
            if not lsh.query(minhash):
                # 如果没有重复，加入 LSH 存储
                lsh.insert(str(processed_count.value), minhash)
                output_queue.put(data)
            else:
                deplication_count.value += 1  # 增加重复计数

def writer_process(output_queue, output_file):
    """写入进程处理函数"""
    buffer = []
    while True:
        item = output_queue.get()
        if item is None:
            if buffer:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for text in buffer:
                        f.write(json.dumps(text, ensure_ascii=False) + "\n")
            break
        buffer.append(item)
        if len(buffer) >= 10000:
            with open(output_file, 'a', encoding='utf-8') as f:
                for text in buffer:
                    f.write(json.dumps(text, ensure_ascii=False) + "\n")
            buffer.clear()

def progress_monitor(processed_count, total):
    """进度监控"""
    with tqdm(total=total, desc="Processing") as pbar:
        while True:
            current = processed_count.value
            pbar.update(current - pbar.n)
            if current >= total:
                break
            time.sleep(0.1)

def main(input_file, output_file, num_workers):
    """主函数"""
    # 共享变量初始化
    processed_count = Value('i', 0)
    deplication_count = Value('i', 0)
    manager = Manager()
    lock = Lock()

    # 创建共享的 MinHashLSH 对象
    lsh = manager.dict()
    lsh['lsh'] = MinHashLSH(threshold=0.8)  # 设置80%的相似度阈值

    # 队列初始化
    input_queue = Queue(maxsize=num_workers * 2)
    output_queue = Queue()

    # 停用词文件路径
    stopwords_files = {
        'chinese': os.path.join(current_directory, 'data/stop_words/cn_stopwords.txt'),
        'english': os.path.join(current_directory, 'data/stop_words/english.txt')
    }

    # 启动写入进程
    writer = Process(target=writer_process, args=(output_queue, output_file))
    writer.start()

    # 启动工作进程
    workers = []
    for _ in range(num_workers):
        p = Process(target=deduplication, args=(input_queue, output_queue, stopwords_files, lsh['lsh'], lock, deplication_count, processed_count))
        p.start()
        workers.append(p)

    total = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))  # 计算总行数

    # 启动进度监控进程
    monitor = Process(target=progress_monitor, args=(processed_count, total))
    monitor.start()

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            input_queue.put(line)

    # 发送结束信号
    for _ in range(num_workers):
        input_queue.put(None)

    for p in workers:
        p.join()

    monitor.terminate()

    output_queue.put(None)
    writer.join()

    print(f"文本总数量：{processed_count.value}, 已找到重复文本数量：{deplication_count.value}")

if __name__ == "__main__":
    # 修改为实际路径和文件名
    input_file = os.path.join(current_directory, './data/origin_data/sft.jsonl')
    output_file = os.path.join(current_directory, './data/origin_data/deduped_sft.jsonl')
    num_workers = 150
    main(input_file, output_file, num_workers)
