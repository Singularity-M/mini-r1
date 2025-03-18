import os
# 获取项目根目录
current_directory = os.getcwd()

import re
import jieba
import fasttext
import json
import time

from tqdm import tqdm
from multiprocessing import Process, Queue, Value, Lock, Manager

def get_total_lines(input_file):
    """计算输入文件的行数"""
    with open(input_file, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def load_stopwords(stopwords_file):
    """加载停用词"""
    stopwords = set()
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords

stopwords_files = {
        'chinese': os.path.join(current_directory, 'data/stop_words/en_stopwords.txt'),
        'english': os.path.join(current_directory, 'data/stop_words/en_stopwords.txt')
    }

punctuation_pattern = re.compile(r'[，。！？；：“”‘’（）《》〈〉【】、~@#￥%……&*——+=|"<>?/\\\[\]!?;:"()<>{}\']')
chinese_stopwords = load_stopwords(stopwords_files['chinese'])
english_stopwords = load_stopwords(stopwords_files['english'])
combined_stopwords = chinese_stopwords.union(english_stopwords)


def preprocessing(text):
    """文本预处理"""
    text = re.sub(punctuation_pattern, '', text)
    words = jieba.lcut(text)
    words = [word for word in words if word not in combined_stopwords and word.strip()]
    return words


def writer_process(output_queue, output_file):
    """写入进程处理函数"""
    buffer = []
    while True:
        item = output_queue.get()
        if item is None:
            if buffer:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for text in buffer:
                        f.write(' '.join(text) + '\n')
            break
        buffer.append(item)
        if len(buffer) >= 10000:
            with open(output_file, 'a', encoding='utf-8') as f:
                for text in buffer:
                    f.write(' '.join(text) + '\n')
            buffer.clear()


def process_large_file(input_queue, model_path,  zh_output_queue, en_output_queue, processed_count, lock):

    model = fasttext.load_model(model_path)
    while True:
        line = input_queue.get()
        if line is None:
            break

        with lock:
            processed_count.value += 1

        data = json.loads(line.strip())
        text = data['text']
        lang = model.predict(text.replace("\n", ""), k=1)[0][0]
        if lang == '__label__zh':
            words = preprocessing(line.strip())
            zh_output_queue.put(words)
                        
        elif lang == '__label__en':
            words = preprocessing(line.strip())
            en_output_queue.put(words)
        

def main(file_path, model_path, zh_output_file, en_output_file, num_workers):
    processed_count = Value('i', 0)
    lock = Lock()

    input_queue = Queue(maxsize=num_workers*2)
    zh_output_queue = Queue()
    en_output_queue = Queue()

    total = get_total_lines(file_path)

    # 启动写入进程
    zh_writer = Process(target=writer_process, args=(zh_output_queue, zh_output_file))
    zh_writer.start()

    en_writer = Process(target=writer_process, args=(en_output_queue, en_output_file))
    en_writer.start()

    workers = []
    for _ in range(num_workers):
        p = Process(target=process_large_file,
                    args=(input_queue, model_path,  zh_output_queue, en_output_queue, processed_count, lock))
        p.start()
        workers.append(p)

    # 进度监控
    def progress_monitor():
        with tqdm(total=total, desc="Processing") as pbar:
            while True:
                current = processed_count.value
                pbar.update(current - pbar.n)
                if current >= total:
                    break
                time.sleep(0.1)
    
    monitor = Process(target=progress_monitor)
    monitor.start()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            input_queue.put(line)

    for _ in range(num_workers):
        input_queue.put(None)
    
    for p in workers:
        p.join()
    monitor.terminate()
    
    zh_output_queue.put(None)
    zh_writer.join()

    en_output_queue.put(None)
    en_writer.join()

    print("完成文档处理")


import kenlm
model = kenlm.Model('/home/mth/project_llm/mini_llm/data/n-gram/model/mobvoi.bin')
file_path = os.path.join(current_directory, 'data/pretrain_data/pretrain_hq.jsonl')  
with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line)['text']
            words = preprocessing(text)
            print(model.score(' '.join(words), bos=True, eos=True)/len(words))



if __name__ == "__main__":

    jieba.initialize()
    file_path = os.path.join(current_directory, 'data/origin_data/mobvoi_seq_monkey_general_open_corpus_1.jsonl')  
    model_path = os.path.join(current_directory, 'data/fasttext_model/lid.176.bin')
    zh_output_file = os.path.join(current_directory, 'data/n-gram/zh_n_gram_train.txt')
    en_output_file= os.path.join(current_directory, 'data/n-gram/en_n_gram_train.txt')

    os.makedirs(os.path.dirname(zh_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(en_output_file), exist_ok=True)

    num_workers = 50

    main(file_path, model_path, zh_output_file, en_output_file, num_workers)
