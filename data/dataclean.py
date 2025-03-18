import os
import re
import jieba
import fasttext
import json
import zhconv
import time

from tqdm import tqdm
from multiprocessing import Process, Queue, Value, Lock, Manager
from transformers import AutoTokenizer

# 获取项目根目录
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


def count_chinese_and_english(text):
    """统计文本中的中英文文本数量"""
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    # 匹配英文单词（以空格分隔的字母序列）
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    return len(chinese_chars), len(english_words)


def DeduplicatorWordLevel(words, threshold=3, n_gram_threshold=15):
    """连续重复词检查+2gram去重"""
    if len(words) <= 1:
        return False

    max_count = 1 
    current_count = 1  
    
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            current_count += 1  
        else:
            max_count = max(max_count, current_count)  
            current_count = 1
    res = {}
    n_gram_max_count = 1
    for i in range(1, len(words)):
        if ''.join(words[i-1:i+1]) not in res:
            res[''.join(words[i-1:i+1])] = 1
        else:
            res[''.join(words[i-1:i+1])] += 1

    n_gram_max_count = max(res.values())
    
    if max_count >= threshold or n_gram_max_count >= n_gram_threshold:
        return True
    else:
        return False

def preprocessing(text, punctuation_pattern, combined_stopwords):
    """文本预处理"""
    text = re.sub(punctuation_pattern, '', text)
    words = jieba.lcut(text)
    words = [word for word in words if word not in combined_stopwords and word.strip()]
    return words

def is_code(text):
    """简单代码检测"""
    code_keywords = ['def ', 'class ', 'import ', 'public ', 'private ', 'function ', 'var ', 'let ', 'const ']
    return any(keyword in text for keyword in code_keywords)

def is_pure_chinese_or_english(text, words, pattern, symbol_threshold=0.3):
    """判断文本是否包含非中英文字符或符号污染"""
    
    words_str = ''.join(words)
    non_en_or_zh = re.findall(pattern, words_str)
    if (len(non_en_or_zh) + 1) / (len(words_str) + 1) > 0.2:
        return False
    
    #符号污染判断
    non_english_or_symbols = re.findall(pattern, text)
    symbol_ratio = (len(non_english_or_symbols) + 1) / (len(words_str) + 1)
    return symbol_ratio <= symbol_threshold

def worker_process(input_queue, output_queue, processed_count, zh_count, en_count, code_count, lock, model_path, stopwords_files, tokenizer_file):
    """工作进程处理函数"""
    jieba.initialize()
    model = fasttext.load_model(model_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file)
    
    
    # 加载停用词
    chinese_stopwords = load_stopwords(stopwords_files['chinese'])
    english_stopwords = load_stopwords(stopwords_files['english'])
    combined_stopwords = chinese_stopwords.union(english_stopwords)
    
    # 编译正则表达式
    punctuation_pattern = re.compile(r'[，。！？；：“”‘’（）《》〈〉【】、~@#￥%……&*——+=|"<>?/\\\[\]!?;:"()<>{}\']')
    pure_pattern = re.compile(r'[^\u4e00-\u9fffA-Za-z0-9\s]')

    while True:
        line = input_queue.get()

        
        if line is None:
            break
        
        with lock:
            processed_count.value += 1
        
        try:
            data = json.loads(line.strip())
            text = data['conversations'][0]['content'] + data['conversations'][1]['content']

        except:
            continue

        #繁体字转换为简体字
        tokens = tokenizer.apply_chat_template(data['conversations'])

        if len(tokens) > 4000:
            continue

        data['tokens'] = len(tokens)

        text = zhconv.convert(text, 'zh-hans')
        # if re.search(r"Memory\s*usage\s*\d+MB\n", text):
        #     continue
        if is_code(text):
            with lock:
                code_count.value += 1
            output_queue.put(data)
            continue
        
        words = preprocessing(text, punctuation_pattern, combined_stopwords)

        if DeduplicatorWordLevel(words):
            continue
        
        if not is_pure_chinese_or_english(text, words, pure_pattern):
            continue
        
        # 鉴别文本语言
        lang = model.predict(text.replace("\n", ""), k=1)[0][0]
        
        if lang == '__label__zh':
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            total_chars = len(text)
            if (english_chars + 1)/ (total_chars + 1) > 0.25:
                continue
            with lock:
                zh_count.value += 1
            output_queue.put(data)
        
        elif lang == '__label__en':
            with lock:
                current_zh = zh_count.value
                current_en = en_count.value
                new_en = current_en + 1
                total = current_zh + new_en
                if total == 0:
                    continue
                if new_en / total <= 0.2:
                    en_count.value = new_en
                    output_queue.put(data)

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

def main(input_path, output_file, num_workers):
    # 共享变量初始化
    processed_count = Value('i', 0)
    zh_count = Value('i', 0)
    en_count = Value('i', 0)
    code_count = Value('i', 0)
    lock = Lock()

    input_files = [os.path.join(root, file)
               for root, dirs, files in os.walk(input_path)
               for file in files if file.endswith('.jsonl')]
    
    # 队列初始化
    input_queue = Queue(maxsize=num_workers*2)
    output_queue = Queue()
    
    # 路径配置
    model_path = './data/fasttext_model/lid.176.bin'
    #预处理阶段可以使用一些训练好的tokenizer
    tokenizer_file = './Qwen2.5-7B-Instruct'
    stopwords_files = {
        'chinese': os.path.join(current_directory, 'data/stop_words/en_stopwords.txt'),
        'english': os.path.join(current_directory, 'data/stop_words/en_stopwords.txt')
    }
    
    # 启动写入进程
    writer = Process(target=writer_process, args=(output_queue, output_file))
    writer.start()
    
    # 启动工作进程
    workers = []
    for _ in range(num_workers):
        p = Process(target=worker_process,
                    args=(input_queue, output_queue, processed_count,
                          zh_count, en_count, code_count, lock, model_path,
                          stopwords_files, tokenizer_file))
        p.start()
        workers.append(p)
    
    total = get_total_lines(input_files)
    
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
    
    
    # 读取输入文件
    for input_file in sorted(input_files)[::-1]:
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
    
    total_text_count = zh_count.value + en_count.value + code_count.value
    print(f"处理完成，删除文本占比：{((total-total_text_count)/total):.2f}，中文文本数占比：{(zh_count.value/total_text_count):.2f}，英文文本数占比：{(en_count.value/total_text_count):.2f}，代码文本数占比：{(code_count.value/total_text_count):.2f}")

if __name__ == "__main__":

    
    input_path = os.path.join(current_directory, './data/original_data')
    output_file = os.path.join(current_directory, './data/data_clean.jsonl')
    num_workers = 100

    main(input_path, output_file, num_workers)
        


