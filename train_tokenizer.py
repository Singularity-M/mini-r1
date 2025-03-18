import random
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import os

random.seed(42)

def train_tokenizer(data_path, tokenizer_dir):
    # 读取JSONL文件并提取文本数据
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['conversations'][0]['content'] + data['conversations'][1]['content']

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    special_tokens = ["<unk>", "<bos>", "<eos>", "<pad>", "<think>", "</think>"]

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=10000,
        special_tokens=special_tokens,  # 确保特殊的token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 读取文本数据
    texts = read_texts_from_jsonl(data_path)

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<bos>") == 1
    assert tokenizer.token_to_id("<eos>") == 2
    assert tokenizer.token_to_id("<pad>") == 3
    assert tokenizer.token_to_id("<think>") == 4
    assert tokenizer.token_to_id("</think>") == 5

    # 保存tokenizer
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<bos>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<eos>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "3": {
                "content": "<pad>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "4": {
                "content": "<think>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "5": {
                "content": "</think>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }

        },
        "additional_special_tokens": [],
        "bos_token": "<bos>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<eos>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": "<pad>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<bos>system\nYou are a helpful assistant.<eos>\n' }}{% endif %}{{'<bos>' + message['role'] + '\n' + message['content'] + '<eos>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<bos>assistant\n' }}{% endif %}"
    }

    # 保存配置文件
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved.")


def eval_tokenizer():

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model/minir1_tokenizer")
    # tokenizer = AutoTokenizer.from_pretrained("/home/mth/TCM_LLM/model/Qwen2.5-7B-Instruct")
    
    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": '是椭圆形的'},
        {"role": "assistant", "content": '456'},
        {"role": "user", "content": '456'},
        {"role": "assistant", "content": '789'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    print(new_prompt)
    # 获取词汇表大小（不包括特殊符号）
    print('tokenizer词表大小：', tokenizer.vocab_size)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print('实际词表长度：', actual_vocab_size)

    new_prompt = 'wenjie，椭圆和⚪的关系是什么呢？因为明天下午要带家人去下医院，所以申请上午在家办公，因为明天下午要带家人去下医院，所以申请上午在家办公，因为明天下午要带家人去下医院，所以申请上午在家办公，下午请半天假~@LWJWe '
    print(new_prompt)
    model_inputs = tokenizer(new_prompt)

    encoded = tokenizer.encode(new_prompt)

    vocab = tokenizer.get_vocab()

    # 反向映射：数字到 token
    # 获取反向词汇表
    reverse_vocab = {v: k for k, v in vocab.items()}

    # 使用数字索引还原为 tokens
    tokens = [reverse_vocab[num] for num in encoded]

    print("Decoded Tokens:", tokens)


    print(model_inputs)
    print('长度：', len(model_inputs['input_ids']))

    input_ids_ = model_inputs['input_ids']

    response = tokenizer.decode(input_ids_)
    print(response, end='')


if __name__ == '__main__':
    data_path = './data/origin_data/deduped_sft.jsonl'
    tokenizer_dir = "./model/minir1_tokenizer"
    # train_tokenizer(data_path, tokenizer_dir)
    eval_tokenizer()