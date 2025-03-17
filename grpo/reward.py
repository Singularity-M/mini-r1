import re
import torch.distributed as dist
from collections import Counter

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    match = re.search(r"\\boxed{(.*)}", answer)
    if match:
        validate = match.group(1)
        return validate.strip()
    else:
        return None
    

def correctness_reward_func(prompt, completions, answer, **kwargs) -> list[float]:
    """检查LLM输出的答案，根据除‘证’外共有字符比例赋分，最高2分"""
    responses = [completion.replace('<eos>', '') for completion in completions]
    extracted_responses = [extract_xml_answer(r.replace('<eos>', '')) for r in responses]
    
    q = prompt[0]['prompt'][-1]['content']
    if is_main_process():
        print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}",
              f"\nExtracted:\n{extracted_responses[0]}")
    
    scores = []
    for r, a in zip(extracted_responses, answer):
        a_str = str(a)
        r_str = str(r)
        # 去除所有‘证’字
        a_clean = a_str.replace('证', '')
        r_clean = r_str.replace('证', '')
        
        if len(r_clean) > len(a_clean):
            score = 0
        # 计算共有字符数
        else:
            a_counter = Counter(a_clean)
            r_counter = Counter(r_clean)
            total_common = 0
            for char, count in a_counter.items():
                if char in r_counter:
                    total_common += min(count, r_counter[char])
                # 按比例赋分，最高2分
            ratio = total_common / len(a_clean)
            score = min(ratio * 2, 2.0)
            
        scores.append(score)
    
    return scores

# def correctness_reward_func(prompt, completions, answer, **kwargs) -> list[float]:
#     """检查LLM输出的答案是否完全正确"""
#     responses = [completion.replace('<eos>','') for completion in completions]
#     extracted_responses = [extract_xml_answer(r.replace('<eos>','')) for r in responses]
    
#     q = prompt[0]['prompt'][-1]['content']
#     if is_main_process():
#         print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}",
#           f"\nExtracted:\n{extracted_responses[0]}")
    

#     return [2.0 if r == str(a) else 0.0 for r, a in zip(extracted_responses, answer)]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """检查LLM输出是否完全按照思维链的格式"""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>\n?$"
    responses = [completion.replace('<eos>','') for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """检查LLM输出是否存在符合思维链格式的部分"""
    pattern = r"<think>.*?</think>.*<answer>.*?</answer>"
    responses = [completion.replace('<eos>','') for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
    if text.count("\n</answer>") == 1:
        count += 0.125
    count -= (len(text.split("\n</answer>")[-1])) * 0.001  # 不以</answer>结尾扣除部分奖励分数
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """思维链不完整也给予一定的奖励分数"""
    contents = [completion.replace('<eos>','') for completion in completions]
    return [count_xml(c) for c in contents]



REWARD_FUNCS = {
    'correctness_reward_func': correctness_reward_func,
    'strict_format_reward_func': strict_format_reward_func,
    'soft_format_reward_func': soft_format_reward_func,
    'xmlcount_reward_func': xmlcount_reward_func
}