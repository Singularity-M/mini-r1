import random
import time

import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniR1
from model.MiniR1Config import MiniR1Config
from model.model_lora import ModelWithLoRA

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/minir1_tokenizer')
    model_from = 1  # 1从权重，2用transformers

    if model_from == 1:
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/sft_{lm_config.dim}{moe_path}.pth'

        model = MiniR1(lm_config)
        state_dict = torch.load(ckp, map_location=device)

        # 处理不需要的前缀
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        for k, v in list(state_dict.items()):
            if 'mask' in k:
                del state_dict[k]

        # 加载到模型中
        model.load_state_dict(state_dict, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained('./minimind-v1-small', trust_remote_code=True)
    model = ModelWithLoRA(model)
    model.eval()
    model = model.to(device)

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')
    return model, tokenizer


def setup_seed(seed):
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置随机种子（如果有）
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置随机种子（如果有）
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动调优，避免不确定性


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    out_dir = 'out'
    start = ""
    temperature = 0.7
    top_k = 16
    # device = 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    max_seq_len = 4096
    lm_config = MiniR1Config()
    lm_config.max_seq_len = max_seq_len
    # 对话是否携带历史对话（当前模型没有在连续对话数据集上训练，增大历史上文基本不会有新的问答能力）
    contain_history_chat = False
    # -----------------------------------------------------------------------------

    model, tokenizer = init_model(lm_config)

    model = model.eval()
    # 推送到huggingface
    # model.push_to_hub("minimind")
    # tokenizer.push_to_hub("minimind")

    # answer_way = int(input('输入0自动测试，输入1问题测试：'))
    answer_way = 0
    stream = True
    rl_eval = True

    prompt_datas = [
        '患者于入院前3天，出现因食辛辣醇厚且劳累后出现肛旁肿胀疼痛，症情渐加重，遂来本院求治。刻下：肛旁肿胀疼痛剧烈，坐卧不宁，行走不利。大便，日行1次，质稀，排出畅，无便血，无排便不尽及肛门坠胀感，无粘液便，小溲畅，无发热恶寒。纳食可，夜寐尚可，舌红，苔黄，脉滑数。诊断：神志清晰，精神尚可，形体形体适中，语言清晰，口唇红润；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对',
        '患者1月来感全身酸痛不适，倦怠乏力，食纳欠佳，腰部隐痛，患者既往有骨质减少、腰椎间盘突出病史，为求进一步明确诊治入住我科；病程中患者神疲乏力，五心烦热，口干咽燥，耳轮干枯，腰膝酸软，脘腹胀满，食纳不香，二便尚调，夜寐可。诊断：神志清晰，精神尚可，形体适中，语言清晰，口唇淡红；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽淡红，双下肢无浮肿，舌淡体胖，苔白而干，脉沉细无力。',
        '患者1周前无明显诱因下出现肛周肿物脱出，肿痛难忍，行走不利，无便血，未予治疗。刻下：肛周肿物脱出，肿痛难忍，无便血，行走不利，坐卧不宁，无肛门坠胀不适。诊断：神志清晰，精神尚可，形体适中，语言清晰，口唇红润；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽红润，双下肢无浮肿，舌淡红，苔白，脉弦。',
        '患者于入院前3天，出现因食辛辣醇厚且劳累后出现肛内肿物外脱，伴便血，点滴而出，色鲜红，量多，未予治疗，症情渐加重，遂来本院求治。刻下：肛内肿物外脱，大便日行1次，质软，排出畅，伴便血，点滴而出，色鲜红，量多，偶伴排便不尽及肛门坠胀感，无粘液便，小溲畅，无发热恶寒。纳食可，夜寐尚可，舌红，苔黄，脉滑数。诊断：神志清晰，精神尚可，形体适中，语言清晰，口唇红润；皮肤正常，无斑疹。头颅大小形态正常，无目窼下陷，白睛无黄染，耳轮正常，无耳瘘及生疮；颈部对称，无青筋暴露，无瘿瘤瘰疬，胸部对称，虚里搏动正常，腹部平坦，无癥瘕痞块，爪甲色泽红润，双下肢无浮肿，舌淡红，苔白，脉滑数。',
    ]

    messages_origin = []
    messages = messages_origin

    i = 0
    while i < len(prompt_datas):
        # Generate a random seed
        random_seed = random.randint(0, 2 ** 32 - 1)
        setup_seed(random_seed)
        if not contain_history_chat:
            messages = messages_origin.copy()

        if answer_way == 1:
            prompt = input('[Q]: ')
        else:
            prompt = prompt_datas[i]
            print(f'[Q]: {prompt}')
            i += 1
        if rl_eval:
            messages.append({"role": "system", "content": "user和assistant之间的对话。user提出问题，assistant解决问题。assistant首先在头脑中思考推理过程，然后向用户提供答案。推理过程分别包含在<think> </think>标签中，答案包含在<answer> </answer>标签中，即<think>\n reasoning process here\n</think>\n<answer>\n answer here \n</answer>。"})
            prompt ="您必须将您的答案放在<answer> </answer>标签中，即<answer>该问题的答案</answer>。你的最终答案将通过\\boxed{{}}标签自动提取出来。\n以下是具体的问题：\n"+ "你是一名专业的中医医生，可以根据病人的病例信息给出病人的辩证。病人的病例：" + prompt
        else:
            prompt ="你好，"+ prompt
        
        messages.append({"role": "user", "content": prompt})

        # print(messages)
        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-(max_seq_len - 1):]

        x = tokenizer(new_prompt).data['input_ids']
        x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])

        answer = new_prompt

        with torch.no_grad():
            res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                                   top_k=top_k, stream=stream)

            print('[A]: ', end='')
            try:
                y = next(res_y)
            except StopIteration:
                print("No answer")
                continue

            history_idx = 0
            while y != None:
                answer = tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue
                # print(answer)
                if not len(answer):
                    try:
                        y = next(res_y)
                    except:
                        break
                    continue

                print(answer[history_idx:], end='', flush=True)
                # print(answer, end='', flush=True)
                try:
                    y = next(res_y)
                except:
                    break
                history_idx = len(answer)
                if not stream:
                    break

            print('\n')

        if contain_history_chat:
            assistant_answer = answer.replace(new_prompt, "")
            messages.append({"role": "assistant", "content": assistant_answer})