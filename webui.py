import gradio as gr
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from model.model import MiniR1
from model.MiniR1Config import MiniR1Config
from model.model_lora import ModelWithLoRA
from threading import Thread
import base64
js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

css = """
html, body {
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.gradio-container, .block-container {
    height: 100vh !important;
}
#chatbot {
    height: calc(100vh - 200px) !important;
}

"""

# 读取图片并转换为 Base64
with open("logo.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

warnings.filterwarnings('ignore')


tokenizer = AutoTokenizer.from_pretrained('./model/minir1_tokenizer')
model_from = 1  # 1从权重，2用transformers
device = "cuda:0"
lm_config = MiniR1Config(ntk=8)


moe_path = '_moe' if lm_config.use_moe else ''
ckp = f'./out/sft_long_{lm_config.dim}{moe_path}.pth'

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
model = ModelWithLoRA(model)
model.load_lora_weights("./out/tcm_lora_512.pth")

model.eval()
model = model.to("cuda:0")

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def predict(message, history, system_prompt, temperature, top_k, repetition_penalty, max_tokens): 
    top_k = int(top_k)
    max_tokens = int(max_tokens)

    history_transformer_format = history + [[message, ""]]  # 将聊天历史转换为 Transformer 模型所需的格式
    messages = []
    messages.append({"role": "user", "content": message})

        # print(messages)
    new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-(max_tokens - 1):]

    x = tokenizer(new_prompt).data['input_ids']
    x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])
    answers = ""

    with torch.no_grad():
        res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_tokens, temperature=temperature,
                                   top_k=top_k, repetition_penalty=repetition_penalty, stream=True)

        try:
            y = next(res_y)
        except StopIteration:
            print("")
        history_idx = 0
        while y != None:
            answer = tokenizer.decode(y[0].tolist())
            if answer and answer[-1] == '�':
                try:
                    y = next(res_y)
                except:
                    break
                continue
            if not len(answer):
                try:
                    y = next(res_y)
                except:
                    break
                continue
            yield answer[:]
            try:
                y = next(res_y)
            except:
                break
            history_idx = len(answer)
with gr.Blocks(title="Mini R1 power by singularity", theme=gr.themes.Soft(),  js=js_func, css=css) as demo:
    with gr.Sidebar(label="控制面板", elem_id="sidebar", open=False):
        description = gr.HTML(f"""
        <div style="
            border: 2px solid #00f3ff;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            box-shadow: 0 0 15px #00f3ff;
        ">
        <img src="data:image/png;base64,{encoded_image}" 
                 alt="AI Core" 
                 style="width:140%; border-radius: 4px;"/>
        </div>
        """)
        
        gr.Markdown("""
        ## <i class="fa fa-cogs" style="color: #00f3ff;"></i> 参数配置
        """)  
        additional_inputs=[
            gr.Textbox("You are helpful AI.", 
                    label="🤖 系统提示语",
                    max_lines=3),
            gr.Slider(0, 1, 
                    label="🌡️ 温度系数",
                    value=0.6,
                    info="0: 保守回答，1: 平衡模式"),
            gr.Slider(1, 100, step=1, 
                    label="🔝 Top-K采样",
                    value=16,
                    info="选择前K个最可能的词"),
            gr.Slider(minimum=1, 
                    maximum=2, 
                    label="🔄 重复惩罚系数",
                    value=1.0,
                    info="控制重复惩罚力度（1: 无惩罚，2: 强惩罚）"),
            gr.Slider(10, 4096, step=1,
                    label="📏 最大生成长度",
                    value=1024,
                    info="生成内容的最大token数"),
        ]
   
    webtitle=gr.Markdown("# Mini R1 Power by Singularity",)
    chat_interface = gr.ChatInterface(
            predict,
            additional_inputs=additional_inputs,
            chatbot=gr.Chatbot(elem_id="chatbot",value=[[None, "您好，我是MINI R1 由中国个人开发者开发的人工智能应用，请问你有什么问题嘛？"]]),
            # 以下可选配置可调整输入区域的高度
            textbox=gr.Textbox(container=False, stop_btn=True, submit_btn=True),
        )

def main():
    demo.launch(debug=True)


if __name__ == "__main__":
    main()
