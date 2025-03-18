import glob
import os
import json
import re

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

# 配置PDF文件目录路径
pdf_directory_path = "./data/pdf"
pdf_search_pattern = os.path.join(pdf_directory_path, "*.pdf")
# 使用glob.glob查找匹配的PDF文件列表
matching_pdf_files = glob.glob(pdf_search_pattern)

for pdf_file_path in matching_pdf_files:
    # 获取PDF文件名（不含扩展名）
    file_name_without_extension = pdf_file_path.split("/")[-1].split(".")[0].strip()

    # 准备环境：创建输出目录
    local_image_output_dir = f"{pdf_directory_path}/output/images"
    local_markdown_output_dir = f"{pdf_directory_path}/output"
    image_output_dir_name = os.path.basename(local_image_output_dir)

    os.makedirs(local_image_output_dir, exist_ok=True)

    # 初始化文件读写器
    image_writer = FileBasedDataWriter(local_image_output_dir)
    markdown_writer = FileBasedDataWriter(local_markdown_output_dir)

    # 读取PDF文件内容
    pdf_reader = FileBasedDataReader("")
    pdf_content_bytes = pdf_reader.read(pdf_file_path)

    # 创建数据集实例
    dataset = PymuDocDataset(pdf_content_bytes)

    # 根据PDF类型进行处理
    if dataset.classify() == SupportedPdfParseMethod.OCR:
        inference_result = dataset.apply(doc_analyze, ocr=True)
        pipeline_result = inference_result.pipe_ocr_mode(image_writer)
    else:
        inference_result = dataset.apply(doc_analyze, ocr=False)
        pipeline_result = inference_result.pipe_txt_mode(image_writer)

    # 输出Markdown文件
    pipeline_result.dump_md(markdown_writer, f"{file_name_without_extension}.md", image_output_dir_name)

# 配置Markdown文件搜索模式
markdown_search_pattern = os.path.join(local_image_output_dir, "*.md")
matching_markdown_files = glob.glob(markdown_search_pattern)

# 配置输出JSONL文件路径
output_jsonl_file_path = "./data/original/name.jsonl"

# 定义正则表达式过滤规则
exclude_patterns = [r"^\!\[*", r"^表\d*", r"^图\d*"]

# 创建并写入JSONL文件
with open(output_jsonl_file_path, mode="w", encoding="utf-8") as output_file:
    # 遍历匹配到的Markdown文件
    for markdown_file_path in matching_markdown_files:
        with open(markdown_file_path, mode="r", encoding="utf-8") as markdown_file:
            current_line = ""
            previous_line = ""
            content_buffer = ""

            for line in markdown_file:
                if line.strip():
                    # 处理第一行
                    if current_line == "":
                        current_line = line
                        previous_line = line
                    else:
                        previous_line = current_line
                        current_line = line

                    # 检查是否为标题行
                    if "#" in current_line:
                        if "#" not in previous_line:  # 处理新的标题行
                            if content_buffer.strip():  # 写入缓冲区内容
                                data = {
                                    "conversations": [
                                        {"role": "user", "content": f"{content_buffer.lstrip().rstrip()}"},
                                        {"role": "assistant", "content": f""}
                                    ]
                                }
                                output_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                            content_buffer = current_line.replace("#", "").lstrip().rstrip() + "\n"
                        else:  # 标题行连续出现，更新缓冲区
                            content_buffer = current_line.replace("#", "").lstrip().rstrip() + "\n"
                    else:
                        # 过滤不符合规则的行并添加到缓冲区
                        if not any(re.match(pattern, current_line) for pattern in exclude_patterns):
                            content_buffer += current_line.lstrip().rstrip()

            # 处理文件剩余内容并写入
            if content_buffer.strip():
                data = {
                    "conversations": [
                        {"role": "user", "content": f"{content_buffer.lstrip().rstrip()}"},
                        {"role": "assistant", "content": f""}
                    ]
                }
                output_file.write(json.dumps(data, ensure_ascii=False) + "\n")