import time
import yaml
import matplotlib.pyplot as plt
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
import base64
import json
from PIL import Image
import io
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any, Optional, Tuple, Union
import concurrent.futures


class ChecklistResponse(BaseModel):
    """评估结果的响应模型"""

    reason: str = Field(
        description="The reason for the score. It should include an interpretation of the evaluation criteria, the reasons for scoring and the reasons for deducting points."
    )
    score: int = Field(description="The score based on specified range.")


def load_checklist(path: str) -> dict:
    """从YAML文件加载检查清单"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compress_image(
    img_array: np.ndarray, max_size: int = 800, quality: int = 80
) -> str:
    """压缩图像以减小大小并返回base64编码的图像字符串

    Args:
        img_array: 图像数组
        max_size: 图像的最大尺寸（宽度或高度）
        quality: JPEG压缩质量（0-100）

    Returns:
        base64编码的图像字符串
    """

    if img_array.dtype == np.float32 and img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    elif img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)

    img = Image.fromarray(img_array)

    width, height = img.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        img = img.resize((new_width, new_height), Image.LANCZOS)

    buffer = io.BytesIO()

    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode()


def prepare_image_data(ground_truth: str, figure: Optional[str]) -> Optional[str]:
    """准备图像数据，返回base64编码的图像

    Args:
        ground_truth: 地面真实数据路径
        figure: 图像文件名（不含扩展名）

    Returns:
        base64编码的图像字符串，如果图像不存在则返回None
    """
    if figure is None:
        return None

    figure_path = f"{ground_truth}/{figure}.png"
    if not os.path.exists(figure_path):
        raise ValueError(f"Figure {figure} not found at {figure_path}")

    fig = plt.imread(figure_path)
    return compress_image(fig)


def create_human_message(
    poster_data: str,
    generated_poster_data: str,
    description: str,
    image_data: Optional[str],
    format_instructions: str,
) -> HumanMessage:
    """创建用于评估的人类消息

    Args:
        poster_data: 原始海报的base64数据
        generated_poster_data: 生成的海报的base64数据
        description: 评估标准描述
        image_data: 参考图像的base64数据（如果有）
        format_instructions: 输出格式说明

    Returns:
        格式化的HumanMessage对象
    """
    content = [
        {
            "type": "text",
            "text": "Below, you will find the official poster as the ground truth:",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{poster_data}"},
        },
    ]

    criteria_text = f"Here are the specific evaluation criteria: \n<criteria>{description}</criteria>.\nIt should be noted that you only need to consider the evaluation criteria and do not need to take into account other parts or its overall effect.\n"

    if image_data:
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"{criteria_text}\nAdditionally, here are the reference images from the official poster.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )
    else:
        content.append({"type": "text", "text": criteria_text})

    content.extend(
        [
            {
                "type": "text",
                "text": "Please carefully analyze whether the generated poster meets the evaluation criteria based on the official poster and provide your assessment.\nBelow is the generated poster to be evaluated:",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{generated_poster_data}"},
            },
            {"type": "text", "text": f"\n{format_instructions}"},
        ]
    )

    return HumanMessage(content=content)


def evaluate_checklist_item(
    llm: BaseChatOpenAI,
    system_prompt: SystemMessage,
    parser: JsonOutputParser,
    poster_data: str,
    generated_poster_data: str,
    item: Dict[str, Any],
    ground_truth: str,
    is_common: bool = False,
) -> Dict[str, Any]:
    """评估单个检查清单项

    Args:
        llm: 语言模型
        system_prompt: 系统提示
        parser: JSON输出解析器
        poster_data: 原始海报的base64数据
        generated_poster_data: 生成的海报的base64数据
        item: 检查清单项
        ground_truth: 地面真实数据路径
        is_common: 是否为通用检查清单项

    Returns:
        评估结果字典
    """
    description = item["description"]
    figure = item.get("figure")

    if is_common:
        max_score = 5  # 通用检查项的最大分数为5
        score_range = "0-5"
    else:
        if item.get("max_score") is None:
            print(f"检查项 '{description}' 没有指定最大分数，使用默认值5")
            max_score = 5
        else:
            max_score = item.get("max_score")
        score_range = f"0-{max_score}"

    max_retries = 10
    attempts = 0

    while attempts < max_retries:
        try:
            image_data = prepare_image_data(ground_truth, figure)

            format_instructions = parser.get_format_instructions()

            score_instruction = f"Please provide a score between 0 and {max_score}. Firstly output the reason, then output the score. Only output a JSON instance, do not output anything else.\n{format_instructions}"

            human_message = create_human_message(
                poster_data,
                generated_poster_data,
                description,
                image_data,
                score_instruction,
            )

            prompt = ChatPromptTemplate.from_messages([system_prompt, human_message])
            chain = prompt | llm | parser

            response = chain.invoke({})

            if 0 <= response["score"] <= max_score:
                return {
                    "reason": response["reason"],
                    "score": response["score"],
                    "max_score": max_score,  # 添加最大分数信息
                }
            else:
                print(f"分数超出范围 ({score_range}): {response['score']}，重新尝试...")
                attempts += 1

        except Exception as e:
            print(f"评估检查项 '{description}' 时出错: {e}")
            attempts += 1

    if attempts == max_retries:
        print(f"多次尝试后仍无法获取有效评估结果，使用默认值")
        return {
            "reason": f"评估失败，无法获取有效结果。",
            "score": -1,
            "max_score": max_score,
        }


def eval_checklist(
    ground_truth: str, generated: str, task_name: str = "all"
) -> Dict[str, Dict[str, Any]]:
    """评估生成的海报是否符合检查清单的要求

    Args:
        ground_truth: 地面真实数据路径
        generated: 生成的海报路径
        task_name: 评估任务类型，可选值为 "all", "common", "specific"

    Returns:
        评估结果字典
    """
    result = {"common": {}, "specific": {}}

    checklist = load_checklist(f"{ground_truth}/checklist.yaml")
    checklist_common = load_checklist("eval/common.yaml")

    def load_image(path):
        try:
            return plt.imread(path)
        except:
            from PIL import Image

            img = Image.open(path)
            return np.array(img)

    poster = load_image(f"{ground_truth}/poster.png")
    poster_data = compress_image(poster)

    generated_poster = load_image(generated)
    generated_poster_data = compress_image(generated_poster)

    llm = BaseChatOpenAI(
        model="gpt-4o-2024-11-20",
        temperature=0,
        max_tokens=16000,
    )

    parser = JsonOutputParser(pydantic_object=ChecklistResponse)

    system_prompt = SystemMessage(
        content="""You are an expert reviewer. Please assist me in evaluating whether the poster I generated meets the requirements outlined in the official poster's checklist (serving as the ground truth)."""
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}

        if task_name in ["all", "specific"]:
            specific_futures = {
                executor.submit(
                    evaluate_checklist_item,
                    llm,
                    system_prompt,
                    parser,
                    poster_data,
                    generated_poster_data,
                    item,
                    ground_truth,
                    False,
                ): item["description"]
                for item in checklist["checklist"]
            }
            futures.update(specific_futures)

        if task_name in ["all", "common"]:
            common_futures = {
                executor.submit(
                    evaluate_checklist_item,
                    llm,
                    system_prompt,
                    parser,
                    poster_data,
                    generated_poster_data,
                    item,
                    ground_truth,
                    True,
                ): item["description"]
                for item in checklist_common["checklist"]
            }
            futures.update(common_futures)

        for future in concurrent.futures.as_completed(futures):
            description = futures[future]
            try:
                result_item = future.result()

                if any(
                    item["description"] == description
                    for item in checklist_common["checklist"]
                ):
                    result["common"][description] = result_item
                else:
                    result["specific"][description] = result_item
            except Exception as exc:
                print(f"检查项 '{description}' 生成了异常: {exc}")
                error_result = f"Error: {str(exc)}"
                if any(
                    item["description"] == description
                    for item in checklist_common["checklist"]
                ):
                    result["common"][description] = error_result
                else:
                    result["specific"][description] = error_result

    return result


if __name__ == "__main__":
    ground_truth = "eval/data/2406.16441v1"
    generated = " "
    results = eval_checklist(ground_truth, generated)

    with open("eval/data/2406.16441v1/results.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
