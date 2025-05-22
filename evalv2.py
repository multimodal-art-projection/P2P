import json
import os
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
import base64
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from rouge import Rouge
from bert_score import BERTScorer
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
import re

from eval.eval_checklist import compress_image, eval_checklist
from eval.predict_with_xgboost import load_common_yaml, predict_score


bert_scorer = None
rouge = None
xgboost_model = None


def init():
    """全局初始化"""
    global bert_scorer
    global rouge
    global xgboost_model

    bert_scorer = BERTScorer(lang="en")

    rouge = Rouge()

    try:
        model_path = "eval/plots/xgboost_model.joblib"
        xgboost_model = joblib.load(model_path)
        print(f"成功加载XGBoost模型: {model_path}")
    except Exception as e:
        print(f"加载XGBoost模型失败: {e}")
        xgboost_model = None


def load_and_process_text(file_path: str) -> str:
    """加载并处理文本内容"""
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            content = json.load(f)
            content = json.dumps(content)
            return remove_image_urls(content)
    else:
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        return "\n".join([page.page_content for page in pages])


def remove_image_urls(text: str) -> str:
    """移除文本中的图片URL"""
    pattern = r"!\[.*?\]\((.*?)\)"
    return re.sub(pattern, "", text)


def evaluate_rouge(text1: str, text2: str) -> Dict:
    """ROUGE评估"""
    scores = rouge.get_scores(text1, text2)[0]
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }


def evaluate_bert(text1: str, text2: str) -> Dict:
    """BERT Score评估"""
    precision, recall, f1 = bert_scorer.score([text1], [text2])
    return {
        "bert_precision": float(precision[0]),
        "bert_recall": float(recall[0]),
        "bert_f1": float(f1[0]),
    }


def parse_json_response(content: str) -> dict:
    """解析JSON响应，支持纯JSON和代码块格式"""

    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    return json.loads(content)


def evaluate_poster_comparison(
    paper_content: str, poster_a_path: str, poster_b_path: str
) -> str:
    """比较两张海报并返回哪一个更好 (A/B/C)"""
    client = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0)

    COMPARISON_PROMPT = """
    You are evaluating two academic posters (A and B) created from the same research paper. 
    
    Please compare these posters based on:
    1. Content accuracy and completeness
    2. Visual design and layout effectiveness
    3. Overall presentation quality
    
    Original Paper Content for Reference:
    {paper_content}

    The first image is Poster A, and the second image is Poster B.
    
    Provide your evaluation in JSON format with only one field:
    - Response: Your detailed evaluation analysis
    - Analysis: Comparative analysis of strengths and weaknesses
    - Judgement: Use "A" if Poster A is better, "B" if Poster B is better. Only output one letter("A", "B").
    
    Note: Focus on objective criteria. Your response should contain ONLY the JSON with the Judgement field.
    """

    def validate_and_extract_judgement(response_content: str) -> str:
        """验证响应并提取单字母判断"""
        try:
            result = parse_json_response(response_content)
            if not isinstance(result, dict) or "Judgement" not in result:
                return None

            judgement = result["Judgement"].strip().upper()
            judgement = "".join(c for c in judgement if c in "ABC")

            return judgement if len(judgement) == 1 and judgement in "ABC" else None

        except Exception:
            return None

    with open(poster_a_path, "rb") as f1, open(poster_b_path, "rb") as f2:
        poster_a_image = base64.b64encode(f1.read()).decode()
        poster_b_image = base64.b64encode(f2.read()).decode()

    prompt = COMPARISON_PROMPT.format(paper_content=paper_content)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.invoke(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{poster_a_image}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{poster_b_image}"
                                },
                            },
                        ],
                    }
                ]
            )

            judgement = validate_and_extract_judgement(response.content)
            if judgement:
                return judgement

        except Exception as e:
            if "image is too large" in str(e):
                poster_a_image = compress_image(plt.imread(poster_a_path))
                poster_b_image = compress_image(plt.imread(poster_b_path))
            if attempt == max_retries - 1:
                raise ValueError(f"海报对比评估失败，已重试{max_retries}次: {str(e)}")
            continue

    raise ValueError(f"在{max_retries}次尝试后未能获得有效判断")


def extract_features_from_results(common_results):
    """从评估结果中提取特征，用于XGBoost预测"""
    common_items = load_common_yaml("eval/common.yaml")
    features = np.zeros(len(common_items))

    for i, item in enumerate(common_items):
        if item in common_results:
            score = common_results[item].get("score", 0)

            if score < 0:
                score = 0
            features[i] = score

    return features


def predict_with_xgboost(common_results):
    """使用XGBoost模型预测分数"""
    global xgboost_model

    if xgboost_model is None:
        print("XGBoost模型未初始化，尝试加载...")
        try:
            model_path = "eval/xgboost_model.joblib"
            xgboost_model = joblib.load(model_path)
        except Exception as e:
            print(f"加载XGBoost模型失败: {e}")
            return None

    try:
        features = extract_features_from_results(common_results)

        score = predict_score(features, xgboost_model)

        return float(score)
    except Exception as e:
        print(f"XGBoost预测失败: {e}")
        return None


def calculate_checklist_score(specific_results):
    """计算checklist得分的归一化值（总得分/总max_score*100）"""
    if not specific_results:
        return 0.0

    total_score = 0.0
    total_max_score = 0.0

    for item_name, item_data in specific_results.items():
        score = item_data.get("score", 0)
        max_score = item_data.get("max_score", 5)  # 默认最大分数为5

        if score < 0:
            score = 0

        total_score += score
        total_max_score += max_score

    if total_max_score == 0:
        return 0.0

    return (total_score / total_max_score) * 100.0


def evaluate_pair_v2(model: str, file_id: str) -> Dict:
    """评估单个模型-文件对"""
    base_path = "eval"

    docmesh = f"{base_path}/temp-v2/{model}/{file_id}/poster.json"
    picture = f"{base_path}/temp-v2/{model}/{file_id}/poster.png"
    paper = f"{base_path}/data/{file_id}/paper.pdf"
    poster = f"{base_path}/data/{file_id}/poster.pdf"
    poster_pic = f"{base_path}/data/{file_id}/poster.png"
    ground_truth = f"{base_path}/data/{file_id}"

    if not os.path.exists(docmesh):
        print(f"警告: 未找到生成的文本 {docmesh}")
    if not os.path.exists(picture):
        print(f"警告: 未找到生成的图片 {picture}")
    if not os.path.exists(paper):
        print(f"警告: 未找到原始论文 {paper}")
    if not os.path.exists(poster):
        print(f"警告: 未找到原始海报 {poster}")
    if not os.path.exists(poster_pic):
        print(f"警告: 未找到原始海报图片 {poster_pic}")

    docmesh_content = load_and_process_text(docmesh) if os.path.exists(docmesh) else ""
    paper_content = load_and_process_text(paper) if os.path.exists(paper) else ""
    poster_content = load_and_process_text(poster) if os.path.exists(poster) else ""

    text_metrics = {}
    if docmesh_content and poster_content:
        text_metrics.update(evaluate_rouge(docmesh_content, poster_content))
        text_metrics.update(evaluate_bert(docmesh_content, poster_content))

    image_metrics = {}
    if os.path.exists(picture) and os.path.exists(poster_pic):
        if paper_content:
            try:
                image_metrics["judge"] = evaluate_poster_comparison(
                    paper_content, poster_pic, picture
                )
            except Exception as e:
                print(f"海报比较评估失败: {e}")
                image_metrics["judge"] = None

    checklist_results = {"common": {}, "specific": {}}
    if os.path.exists(picture) and os.path.exists(ground_truth):
        try:
            checklist_results = eval_checklist(ground_truth, picture)
        except Exception as e:
            print(f"Checklist评估失败: {e}")

    checklist_score = calculate_checklist_score(checklist_results.get("specific", {}))

    human_score = None
    if checklist_results and checklist_results["common"]:
        try:
            human_score = predict_with_xgboost(checklist_results["common"])
        except Exception as e:
            print(f"XGBoost评分失败: {e}")

    results = {
        "model": model,
        "file_id": file_id,
        "text_metrics": text_metrics,
        "image_metrics": image_metrics,
        "common": checklist_results.get("common", {}),
        "specific": checklist_results.get("specific", {}),
        "checklist": checklist_score,
        "human": human_score,
    }

    return results


def process_single_pair_v2(args):
    """处理单个评估对"""
    model, file_id = args
    try:
        result = evaluate_pair_v2(model, file_id)
        return result
    except Exception as e:
        print(f"评估失败 {model}-{file_id}: {str(e)}")
        return None


def batch_evaluate_v2(
    model_file_pairs: List[Tuple[str, str]], max_workers: int = 8
) -> List[Dict]:
    """批量评估多个模型-文件对"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(
            tqdm(
                executor.map(process_single_pair_v2, model_file_pairs),
                total=len(model_file_pairs),
                desc="评估进度",
            )
        )

        results = [result for result in futures if result is not None]

    return results


if __name__ == "__main__":
    init()
    pairs = [("gpt-4o-2024-11-20", "6562"), ("qwen", "10488")]
    results = batch_evaluate_v2(pairs)
    print(json.dumps(results, indent=2, ensure_ascii=False))
