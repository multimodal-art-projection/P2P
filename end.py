import os
import json
import glob
import sys
from evalv2 import init, evaluate_pair_v2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading


def get_model_file_pairs(base_dir):
    """获取所有需要评估的模型-文件对"""
    pairs = []


    model_dirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    for model in model_dirs:

        for root, dirs, files in os.walk(os.path.join(base_dir, model)):

            png_files = glob.glob(os.path.join(root, "poster.png"))
            for png_file in png_files:

                file_id = os.path.basename(os.path.dirname(png_file))


                required_files = [
                    f"eval/data/{file_id}/poster.png",
                    f"eval/data/{file_id}/checklist.yaml",
                ]

                if all(os.path.exists(f) for f in required_files):
                    pairs.append((model, file_id))
                else:
                    print(f"警告: 跳过 {model}/{file_id}，缺少必要文件")

    return pairs


def is_valid_result(result):
    """检查结果是否有效（不包含null值）"""
    if result is None:
        return False

    def check_dict(d):
        for v in d.values():
            if v is None:
                return False
            if isinstance(v, dict):
                if not check_dict(v):
                    return False
        return True

    return check_dict(result)



file_lock = threading.Lock()


def save_result(result, result_file):
    """线程安全地保存单个结果"""
    if result and is_valid_result(result):
        with file_lock:
            with open(result_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        if result:
            print(f"跳过无效结果: {result['model']}/{result['file_id']}")


def process_single_pair(args):
    """处理单个评估对"""
    model, file_id = args
    try:
        result = evaluate_pair_v2(model, file_id)
        return result
    except Exception as e:
        print(f"评估失败 {model}-{file_id}: {str(e)}")
        return None


def main(base_dir):
    print("初始化评估器...")
    init()


    result_file = f"{base_dir}/results.jsonl"


    evaluated_pairs = set()
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    evaluated_pairs.add((result["model"], result["file_id"]))
                except:
                    continue


    all_pairs = get_model_file_pairs(base_dir)


    pending_pairs = [pair for pair in all_pairs if pair not in evaluated_pairs]

    if not pending_pairs:
        print("所有文件都已评估完成")
        return

    print(f"开始评估 {len(pending_pairs)} 个文件...")


    max_workers = 2  # 可以根据需要调整线程数
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = []
        for pair in pending_pairs:
            future = executor.submit(process_single_pair, pair)
            futures.append(future)


        for future in tqdm(futures, total=len(pending_pairs), desc="评估进度"):
            try:
                result = future.result()

                save_result(result, result_file)
            except Exception as e:
                print(f"处理结果时出错: {str(e)}")


if __name__ == "__main__":
    sys.setrecursionlimit(16000)
    base_dir = "eval/temp-v2"

    main(base_dir)
