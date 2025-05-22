import logging
import os
import glob
from main import generate_paper_poster
from tqdm import tqdm
import concurrent.futures


def process_papers(input_dir, output_dir, url, model):
    os.makedirs(output_dir, exist_ok=True)

    paper_files = os.listdir(input_dir)
    pdf_files = [
        os.path.join(input_dir, file, "paper.pdf")
        for file in paper_files
        if os.path.isdir(os.path.join(input_dir, file))
    ]

    def process_single_pdf(pdf_file):
        try:
            file_id = os.path.basename(os.path.dirname(pdf_file))
            poster_dir = os.path.join(output_dir, file_id)
            os.makedirs(poster_dir, exist_ok=True)
            output_file = os.path.join(poster_dir, "poster.json")
            output_png = os.path.join(poster_dir, "poster.png")

            if os.path.exists(output_file) and os.path.exists(output_png):
                print(f"跳过已存在的文件: {output_file}")
                return

            generate_paper_poster(
                url=url,
                pdf=pdf_file,
                model=model,
                output=output_file,
                text_prompt=" ",
                figures_prompt=" ",
            )
            print(f"成功生成: {output_file}")

        except Exception as e:
            print(f"处理文件 {pdf_file} 时出错: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(process_single_pdf, pdf_file) for pdf_file in pdf_files
        ]

        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"处理文件 {model}",
        ):
            pass


if __name__ == "__main__":
    url = ""
    input_dir = "eval/data"
    models = []
    for model in models:
        output_dir = f"eval/temp-v2/{model.replace('/', '-')}"
        process_papers(input_dir, output_dir, url, model)
