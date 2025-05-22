import base64
import copy
import json
import fire
import os
import pathlib

from poster.figures import extract_figures
from poster.poster import (
    generate_html,
    generate_html_v2,
    generate_latex,
    generate_poster_v3,
    generate_svg,
    latex_to_png,
    replace_figures_in_poster,
    replace_figures_size_in_poster,
    svg_to_png,
    take_screenshot,
)


def generate_paper_poster(
    url: str,
    pdf: str,
    vendor: str = "openai",
    model: str = "gpt-4o-mini",
    text_prompt: str = "",
    figures_prompt: str = "",
    output: str = "poster.json",
):
    """Generate a paper poster

    Args:
        url: URL of the PDF file
        pdf: Local path of the PDF file
        model: Name of the model to use, default is gpt-4o-mini
        text_prompt: Text prompt template,
        figures_prompt: Figures prompt template,
        output: Output file path, default is poster.json
    """
    pdf_stem = pdf.replace(".pdf", "")
    figures_cache = f"{pdf_stem}_figures.json"
    figures_cap_cache = f"{pdf_stem}_figures_cap.json"

    figures = []
    figures_cap = []
    if os.path.exists(figures_cache) and os.path.exists(figures_cap_cache):
        print(f"使用缓存的图片: {figures_cache}")
        with open(figures_cache, "r") as f:
            figures = json.load(f)
        with open(figures_cap_cache, "r") as f:
            figures_cap = json.load(f)
    else:
        figures_img = extract_figures(url, pdf, task="figure")
        figures_table = extract_figures(url, pdf, task="table")
        img_caption = extract_figures(url, pdf, task="figurecaption")
        table_caption = extract_figures(url, pdf, task="tablecaption")
        threshold = 0.85
        while True:
            figures = [
                image
                for image, score in figures_img + figures_table
                if score >= threshold
            ]
            figures_cap = [
                image
                for image, score in img_caption + table_caption
                if score >= threshold
            ]
            print(f"{threshold:.2f} 提取到 {len(figures)} / {len(figures_cap)} 张图像")
            if len(figures) == len(figures_cap):
                break
            threshold -= 0.05

        with open(figures_cache, "w") as f:
            json.dump(figures, f, ensure_ascii=False)
        with open(figures_cap_cache, "w") as f:
            json.dump(figures_cap, f, ensure_ascii=False)

    while True:
        try:
            result = generate_poster_v3(
                vendor, model, text_prompt, figures_prompt, pdf, figures_cap, figures
            )

            poster = result["image_based_poster"]
            backup_poster = copy.deepcopy(poster)

            poster = replace_figures_in_poster(poster, figures)

            with open(output, "w") as f:
                json.dump(poster.model_dump(), f, ensure_ascii=False)

            poster_size = replace_figures_size_in_poster(backup_poster, figures)

            result = generate_html_v2(vendor, model, poster_size, figures)

            html = result["html_with_figures"]

            with open(output.replace(".json", ".html"), "w") as f:
                f.write(html)
            take_screenshot(output, html)

            return

        except Exception as e:
            if (
                "content management policy" in str(e)
                or "message larger than max" in str(e)
                or "exceeds the maximum length" in str(e)
                or "maximum context length" in str(e)
                or "Input is too long" in str(e)
                or "image exceeds 5 MB" in str(e)
                or "too many total text bytes" in str(e)
                or "Range of input length" in str(e)
                or "Invalid text" in str(e)
            ):
                raise
            print(f"处理文件 {pdf} 时出错: {e}")


if __name__ == "__main__":
    fire.Fire(generate_paper_poster)
