import base64
import requests
import os
from pathlib import Path

from io import BytesIO
from PIL import Image
from retry import retry

from .loader import ImagePDFLoader


@retry(tries=3)
def _extract_figures(
    url: str, img: Image.Image, task: str = "figure"
) -> list[tuple[Image.Image, float]]:
    figures = []

    with BytesIO() as buffer:
        img.save(buffer, format="PNG")

        files = [("img", ("image.png", buffer.getvalue(), "image/png"))]
        payload = {"task": task}
        rsp = requests.request("POST", url, data=payload, files=files)
        rsp.raise_for_status()

    for data in rsp.json():
        figures.append((img.crop(data["box"]), data["score"]))

    return figures


def extract_figures(
    url: str, pdf: str, task: str = "figure"
) -> list[tuple[str, float]]:
    loader = ImagePDFLoader(pdf)
    images = loader.load()

    figures = []
    for image in images:
        figures.extend(_extract_figures(url, image, task))

    base64_figures = []
    for figure, score in figures:
        with BytesIO() as buffer:
            figure.save(buffer, format="PNG")
            base64_figures.append(
                (base64.b64encode(buffer.getvalue()).decode("utf-8"), score)
            )

    return base64_figures


if __name__ == "__main__":
    url = ""
    pdf = "1.pdf"

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    base64_figures = extract_figures(url, pdf, task="figurecaption")

    print(f"提取到 {len(base64_figures)} 张图像")

    for i, (b64_str, score) in enumerate(base64_figures):
        img_data = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_data))

        output_path = output_dir / f"figure_{i + 1}.png"
        img.save(output_path)
        print(f"图像已保存到: {output_path}")

    print(f"所有图像已保存到 {output_dir} 目录")
