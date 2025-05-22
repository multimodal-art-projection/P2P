import base64
import io
from PIL import Image


def compress_image(base64_str, quality=85, max_size=(1024, 1024)):
    """
    压缩base64编码的图片

    参数:
        base64_str: base64编码的图片字符串
        quality: 压缩质量 (1-100)
        max_size: 最大尺寸 (宽, 高)

    返回:
        压缩后的base64编码字符串
    """
    try:

        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))


        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.LANCZOS)


        output = io.BytesIO()
        img.save(output, format="PNG", optimize=True, quality=quality)


        compressed_base64 = base64.b64encode(output.getvalue()).decode("utf-8")
        return compressed_base64
    except Exception as e:
        print(f"图片压缩失败: {e}")
        return base64_str  # 如果压缩失败，返回原始图片
