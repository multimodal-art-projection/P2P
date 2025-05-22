import fitz

from PIL import Image
from langchain_community.document_loaders import PyMuPDFLoader


class ImagePDFLoader(PyMuPDFLoader):
    def load_pdf_page(self, page: fitz.Page, dpi: int) -> Image.Image:
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if pix.width > 3000 or pix.height > 3000:
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        return image

    def load(self) -> list[Image.Image]:
        images = []

        doc = fitz.open(self.file_path)
        for i in range(len(doc)):
            page = doc[i]
            image = self.load_pdf_page(page, dpi=250)
            images.append(image)

        return images
