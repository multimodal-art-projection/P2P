import modal
from io import BytesIO
from pathlib import Path
from fastapi import File, UploadFile, Form
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])  # install system libraries for graphics handling
    .pip_install(
        "ultralytics>=8.2.85",
        "doclayout-yolo==0.0.2",
        "huggingface-hub",
        "fastapi",
    )
)
volume = modal.Volume.from_name("yolo-layout-detection", create_if_missing=True)
volume_path = Path("/root") / "data"
model_path = volume_path / "path2doclayout_yolo_ft.pt"
app = modal.App(
    "yolo-layout-detection-temp",
    image=image,
    volumes={volume_path: volume},
)
@app.function()
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="opendatalab/pdf-extract-kit-1.0",
        local_dir=volume_path,
        allow_patterns='path2*',
        max_workers=20,
    )
@app.cls(gpu="a10g")
class LayoutDetection:
    @modal.enter()
    def load_model(self):
        from doclayout_yolo import YOLOv10
        self.model = YOLOv10(model_path)
    @modal.web_endpoint(method="POST", docs=True)
    async def predict(self, img: UploadFile = File(...), task: str = Form(...)):
        from PIL import Image
        img_bytes = await img.read()
        img = Image.open(BytesIO(img_bytes))
        results = self.model.predict(img)
        # parse results
        figs = []
        for result in results:
            boxes = result.__dict__['boxes'].xyxy.cpu().tolist()
            classes = result.__dict__['boxes'].cls.cpu().tolist()
            scores = result.__dict__['boxes'].conf.cpu().tolist()
            targets, captions = [], []
            for box, cls, score in zip(boxes, classes, scores):
                if task == "figure":
                    if cls == 3:
                        targets.append({"box": box, "score": score})
                elif task == "table":
                    if cls == 5:
                        targets.append({"box": box, "score": score})
                elif task == "figurecaption":
                    if cls == 3:
                        targets.append({"box": box, "score": score})
                    elif cls == 4:
                        captions.append({"box": box, "score": score})
                elif task == "tablecaption":
                    if cls == 5:
                        targets.append({"box": box, "score": score})
                    elif cls == 6 or cls == 7:
                        captions.append({"box": box, "score": score})
            if not captions:
                figs = targets
            else:
                matches = []
                for target in targets:
                    min_distance = float('inf')
                    for caption in captions:
                        target_box, caption_box = target["box"], caption["box"]
                        distance = abs(target_box[0] - caption_box[0]) + abs(target_box[3] - caption_box[1])
                        if distance < min_distance:
                            min_distance = distance
                            correct_match = (target, caption)
                    matches.append(correct_match)
                for target, caption in matches:
                    target_box, caption_box = target["box"], caption["box"]
                    union_box = [
                        min(target_box[0], caption_box[0]),
                        min(target_box[1], caption_box[1]),
                        max(target_box[2], caption_box[2]),
                        max(target_box[3], caption_box[3]),
                    ]
                    figs.append({"box": union_box, "score": 1.0})
        return figs