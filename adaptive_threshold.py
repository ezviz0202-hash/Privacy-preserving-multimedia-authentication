from __future__ import annotations
import numpy as np
import torchvision.transforms as T
from PIL import Image
import torch
import torchvision.models as models
import colorsys

THRESHOLDS = {
    "face":     0.82,
    "document": 0.88,
    "scene":    0.70,
    "unknown":  0.75,
}

_DOCUMENT_HINT = {895, 659, 760, 673}
_SCENE_RANGE   = set(range(440, 700))


class ContentClassifier:
    
    _TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, device="cpu"):
        self.device = device
        self._model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        ).to(device)
        self._model.eval()

    def _top5(self, image_path):
        img = Image.open(image_path).convert("RGB")
        t = self._TRANSFORM(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(t)
        return torch.topk(logits, 5).indices.squeeze(0).tolist()

    @staticmethod
    def _colour_std(image_path):
        # low std = likely document (white background)
        img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        return float(np.mean([img[:, :, c].std() for c in range(3)]))

    @staticmethod
    def _skin_ratio(image_path):
        # rough face proxy: fraction of pixels in skin-tone HSV range
        img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = img.shape
        count = 0
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                r, g, b = img[y, x]
                hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
                if (0.02 < hue < 0.15) and (0.2 < sat < 0.8) and (val > 0.3):
                    count += 1
        total = (h // 4) * (w // 4)
        return count / max(total, 1)

    def classify(self, image_path):
        top5 = self._top5(image_path)
        std  = self._colour_std(image_path)
        skin = self._skin_ratio(image_path)

        if std < 40.0:
            return "document"
        if skin > 0.12:
            return "face"
        if any(i in _SCENE_RANGE for i in top5):
            return "scene"
        if any(i in _DOCUMENT_HINT for i in top5):
            return "document"
        return "unknown"


class AdaptiveThreshold:

    def __init__(self, custom=None):
        self._table = {**THRESHOLDS, **(custom or {})}
        self._clf = None

    def get_threshold(self, content_type):
        return self._table.get(content_type, self._table["unknown"])

    def get_threshold_for_image(self, image_path, device="cpu"):
        if self._clf is None:
            self._clf = ContentClassifier(device)
        ct = self._clf.classify(image_path)
        return ct, self.get_threshold(ct)

    def threshold_table(self):
        return dict(self._table)