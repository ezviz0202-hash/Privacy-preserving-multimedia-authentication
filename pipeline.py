from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from feature_extractor import PrivacySuppressedExtractor, build_extractor, extract_features
from auth_protocol import AuthProtocol, AuthTemplate
from adaptive_threshold import AdaptiveThreshold


@dataclass
class PipelineResult:
    stage: str
    image_id: str
    image_path: str
    content_type: str
    threshold: float
    matched: Optional[bool]
    similarity: Optional[float]
    method: Optional[str]
    elapsed_ms: float
    details: str

    def __str__(self):
        lines = [
            f"{'─'*55}",
            f"  Stage       : {self.stage.upper()}",
            f"  Image ID    : {self.image_id}",
            f"  Content type: {self.content_type}",
            f"  Threshold   : {self.threshold:.4f}",
        ]
        if self.stage == "verify":
            status = "✅ AUTHENTIC" if self.matched else "❌ TAMPERED / REJECTED"
            lines += [
                f"  Similarity  : {self.similarity:.4f}",
                f"  Decision    : {status}",
                f"  Method      : {self.method}",
            ]
        lines += [
            f"  Time        : {self.elapsed_ms:.1f} ms",
            f"{'─'*55}",
        ]
        return "\n".join(lines)


class PrivAuthPipeline:

    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        embedding_dim=256,
        device="cpu",
        template_db="templates.json",
        secret_key=b"privauth-mm-secret",
    ):
        self.device = device
        self.epsilon = epsilon

        self.extractor = build_extractor(embedding_dim=embedding_dim, device=device)
        self.protocol  = AuthProtocol(
            epsilon=epsilon, delta=delta,
            secret_key=secret_key, template_db=template_db,
        )
        self.adaptive = AdaptiveThreshold()

    def enroll(self, image_id, image_path):
        t0 = time.perf_counter()
        ct, thr = self.adaptive.get_threshold_for_image(image_path, device=self.device)
        emb = extract_features(self.extractor, image_path, device=self.device)
        self.protocol.enroll(image_id, emb, content_type=ct)
        ms = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            stage="enroll", image_id=image_id, image_path=image_path,
            content_type=ct, threshold=thr,
            matched=None, similarity=None, method=None,
            elapsed_ms=ms, details=f"enrolled. eps={self.epsilon}",
        )

    def verify(self, image_id, probe_path, custom_threshold=None):
        t0 = time.perf_counter()
        ct, thr = self.adaptive.get_threshold_for_image(probe_path, device=self.device)
        if custom_threshold is not None:
            thr = custom_threshold
        probe_emb = extract_features(self.extractor, probe_path, device=self.device)
        res = self.protocol.verify(image_id, probe_emb, threshold=thr)
        ms = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            stage="verify", image_id=image_id, image_path=probe_path,
            content_type=ct, threshold=thr,
            matched=res["matched"], similarity=res["similarity"],
            method=res["method"], elapsed_ms=ms, details=res["details"],
        )

    @staticmethod
    def simulate_attack(image_path, attack="jpeg", output_path="attacked.jpg", **kw):
        from PIL import Image, ImageFilter, ImageEnhance
        img = Image.open(image_path).convert("RGB")

        if attack == "jpeg":
            img.save(output_path, "JPEG", quality=kw.get("quality", 70))
        elif attack == "crop":
            ratio = kw.get("ratio", 0.9)
            w, h = img.size
            nw, nh = int(w * ratio), int(h * ratio)
            l, t = (w - nw) // 2, (h - nh) // 2
            img.crop((l, t, l + nw, t + nh)).save(output_path)
        elif attack == "blur":
            img.filter(ImageFilter.GaussianBlur(radius=kw.get("sigma", 2))).save(output_path)
        elif attack == "noise":
            arr = np.array(img, dtype=np.int32)
            arr = np.clip(arr + np.random.normal(0, kw.get("std", 15), arr.shape).astype(np.int32), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(output_path)
        elif attack == "flip":
            img.transpose(Image.FLIP_LEFT_RIGHT).save(output_path)
        elif attack == "bright":
            ImageEnhance.Brightness(img).enhance(kw.get("factor", 1.4)).save(output_path)
        else:
            raise ValueError(f"unknown attack: {attack}")
        return output_path