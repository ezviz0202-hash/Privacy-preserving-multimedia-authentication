import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

def gaussian_mechanism(embedding, epsilon=1.0, delta=1e-5, sensitivity=1.0):
    # analytic gaussian mechanism: sigma = sens * sqrt(2*ln(1.25/delta)) / eps
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size=embedding.shape)
    noisy = embedding + noise
    n = np.linalg.norm(noisy)
    if n > 0:
        noisy = noisy / n
    return noisy.astype(np.float32)

def _quantise(embedding, n_bits=8):
    # map [-1,1] -> [0, 255]
    scale = (2 ** n_bits - 1) / 2
    q = np.clip((embedding + 1) * scale, 0, 2 ** n_bits - 1)
    return q.astype(np.uint8).tobytes()


def compute_auth_hash(embedding, secret_key=b"privauth-mm-secret"):
    payload = _quantise(embedding)
    return hmac.new(secret_key, payload, hashlib.sha256).hexdigest()

@dataclass
class AuthTemplate:
    image_id: str
    auth_hash: str
    clean_embedding: list   
    epsilon: float
    delta: float
    content_type: str
    enrolled_at: str

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return AuthTemplate(**d)

class AuthProtocol:

    def __init__(
        self,
        epsilon=1.0,
        delta=1e-5,
        secret_key=b"privauth-mm-secret",
        template_db="templates.json",
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.secret_key = secret_key
        self.template_db = template_db
        self._db = self._load_db()

    def _load_db(self):
        if os.path.exists(self.template_db):
            with open(self.template_db, "r") as f:
                return json.load(f)
        return {}

    def _save_db(self):
        with open(self.template_db, "w") as f:
            json.dump(self._db, f, indent=2)

    def enroll(self, image_id, embedding, content_type="unknown"):
        # dp noise only goes into the hash; clean embedding kept for soft verify
        noisy_emb = gaussian_mechanism(embedding, self.epsilon, self.delta)
        auth_hash = compute_auth_hash(noisy_emb, self.secret_key)

        template = AuthTemplate(
            image_id=image_id,
            auth_hash=auth_hash,
            clean_embedding=embedding.tolist(),
            epsilon=self.epsilon,
            delta=self.delta,
            content_type=content_type,
            enrolled_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self._db[image_id] = template.to_dict()
        self._save_db()
        return template

    def verify(self, image_id, probe_embedding, threshold=None):
        if image_id not in self._db:
            return {
                "matched": False,
                "similarity": 0.0,
                "method": "N/A",
                "details": f"ID '{image_id}' not found.",
            }

        template = AuthTemplate.from_dict(self._db[image_id])
        enrolled_emb = np.array(template.clean_embedding, dtype=np.float32)

        
        probe_noisy = gaussian_mechanism(probe_embedding, self.epsilon, self.delta)
        probe_hash = compute_auth_hash(probe_noisy, self.secret_key)
        hash_match = hmac.compare_digest(probe_hash, template.auth_hash)

        sim = float(np.dot(probe_embedding, enrolled_emb))

        if threshold is None:
            from adaptive_threshold import AdaptiveThreshold
            threshold = AdaptiveThreshold().get_threshold(template.content_type)

        soft_match = sim >= threshold
        matched = hash_match or soft_match

        method_parts = []
        if hash_match:
            method_parts.append("hash")
        if soft_match:
            method_parts.append(f"cosine({sim:.3f}\u2265{threshold:.3f})")

        return {
            "matched": matched,
            "similarity": round(sim, 4),
            "method": " + ".join(method_parts) if method_parts else "none",
            "details": (
                f"Hash match: {hash_match} | "
                f"Cosine: {sim:.4f} | "
                f"Threshold: {threshold:.4f} | "
                f"Type: {template.content_type}"
            ),
        }