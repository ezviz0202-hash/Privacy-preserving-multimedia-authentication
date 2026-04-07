import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from feature_extractor import build_extractor, extract_features
from auth_protocol import gaussian_mechanism, compute_auth_hash
from adaptive_threshold import AdaptiveThreshold

os.makedirs("results", exist_ok=True)
os.makedirs("sample_images", exist_ok=True)


def make_face(path, size=224):
    img = Image.new("RGB", (size, size), (240, 230, 220))
    d = ImageDraw.Draw(img)
    d.ellipse([40, 30, 184, 200], fill=(210, 170, 130))
    d.ellipse([70, 80, 95, 100], fill=(60, 40, 20))
    d.ellipse([129, 80, 154, 100], fill=(60, 40, 20))
    d.ellipse([104, 120, 120, 135], fill=(190, 145, 110))
    d.arc([85, 145, 139, 175], 0, 180, fill=(160, 80, 80), width=3)
    img.save(path)
    return path


def make_scene(path, size=224):
    img = Image.new("RGB", (size, size), (135, 206, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([0, size // 2, size, size], fill=(80, 140, 60))
    d.ellipse([160, 20, 210, 70], fill=(255, 230, 50))
    d.rectangle([100, 130, 115, 180], fill=(100, 60, 20))
    d.ellipse([70, 80, 145, 140], fill=(34, 120, 34))
    img.save(path)
    return path


ATTACKS = [
    ("jpeg_mild", lambda p, o: Image.open(p).save(o, "JPEG", quality=85)),
    ("jpeg_heavy", lambda p, o: Image.open(p).save(o, "JPEG", quality=30)),
    ("blur", lambda p, o: Image.open(p).filter(ImageFilter.GaussianBlur(2)).save(o)),
    ("noise", lambda p, o: _add_noise(p, o, std=20)),
    ("crop", lambda p, o: _crop(p, o, ratio=0.85)),
]


def _add_noise(src, dst, std=20):
    arr = np.array(Image.open(src), dtype=np.int32)
    arr = np.clip(
        arr + np.random.normal(0, std, arr.shape).astype(np.int32),
        0,
        255
    ).astype(np.uint8)
    Image.fromarray(arr).save(dst)


def _crop(src, dst, ratio=0.85):
    img = Image.open(src)
    w, h = img.size
    nw, nh = int(w * ratio), int(h * ratio)
    l, t = (w - nw) // 2, (h - nh) // 2
    img.crop((l, t, l + nw, t + nh)).save(dst)


EPSILONS = [0.1, 0.5, 1.0, 2.0, 5.0, float("inf")]
IMAGES = {
    "face_01": "sample_images/face_01.jpg",
    "scene_01": "sample_images/scene_01.jpg",
}


def apply_dp(embedding, eps, delta=1e-5):
    if eps == float("inf"):
        return embedding.copy().astype(np.float32)
    return gaussian_mechanism(embedding, epsilon=eps, delta=delta)


def run_experiment():
    print("Generating synthetic test images...")
    make_face(IMAGES["face_01"])
    make_scene(IMAGES["scene_01"])

    print("Loading feature extractor...")
    extractor = build_extractor(embedding_dim=256, device="cpu")
    adaptive = AdaptiveThreshold()

    clean_embs = {}
    thresholds = {}
    for img_id, img_path in IMAGES.items():
        clean_embs[img_id] = extract_features(extractor, img_path, device="cpu")
        ct, thr = adaptive.get_threshold_for_image(img_path)
        thresholds[img_id] = thr
        print(f"  {img_id}: type={ct}, threshold={thr:.3f}")

    records = []

    for eps in EPSILONS:
        eps_label = "∞ (no DP)" if eps == float("inf") else str(eps)
        print(f"\n--- epsilon = {eps_label} ---")

        tar_total, tar_count = 0, 0
        far_total, far_count = 0, 0

        for img_id, img_path in IMAGES.items():
            emb = clean_embs[img_id]
            thr = thresholds[img_id]

            enrolled_noisy = apply_dp(emb, eps)
            enrolled_hash = compute_auth_hash(enrolled_noisy)

            # authentic verification
            probe_noisy = apply_dp(emb, eps)
            probe_hash = compute_auth_hash(probe_noisy)
            hash_ok = (probe_hash == enrolled_hash)
            sim = float(np.dot(probe_noisy, enrolled_noisy))
            cos_ok = sim >= thr
            accepted = hash_ok or cos_ok

            tar_total += int(accepted)
            tar_count += 1

            # attacked verification
            for atk_name, atk_fn in ATTACKS:
                atk_path = f"results/{img_id}_{atk_name}_tmp.jpg"
                try:
                    atk_fn(img_path, atk_path)
                    atk_emb = extract_features(extractor, atk_path, device="cpu")

                    atk_noisy = apply_dp(atk_emb, eps)
                    atk_hash = compute_auth_hash(atk_noisy)

                    hash_ok = (atk_hash == enrolled_hash)
                    sim = float(np.dot(atk_noisy, enrolled_noisy))
                    cos_ok = sim >= thr
                    accepted = hash_ok or cos_ok

                    far_total += int(accepted)
                    far_count += 1
                    os.remove(atk_path)
                except Exception as e:
                    print(f"    attack {atk_name} failed: {e}")

        tar = tar_total / max(tar_count, 1)
        far = far_total / max(far_count, 1)
        print(f"  TAR={tar:.3f}  FAR={far:.3f}")

        records.append({
            "epsilon": eps_label,
            "epsilon_val": eps,
            "TAR": tar,
            "FAR": far,
        })

    return records


def plot_results(records):
    df = pd.DataFrame(records)
    df.to_csv("results/tradeoff_data.csv", index=False)

    labels = df["epsilon"].tolist()
    tars = df["TAR"].tolist()
    fars = df["FAR"].tolist()
    x = range(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------------------------
    # Left: grouped bar chart
    # ---------------------------
    ax = axes[0]
    w = 0.35
    bars1 = ax.bar([i - w / 2 for i in x], tars, w, color="#2ecc71", label="TAR (authentic)")
    bars2 = ax.bar([i + w / 2 for i in x], fars, w, color="#e74c3c", label="FAR (attacked)")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"ε={l}" for l in labels], fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_title("Authentication Accuracy vs. Privacy Budget (ε)\nTAR = True Accept Rate  |  FAR = False Accept Rate")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Dynamically scale y-axis when values are small
    max_val = max(tars + fars) if (tars + fars) else 1.0
    if max_val <= 0.15:
        ax.set_ylim(0, max_val * 1.4 + 0.01)
    else:
        ax.set_ylim(0, 1.1)

    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(0.01, max_val * 0.03),
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(0.01, max_val * 0.03),
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    # ---------------------------
    # Right: tradeoff scatter
    # ---------------------------
    ax2 = axes[1]
    cmap = plt.cm.RdYlGn
    colors = [cmap(i / max(len(records) - 1, 1)) for i in range(len(records))]

    # Predefined label offsets to reduce overlap
    offsets = [
        (8, 6),
        (8, -8),
        (-25, 8),
        (-25, -10),
        (10, 14),
        (10, -16),
    ]

    # Add slight jitter only when multiple points overlap exactly
    seen_points = {}
    jitter_scale = 0.01

    for i, (tar, far, lbl) in enumerate(zip(tars, fars, labels)):
        key = (round(far, 4), round(tar, 4))
        count = seen_points.get(key, 0)
        seen_points[key] = count + 1

        jitter_x = 0.0
        jitter_y = 0.0
        if count > 0:
            jitter_x = (count % 3 - 1) * jitter_scale
            jitter_y = ((count // 3) % 3 - 1) * jitter_scale

        plot_far = far + jitter_x
        plot_tar = tar + jitter_y

        ax2.scatter(plot_far, plot_tar, color=colors[i], s=120, zorder=3, alpha=0.9)

        dx, dy = offsets[i % len(offsets)]
        ax2.annotate(
            f"ε={lbl}",
            (plot_far, plot_tar),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8
        )

    ax2.plot(fars, tars, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax2.set_xlabel("FAR (False Accept Rate) — lower is better →")
    ax2.set_ylabel("TAR (True Accept Rate) — higher is better ↑")
    ax2.set_title("Privacy-Accuracy Tradeoff Curve\n(ideal: top-left corner — high TAR, low FAR)")
    ax2.set_xlim(-0.05, 1.1)
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(alpha=0.3)
    ax2.axhline(1.0, color="#2ecc71", linewidth=0.5, linestyle=":")
    ax2.axvline(0.0, color="#e74c3c", linewidth=0.5, linestyle=":")

    fig.suptitle(
        "PrivAuth-MM: Differential Privacy vs. Authentication Performance",
        fontsize=12,
        fontweight="bold",
        y=1.01
    )
    fig.tight_layout()
    fig.savefig("results/privacy_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved: results/privacy_tradeoff.png")
    print("Saved: results/tradeoff_data.csv")


if __name__ == "__main__":
    np.random.seed(42)
    records = run_experiment()
    plot_results(records)

    print("\nDone. Summary:")
    for r in records:
        print(f"  epsilon={r['epsilon']:>12}  TAR={r['TAR']:.3f}  FAR={r['FAR']:.3f}")
