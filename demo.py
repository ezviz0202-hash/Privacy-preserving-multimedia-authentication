import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw

from pipeline import PrivAuthPipeline

EPSILON   = 1.0
OUT_DIR   = "results"
IMG_DIR   = "sample_images"
DB_PATH   = "results/templates.json"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


def make_face_image(path, size=224):
    img = Image.new("RGB", (size, size), color=(240, 230, 220))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 30, 184, 200], fill=(210, 170, 130))
    draw.ellipse([70, 80, 95, 100],  fill=(60, 40, 20))
    draw.ellipse([129, 80, 154, 100], fill=(60, 40, 20))
    draw.ellipse([104, 120, 120, 135], fill=(190, 145, 110))
    draw.arc([85, 145, 139, 175], start=0, end=180, fill=(160, 80, 80), width=3)
    img.save(path)
    return path


def make_scene_image(path, size=224):
    img = Image.new("RGB", (size, size), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, size // 2, size, size], fill=(80, 140, 60))
    draw.ellipse([160, 20, 210, 70], fill=(255, 230, 50))
    draw.rectangle([100, 130, 115, 180], fill=(100, 60, 20))
    draw.ellipse([70, 80, 145, 140], fill=(34, 120, 34))
    draw.ellipse([10, 30, 80, 60],  fill=(255, 255, 255))
    draw.ellipse([30, 20, 100, 55], fill=(255, 255, 255))
    img.save(path)
    return path


def generate_sample_images():
    paths = {}
    paths["face_01"]  = make_face_image(os.path.join(IMG_DIR, "face_01.jpg"))
    paths["scene_01"] = make_scene_image(os.path.join(IMG_DIR, "scene_01.jpg"))
    print(f"  Generated: face_01.jpg, scene_01.jpg  ->  {IMG_DIR}/")
    return paths


ATTACKS = [
    ("jpeg_mild",  "jpeg",   {"quality": 85}),
    ("jpeg_heavy", "jpeg",   {"quality": 30}),
    ("crop_mild",  "crop",   {"ratio": 0.95}),
    ("crop_heavy", "crop",   {"ratio": 0.70}),
    ("blur",       "blur",   {"sigma": 2}),
    ("noise",      "noise",  {"std": 20}),
    ("brightness", "bright", {"factor": 1.5}),
]


def run_demo():
    print("\n" + "="*55)
    print("  PrivAuth-MM — Privacy-Preserving Multimedia Auth Demo")
    print("="*55 + "\n")

    print("[1/4] Generating synthetic test images ...")
    image_paths = generate_sample_images()
    print()

    print("[2/4] Initialising pipeline (epsilon={}) ...".format(EPSILON))
    pipe = PrivAuthPipeline(epsilon=EPSILON, template_db=DB_PATH)
    print("  Done.\n")

    print("[3/4] Enrolling images ...")
    for img_id, img_path in image_paths.items():
        result = pipe.enroll(img_id, img_path)
        print(result)

    print("[4/4] Verification experiments ...\n")
    all_labels, all_sims, all_matched, all_thresholds = [], [], [], []

    for img_id, img_path in image_paths.items():
        print(f"-- Image: {img_id} --")
        r = pipe.verify(img_id, img_path)
        print(r)
        all_labels.append(f"{img_id}\n(authentic)")
        all_sims.append(r.similarity or 0.0)
        all_matched.append(r.matched)
        all_thresholds.append(r.threshold)

        for atk_name, atk_type, atk_kwargs in ATTACKS:
            atk_path = os.path.join(OUT_DIR, f"{img_id}_{atk_name}.jpg")
            try:
                pipe.simulate_attack(img_path, attack=atk_type,
                                     output_path=atk_path, **atk_kwargs)
                r = pipe.verify(img_id, atk_path)
                print(r)
                all_labels.append(f"{img_id}\n({atk_name})")
                all_sims.append(r.similarity or 0.0)
                all_matched.append(r.matched)
                all_thresholds.append(r.threshold)
            except Exception as e:
                print(f"  Attack '{atk_name}' failed: {e}")

    _plot_results(all_labels, all_sims, all_matched, all_thresholds)
    print(f"\nChart saved to {OUT_DIR}/similarity_chart.png")
    print("="*55 + "\n")


def _plot_results(labels, similarities, matched, thresholds):
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    colours = ["#2ecc71" if m else "#e74c3c" for m in matched]
    ax.bar(range(len(labels)), similarities, color=colours,
           edgecolor="white", linewidth=0.8, zorder=3)
    for i, thr in enumerate(thresholds):
        ax.plot([i - 0.4, i + 0.4], [thr, thr],
                color="#2c3e50", linewidth=1.5, linestyle="--", zorder=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Cosine Similarity (noisy embeddings)")
    ax.set_title(
        "PrivAuth-MM: Verification Similarity Under Various Attacks\n"
        "(dashed line = adaptive threshold per content type)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    pass_patch = mpatches.Patch(color="#2ecc71", label="AUTHENTIC (matched)")
    fail_patch = mpatches.Patch(color="#e74c3c", label="REJECTED (tampered)")
    ax.legend(handles=[pass_patch, fail_patch], loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/similarity_chart.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_demo()