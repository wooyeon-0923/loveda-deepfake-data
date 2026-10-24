"""
조작 규칙과 학습용 구멍이 의도대로 만들어지는지 눈으로 확인하는 스크립트.
발표 자료용 그림으로도 그대로 쓸 수 있다.

출력 (OUT 폴더):
  rules_water_expand.png     원본 | 원본 마스크 | 편집 마스크 | 구멍(=조작 위치 정답) | 모델 입력
  rules_building_remove.png  (같은 구성)
  train_holes.png            학습용 구멍 예시 (종류별)
  그리고 규칙 성공률(조건을 만족한 패치 비율)을 출력

사용법:
  python demo_rules.py --patches data/LoveDA_patches --out outputs
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loveda_lib import SCENARIO_CANDIDATE, SCENARIOS, colorize, load_patch, sample_train_hole


def overlay_hole(img, hole, color=(255, 0, 255), alpha=0.55):
    out = img.astype(np.float32).copy()
    out[hole] = (1 - alpha) * out[hole] + alpha * np.array(color)
    return out.astype(np.uint8)


def masked_view(img, hole):
    out = img.copy()
    out[hole] = 0
    return out


def read_index(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=4, help="그림에 넣을 예시 수")
    ap.add_argument("--n_stat", type=int, default=300, help="성공률을 셀 후보 패치 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", nargs="*", default=None, help="그릴 시나리오만 (기본 전체)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = [r for r in read_index(a.patches / "index.csv") if r["split"] == a.split]

    cand = {name: [r for r in rows if fn(r)] for name, fn in SCENARIO_CANDIDATE.items()}

    for name, fn in SCENARIOS.items():
        if a.scenarios and name not in a.scenarios:
            continue
        pool = cand[name]
        if not pool:
            print(f"[{name}] 후보 패치가 없어요.")
            continue
        picks = rng.permutation(len(pool))
        # 성공률
        ok, ratios = 0, []
        for j in picks[:a.n_stat]:
            _, m = load_patch(a.patches, a.split, pool[j]["patch_id"])
            res = fn(m, rng=rng)
            if res is not None:
                ok += 1
                ratios.append(res[2]["hole_ratio"])
        n_try = min(a.n_stat, len(pool))
        msg = f"[{name}] 후보 {len(pool)}개 중 {n_try}개 시도 → 성공 {ok}개 ({ok / n_try * 100:.1f}%)"
        if ratios:
            msg += f", 조작 면적 평균 {np.mean(ratios) * 100:.1f}% (패치 대비)"
        print(msg)

        # 그림
        examples = []
        for j in picks:
            pid = pool[j]["patch_id"]
            img, m = load_patch(a.patches, a.split, pid)
            res = fn(m, rng=rng)
            if res is not None:
                examples.append((pid, img, m, *res))
            if len(examples) >= a.n:
                break
        cols = ["Image", "Original mask", "Edited mask", "Hole = GT location", "Model input (masked)"]
        fig, axes = plt.subplots(len(examples), 5, figsize=(15, 3.1 * len(examples)), squeeze=False)
        for r, (pid, img, m, edited, hole, info) in enumerate(examples):
            panels = [img, colorize(m), colorize(edited), overlay_hole(img, hole), masked_view(img, hole)]
            for c, p in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(p)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(cols[c], fontsize=11)
            axes[r, 0].set_ylabel(f"{pid}\nhole {info['hole_ratio'] * 100:.1f}%", fontsize=8)
        fig.suptitle(f"Manipulation rule: {name}", fontsize=13)
        fig.tight_layout()
        fig.savefig(a.out / f"rules_{name}.png", dpi=110)
        plt.close(fig)
        print("  저장:", a.out / f"rules_{name}.png")

    # 학습용 구멍 예시
    pool = [r for r in read_index(a.patches / "index.csv") if r["split"] == "train"] or rows
    kinds, shown = Counter(), []
    for j in rng.permutation(len(pool))[:400]:
        _, m = load_patch(a.patches, pool[j]["split"], pool[j]["patch_id"])
        hole, kind = sample_train_hole(m, rng)
        kinds[kind] += 1
        if len(shown) < 8:
            img, _ = load_patch(a.patches, pool[j]["split"], pool[j]["patch_id"])
            shown.append((img, m, hole, kind))
    print("[학습용 구멍] 종류 분포 (400개 샘플):", dict(kinds))
    fig, axes = plt.subplots(2, len(shown), figsize=(2.6 * len(shown), 5.4), squeeze=False)
    for c, (img, m, hole, kind) in enumerate(shown):
        axes[0, c].imshow(overlay_hole(img, hole)); axes[0, c].set_title(kind, fontsize=10)
        axes[1, c].imshow(colorize(m))
        for r in range(2):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("image + hole"); axes[1, 0].set_ylabel("mask (unchanged)")
    fig.suptitle("Training holes (mask is NOT edited during training)", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.out / "train_holes.png", dpi=110)
    plt.close(fig)
    print("  저장:", a.out / "train_holes.png")


if __name__ == "__main__":
    main()
