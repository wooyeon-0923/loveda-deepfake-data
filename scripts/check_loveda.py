"""
LoveDA 다운로드 결과 점검 스크립트
- 폴더 구조, 이미지/마스크 짝, 해상도, 마스크 값 체계, 클래스 비율을 확인
- 사용법: python check_loveda.py ./data/LoveDA [--sample 300]
- 필요 패키지: pip install numpy pillow tqdm
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# LoveDA 클래스 (공식 README 기준으로 알려진 값: 0=no-data, 1~7=클래스)
# 아래 '마스크 값' 출력으로 실제 값을 꼭 확인할 것
CLASS_NAMES = {
    0: "no-data", 1: "background", 2: "building", 3: "road",
    4: "water", 5: "barren", 6: "forest", 7: "agriculture",
}


def find_pairs(root: Path):
    """split/domain 별로 (image, mask) 짝을 찾는다."""
    results = {}
    for img_dir in sorted(root.glob("*/*/images_png")):
        split, domain = img_dir.parent.parent.name, img_dir.parent.name
        mask_dir = img_dir.parent / "masks_png"
        imgs = sorted(img_dir.glob("*.png"))
        pairs, missing = [], 0
        for p in imgs:
            m = mask_dir / p.name
            if m.exists():
                pairs.append((p, m))
            else:
                missing += 1
        results[(split, domain)] = {
            "n_img": len(imgs), "pairs": pairs, "missing_mask": missing,
            "has_mask_dir": mask_dir.exists(),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--sample", type=int, default=300,
                    help="통계를 낼 마스크 샘플 수 (split/domain 당). 0이면 전체")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    groups = find_pairs(args.root)
    if not groups:
        print("[error] */*/images_png 폴더를 못 찾았어요. 압축 해제 경로를 확인하세요.")
        print("        현재 하위 폴더:", [p.name for p in args.root.iterdir()])
        return

    print("=" * 60)
    print("1) 폴더 구조와 이미지-마스크 짝")
    print("=" * 60)
    for (split, domain), g in groups.items():
        print(f"{split:6s}/{domain:6s}  이미지 {g['n_img']:5d}장  "
              f"마스크 폴더 {'O' if g['has_mask_dir'] else 'X (라벨 없음)'}  "
              f"짝 없는 이미지 {g['missing_mask']}")

    print("\n" + "=" * 60)
    print("2) 해상도 / 마스크 값 / 클래스 비율 (샘플 기준)")
    print("=" * 60)
    total_counter = Counter()
    for (split, domain), g in groups.items():
        pairs = g["pairs"]
        if not pairs:
            continue
        sample = pairs if args.sample == 0 else random.sample(pairs, min(args.sample, len(pairs)))

        sizes, counter = Counter(), Counter()
        water_patches = 0  # 256 패치 기준 수역 5% 이상인 패치 수 (수역 조작 후보)
        n_patches = 0
        for img_p, mask_p in tqdm(sample, desc=f"{split}/{domain}", leave=False):
            with Image.open(img_p) as im:
                sizes[(im.size, im.mode)] += 1
            m = np.array(Image.open(mask_p))
            vals, cnts = np.unique(m, return_counts=True)
            counter.update(dict(zip(vals.tolist(), cnts.tolist())))

            h, w = m.shape[:2]
            for y in range(0, h - 255, 256):
                for x in range(0, w - 255, 256):
                    patch = m[y:y + 256, x:x + 256]
                    n_patches += 1
                    if (patch == 4).mean() >= 0.05:
                        water_patches += 1

        total_counter.update(counter)
        total = sum(counter.values())
        print(f"\n[{split}/{domain}] 샘플 {len(sample)}장")
        print("  해상도/모드:", dict(sizes))
        print("  마스크 값 분포:")
        for v in sorted(counter):
            name = CLASS_NAMES.get(v, "??? (예상 밖 값)")
            print(f"    {v:3d} {name:12s} {counter[v] / total * 100:6.2f}%")
        if n_patches:
            print(f"  256px 패치 중 수역 5% 이상: {water_patches}/{n_patches} "
                  f"({water_patches / n_patches * 100:.1f}%)")

    unexpected = [v for v in total_counter if v not in CLASS_NAMES]
    print("\n" + "=" * 60)
    if unexpected:
        print(f"[warn] 예상 밖 마스크 값 발견: {unexpected} → CLASS_NAMES 수정 필요")
    else:
        print("[ok] 마스크 값이 0~7 범위 안에 있어요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
