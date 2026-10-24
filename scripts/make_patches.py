"""
LoveDA 1024px 원본을 256px 패치로 자르고, '원본 사진 단위'로 train/val/test를 나눈다.

[팀 기준 분할: --team_repo 사용, 권장]
  팀원 GAN 저장소(loveda-semantic-editing)의 목록을 그대로 따른다.
- train: data/train_sources.json 의 공식 Train 2,490장 (팀원 GAN 학습과 동일)
- val  : 공식 Train 중 과거 평가에 쓴 32장 + 공식 Val 중 최종 테스트 예약이 아닌 562장 (개발·설정 선택용)
- test : data/final_test_sources.json 의 공식 Val 1,107장 (최종 평가·조작 세트·기말 탐지용)

[기존 분할: --team_repo 없이]
- 공식 Train → train / val (도메인별 9:1), 공식 Val → test
- no-data(0) 비율이 높은 패치는 제외
- 저장은 전부 PNG (무손실). 진짜/가짜 저장 형식을 통일하기 위함

출력 구조:
  OUT/{train,val,test}/images/{patch_id}.png   RGB
  OUT/{train,val,test}/masks/{patch_id}.png    클래스 번호 0~7 (그대로 보면 검정)
  OUT/index.csv   patch_id, split, src_split, domain, src_id, y, x, nodata, 클래스별 비율

사용법:
  python make_patches.py --root data/LoveDA --out data/LoveDA_patches
"""
import argparse
import csv
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

CLASS_KEYS = ["background", "building", "road", "water", "barren", "forest", "agriculture"]


def sort_key(p):
    return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)


def build_jobs_team(root, repo):
    """팀원 저장소의 원본 목록으로 split을 정한다."""
    import json
    tr = json.load(open(repo / "data" / "train_sources.json", encoding="utf-8"))
    train_set = {r["source_member"] for r in tr["sources"]}
    test_set = set(json.load(open(repo / "data" / "final_test_sources.json", encoding="utf-8")))
    jobs = []
    for src_split in ["Train", "Val"]:
        for domain in ["Urban", "Rural"]:
            for p in sorted((root / src_split / domain / "images_png").glob("*.png"), key=sort_key):
                member = f"{src_split}/{domain}/images_png/{p.name}"
                if member in train_set:
                    split = "train"
                elif member in test_set:
                    split = "test"
                else:
                    split = "val"
                jobs.append((split, src_split, domain, p, p.parent.parent / "masks_png" / p.name))
    found = {j[0] for j in jobs}
    n_train = sum(j[0] == "train" for j in jobs)
    n_test = sum(j[0] == "test" for j in jobs)
    if n_train != len(train_set) or n_test != len(test_set):
        raise SystemExit(f"[error] 팀 목록과 원본 수가 달라요: train {n_train}/{len(train_set)}, test {n_test}/{len(test_set)}")
    return jobs


def build_jobs(root, val_ratio, seed):
    """원본 사진마다 split을 정한다. 패치가 아니라 사진 단위라서 같은 사진의 패치가 섞이지 않는다."""
    rng = random.Random(seed)
    jobs = []
    for domain in ["Urban", "Rural"]:
        imgs = sorted((root / "Train" / domain / "images_png").glob("*.png"), key=sort_key)
        order = list(range(len(imgs)))
        rng.shuffle(order)
        val_idx = set(order[:round(len(imgs) * val_ratio)])
        for i, p in enumerate(imgs):
            split = "val" if i in val_idx else "train"
            jobs.append((split, "Train", domain, p, p.parent.parent / "masks_png" / p.name))
        for p in sorted((root / "Val" / domain / "images_png").glob("*.png"), key=sort_key):
            jobs.append(("test", "Val", domain, p, p.parent.parent / "masks_png" / p.name))
    return jobs


def process(args):
    (split, src_split, domain, ip, mp), out, size, max_nodata = args
    img = np.array(Image.open(ip).convert("RGB"))
    m = np.array(Image.open(mp))
    rows, dropped = [], 0
    H, W = m.shape[:2]
    for r, y in enumerate(range(0, H - size + 1, size)):
        for c, x in enumerate(range(0, W - size + 1, size)):
            pm = m[y:y + size, x:x + size]
            nod = float((pm == 0).mean())
            if nod > max_nodata:
                dropped += 1
                continue
            pid = f"{src_split}-{domain}-{ip.stem}-r{r}c{c}"
            Image.fromarray(img[y:y + size, x:x + size]).save(out / split / "images" / f"{pid}.png")
            Image.fromarray(pm.astype(np.uint8)).save(out / split / "masks" / f"{pid}.png")
            ratios = [round(float((pm == k).mean()), 4) for k in range(1, 8)]
            rows.append([pid, split, src_split, domain, ip.stem, y, x, round(nod, 4)] + ratios)
    return rows, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="LoveDA 원본 폴더 (Train, Val이 들어 있는 곳)")
    ap.add_argument("--out", type=Path, required=True, help="패치를 저장할 폴더")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max_nodata", type=float, default=0.10, help="no-data 비율이 이보다 크면 패치 제외")
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--team_repo", type=Path, default=None,
                    help="팀원 GAN 저장소 경로. 주면 그 저장소의 train/최종테스트 목록으로 분할")
    a = ap.parse_args()

    jobs = build_jobs_team(a.root, a.team_repo) if a.team_repo else build_jobs(a.root, a.val_ratio, a.seed)
    if not jobs:
        raise SystemExit(f"[error] {a.root}/Train/*/images_png 를 못 찾았어요.")
    for s in ["train", "val", "test"]:
        for sub in ["images", "masks"]:
            (a.out / s / sub).mkdir(parents=True, exist_ok=True)

    src_count = Counter(j[0] for j in jobs)
    print("원본 사진 수:", dict(src_count))

    rows, dropped = [], 0
    tasks = [(j, a.out, a.size, a.max_nodata) for j in jobs]
    with Pool(a.workers) as pool:
        for r, d in tqdm(pool.imap_unordered(process, tasks, chunksize=4), total=len(tasks)):
            rows.extend(r)
            dropped += d
    rows.sort(key=lambda r: r[0])

    with open(a.out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch_id", "split", "src_split", "domain", "src_id", "y", "x", "nodata"] + CLASS_KEYS)
        w.writerows(rows)

    print(f"\n저장한 패치 {len(rows)}개 / no-data 많아서 제외 {dropped}개")
    water_col = 8 + CLASS_KEYS.index("water")
    bld_col = 8 + CLASS_KEYS.index("building")
    for s in ["train", "val", "test"]:
        sr = [r for r in rows if r[1] == s]
        nw = sum(r[water_col] >= 0.05 for r in sr)
        nb = sum(r[bld_col] >= 0.05 for r in sr)
        print(f"  {s:5s}: {len(sr):6d}개  (물 5% 이상 {nw}개, 건물 5% 이상 {nb}개)")
    print("index:", a.out / "index.csv")


if __name__ == "__main__":
    main()
