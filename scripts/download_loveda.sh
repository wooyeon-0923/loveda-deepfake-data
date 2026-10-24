#!/usr/bin/env bash
# LoveDA 다운로드 + 무결성 확인 + 압축 해제
# 사용법: bash download_loveda.sh [저장경로]   (기본: ./data/LoveDA)
# Test.zip(라벨 없음, 3.1GB)도 받으려면: WITH_TEST=1 bash download_loveda.sh
set -euo pipefail

DATA_DIR="${1:-./data/LoveDA}"
BASE_URL="https://zenodo.org/records/5706578/files"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# 파일명과 Zenodo 공식 md5
declare -A MD5=(
  [Train.zip]=de2b196043ed9b4af1690b3f9a7d558f
  [Val.zip]=84cae2577468ff0b5386758bb386d31d
  [Test.zip]=a489be0090465e01fb067795d24e6b47
)
FILES=(Train.zip Val.zip)
if [[ "${WITH_TEST:-0}" == "1" ]]; then FILES+=(Test.zip); fi

# 디스크 여유 확인 (zip + 압축 해제분 대략 15GB 필요)
AVAIL_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
echo "[info] 저장 경로: $(pwd) / 여유 공간: ${AVAIL_GB}GB"
if (( AVAIL_GB < 15 )); then
  echo "[warn] 여유 공간이 15GB보다 적습니다. 다른 경로를 인자로 주세요."
fi

for f in "${FILES[@]}"; do
  if [[ -f "$f" ]] && echo "${MD5[$f]}  $f" | md5sum -c --status; then
    echo "[skip] $f 이미 받았고 md5 일치"
  else
    echo "[down] $f 다운로드 (중단돼도 다시 실행하면 이어받기)"
    wget -c --tries=20 --retry-connrefused --waitretry=5 \
         -O "$f" "${BASE_URL}/${f}?download=1"
    echo "${MD5[$f]}  $f" | md5sum -c - || { echo "[error] $f md5 불일치. 파일 지우고 다시 실행하세요."; exit 1; }
  fi

  dir="${f%.zip}"
  if [[ -d "$dir" ]]; then
    echo "[skip] $dir 이미 압축 해제됨"
  else
    echo "[unzip] $f"
    unzip -q "$f"
  fi
done

echo "[done] 완료. 다음: python check_loveda.py $(pwd)"
