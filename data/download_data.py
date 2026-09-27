#!/usr/bin/env python3
# Attribution: https://nihcc.app.box.com/v/ChestXray-NIHCC (Wang et al., CVPR 2017)

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

ALL_LINKS = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",  # 001
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",  # 002
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",  # 003
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",  # 004
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",  # 005
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",  # 006
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",  # 007
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",  # 008
    "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",  # 009
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",  # 010
    "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",  # 011
    "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",  # 012
]

LABELS_CSV_URL = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/raw/main/data/Data_Entry_2017_v2020.csv"


class DownloadProgressBar(tqdm):
    pass


def download_file(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        return
    tmp_path = dest_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with DownloadProgressBar(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(dest_path)
    ) as t, urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        t.total = total
        block = 1024 * 64
        while chunk := resp.read(block):
            f.write(chunk)
            t.update(len(chunk))
    os.rename(tmp_path, dest_path)


def main():
    parser = argparse.ArgumentParser(description="Download first N NIH ChestX-ray14 image archives.")
    parser.add_argument("--out", default="data/raw", help="Output directory (default: data/raw)")
    parser.add_argument("--n", type=int, default=3, help="Number of image zip archives to download (default: 3)")
    parser.add_argument("--skip-labels", action="store_true", help="Skip downloading Data_Entry_2017.csv")
    args = parser.parse_args()

    if not (1 <= args.n <= len(ALL_LINKS)):
        print(f"--n must be between 1 and {len(ALL_LINKS)}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    for idx in range(1,len(ALL_LINKS)):
        filename = f"images_{idx + 1:03d}.tar.gz"
        download_file(ALL_LINKS[idx], os.path.join(args.out, filename))

    if not args.skip_labels:
        download_file(LABELS_CSV_URL, os.path.join(args.out, "Data_Entry_2017.csv"))


if __name__ == "__main__":
    main()