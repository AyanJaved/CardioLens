#!/usr/bin/env python3
# Patient-level train/val/test split (no patient appears in more than one split).
# Label: Cardiomegaly present vs. everything else (other diseases + No Finding).

import argparse
import os

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def print_stats(name, df):
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    n_patients = df["Patient ID"].nunique()
    pct = 100 * n_pos / len(df) if len(df) else 0
    print(f"{name}: images={len(df)} patients={n_patients} pos={n_pos} neg={n_neg} pos%={pct:.1f}")


def group_split(df, test_size, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_a, idx_b = next(splitter.split(df, groups=df["Patient ID"]))
    return df.iloc[idx_a].copy(), df.iloc[idx_b].copy()


def balance(df, seed):
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n = min(len(pos), len(neg))
    pos = pos.sample(n=n, random_state=seed)
    neg = neg.sample(n=n, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument("--out-dir", default="processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = os.path.join(args.raw_dir, "Data_Entry_2017.csv")
    images_dir = os.path.join(args.raw_dir, "images")

    df = pd.read_csv(csv_path)
    available = set(os.listdir(images_dir))
    df = df[df["Image Index"].isin(available)].copy()
    df["label"] = df["Finding Labels"].str.contains("Cardiomegaly").astype(int)
    df = df[["Image Index", "Patient ID", "label"]].rename(columns={"Image Index": "filename"})

    train, temp = group_split(df, test_size=0.2, seed=args.seed)
    val, test = group_split(temp, test_size=0.5, seed=args.seed)

    train_p = set(train["Patient ID"])
    val_p = set(val["Patient ID"])
    test_p = set(test["Patient ID"])
    assert train_p.isdisjoint(val_p)
    assert train_p.isdisjoint(test_p)
    assert val_p.isdisjoint(test_p)

    print("-- raw patient-level split --")
    print_stats("train", train)
    print_stats("val", val)
    print_stats("test", test)

    train = balance(train, args.seed)
    val = balance(val, args.seed)
    test = balance(test, args.seed)

    print("-- after per-split balancing --")
    print_stats("train", train)
    print_stats("val", val)
    print_stats("test", test)

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main()