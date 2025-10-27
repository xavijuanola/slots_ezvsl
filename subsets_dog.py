#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import pandas as pd

def build_filename(row):
    # first column + '_' + zero-padded second column (6 digits)
    try:
        sec_int = int(row['second'])
    except Exception:
        try:
            sec_int = int(float(str(row['second']).strip()))
        except Exception:
            raise ValueError(f"Cannot parse 'second' value: {row['second']} for id {row['yt_id']}")
    return f"{row['yt_id']}_{str(sec_int).zfill(6)}"

def verify_exists(name: str, frames_root: Path, audio_root: Path) -> bool:
    frame_dir = frames_root / name
    audio_file = audio_root / f"{name}.wav"
    
    # Check if audio file exists
    if not audio_file.is_file():
        return False
    
    # Check if frame directory exists
    if not frame_dir.is_dir():
        return False
    
    # Check if frame directory contains at least one .jpg file
    # The dataset loader expects numbered .jpg files (000.jpg, 001.jpg, etc.)
    jpg_files = list(frame_dir.glob("*.jpg"))
    if not jpg_files:
        return False
    
    # Additional check: ensure there are enough frames for the dataset loader
    # The dataset loader uses the middle frame, so we need at least 1 frame
    # But let's be more lenient and just check for any .jpg files
    return True

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create two 150-row subsets of 'dog growling' (train split). "
            "Optionally verify that each item exists as a frames directory "
            "and as a .wav file in the provided roots before sampling."
        )
    )
    ap.add_argument("--seed", type=int, default=1, help="Random seed for sampling (default: 42)")
    ap.add_argument("--outdir", default=".", help="Directory to write subset1.csv and subset2.csv")
    ap.add_argument("--class_name", default="dog growling", help="Target class name (default: 'dog growling')")
    ap.add_argument("--subset1_size", type=int, default=300, help="Rows per subset (default: 300)")
    ap.add_argument("--subset2_size", type=int, default=50, help="Rows per subset (default: 50)")
    ap.add_argument("--frames_root", type=str, default=None, help="Root directory of total_video_frames (dirs named like Filename)")
    ap.add_argument("--audio_root", type=str, default=None, help="Root directory of total_video_3s_audio (files named Filename.wav)")
    ap.add_argument("--write_reports", action="store_true",
                    help="If set, write verified_pool.csv and missing_report.csv to --outdir")
    args = ap.parse_args()

    path = Path('/media/y/datasets/vggsound/train/vggsound.csv')
    if not path.exists():
        raise SystemExit(f"Input CSV not found: {path}")

    # Read CSV with explicit column names
    df = pd.read_csv(path, header=None, names=["yt_id","second","label","split"], dtype=str)
    # Clean spaces/quotes in label and split
    df["label"] = df["label"].str.strip().str.strip('"').str.strip("'")
    df["split"] = df["split"].str.strip()

    # Filter by split and class
    mask = (df["split"].str.lower() == "train") & (df["label"].str.lower() == args.class_name.lower())
    df_filt = df.loc[mask].copy()

    if df_filt.empty:
        raise SystemExit(f"No rows found with split=='train' and class=='{args.class_name}'.")

    # Build Filename column early (we'll use it for verification/filtering)
    df_filt["Filename"] = df_filt.apply(build_filename, axis=1)
    df_filt.rename(columns={"label":"class"}, inplace=True)

    # Optional existence verification
    verified = df_filt
    missing = pd.DataFrame(columns=df_filt.columns)
    if args.frames_root and args.audio_root:
        frames_root = Path(args.frames_root)
        audio_root = Path(args.audio_root)
        if not frames_root.exists():
            raise SystemExit(f"Frames root does not exist: {frames_root}")
        if not audio_root.exists():
            raise SystemExit(f"Audio root does not exist: {audio_root}")

        exists_mask = df_filt["Filename"].apply(
            lambda n: verify_exists(n, frames_root, audio_root)
        )
        verified = df_filt[exists_mask].reset_index(drop=True)
        missing = df_filt[~exists_mask].reset_index(drop=True)

        if verified.empty:
            raise SystemExit(
                "After verification, no items remain that exist in BOTH roots. "
                "Please check your paths or dataset."
            )

    # Ensure we have enough to sample 2 subsets
    needed = args.subset1_size + args.subset2_size
    if len(verified) < needed:
        raise SystemExit(
            f"Not enough verified rows to sample {needed} items. "
            f"Available after filtering: {len(verified)}."
        )

    # Deterministic sample without replacement
    df_sample = verified.sample(n=needed, random_state=args.seed).reset_index(drop=True)

    # Keep only requested columns
    df_sample = df_sample[["Filename","class"]]

    # Split into two subsets
    subset1 = df_sample.iloc[:args.subset1_size].copy()
    subset2 = df_sample.iloc[args.subset1_size:args.subset1_size + args.subset2_size].copy()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    subset1_path = outdir / "subset1.csv"
    subset2_path = outdir / "subset2.csv"
    subset1.to_csv(subset1_path, index=False)
    subset2.to_csv(subset2_path, index=False)

    # Optional reports
    if args.write_reports:
        verified[["Filename","class"]].to_csv(outdir / "verified_pool.csv", index=False)
        if not missing.empty:
            missing[["Filename","class"]].to_csv(outdir / "missing_report.csv", index=False)

    print(f"Wrote {len(subset1)} rows to {subset1_path}")
    print(f"Wrote {len(subset2)} rows to {subset2_path}")
    if args.frames_root and args.audio_root:
        print(f"Verified against frames_root={args.frames_root} and audio_root={args.audio_root}")
        if args.write_reports:
            print("Also wrote verified_pool.csv and (if applicable) missing_report.csv")

if __name__ == "__main__":
    main()
