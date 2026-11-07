import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
try:
    from tqdm import tqdm
except ImportError:         
    def tqdm(x, **kw):
        return x

SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

CLASSES = {"AiArtData": ("ai", 1), "RealArt": ("real", 0)}


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def inspect(path):
    """Return width, height, mode, format, error. error is '' for a good file."""
    try:
        with Image.open(path) as im:
            im.verify()               
        with Image.open(path) as im:      
            im.load()                      
            return im.width, im.height, im.mode, im.format, ""
    except Exception as e:               
        return None, None, None, None, f"{type(e).__name__}: {e}"


def main(data_dir, out_reports, out_splits):
    data_dir, out_reports, out_splits = Path(data_dir), Path(out_reports), Path(out_splits)
    out_reports.mkdir(exist_ok=True)
    out_splits.mkdir(exist_ok=True)

    rows = []
    for folder, (label, label_int) in CLASSES.items():
        files = [p for p in (data_dir / folder).rglob("*") if p.is_file()]
        for p in tqdm(files, desc=folder):
            w, h, mode, fmt, err = inspect(p)
            rows.append({
                "path": p.as_posix(), "class": label, "label": label_int,
                "file_size": p.stat().st_size, "width": w, "height": h,
                "mode": mode, "format": fmt, "sha256": sha256_of(p),
                "corrupt": bool(err), "error": err,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_reports / "audit.csv", index=False)

    good = df[~df["corrupt"]]
    dup_mask = good.duplicated("sha256", keep=False)
    extra_copies = int(good.duplicated("sha256", keep="first").sum())
    cross = good.groupby("sha256")["class"].nunique()
    cross_hashes = set(cross[cross > 1].index)

    lines = ["=== COUNTS PER CLASS (all files) ===", df["class"].value_counts().to_string(), ""]
    for col in ["width", "height", "file_size"]:
        stats = good.groupby("class")[col].agg(["min", "median", "max"])
        lines += [f"=== {col}: min / median / max per class ===", stats.to_string(), ""]
    lines += [
        "=== FORMATS ===", good.groupby(["class", "format"]).size().to_string(), "",
        "=== COLOUR MODES ===", good.groupby(["class", "mode"]).size().to_string(), "",
        f"Corrupt / unreadable files : {int(df['corrupt'].sum())}",
        f"Files in duplicate groups  : {int(dup_mask.sum())}",
        f"Extra copies (removable)   : {extra_copies}",
        f"Hashes present in BOTH classes: {len(cross_hashes)}",
    ]
    summary = "\n".join(lines)
    (out_reports / "summary.txt").write_text(summary)
    print("\n" + summary)

    clean = good[~good["sha256"].isin(cross_hashes)].drop_duplicates("sha256", keep="first")
    cols = ["path", "class", "label", "sha256"]
    train, rest = train_test_split(clean[cols], test_size=0.30, stratify=clean["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out_splits / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} images  {part['class'].value_counts().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--splits", default="splits")
    a = ap.parse_args()
    main(a.data, a.reports, a.splits)