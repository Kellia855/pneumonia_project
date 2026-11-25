"""Create the small data folders that go into the GitHub repo / Docker image.

The full dataset is too big for the repo, so this writes resized 224x224 JPEG samples:
  data/train/<CLASS>   replay data used when retraining (so the model does not forget the original data)
  data/val/<CLASS>     used to decide whether a retrained model is promoted
  data/test/<CLASS>    used by the /evaluate endpoint and for demo predictions
  demo_uploads/<CLASS> a ready-made batch of images the model has never seen, for the upload + retrain demo

It reads data_manifest.csv, which the notebook saved (columns: path, label, split, md5, local_path).

Usage on Kaggle (after the notebook has run, so the dataset and manifest exist):
  !python make_repo_data.py --manifest /kaggle/working/pneumonia_project/data_manifest.csv \
      --out /kaggle/working/repo_data/data --demo-out /kaggle/working/repo_data/demo_uploads

Usage on your computer, with the full Kaggle dataset downloaded to a folder (any layout, e.g. chest_xray/...):
  python scripts/make_repo_data.py --manifest data_manifest.csv --data-root path/to/full_dataset \
      --out data_small --demo-out demo_uploads
  Files are matched by folder/class/filename and checked against the manifest's MD5 hash.
  Write --out to a NEW folder, then replace data/ with it.
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
import tensorflow as tf

CLASSES = ["NORMAL", "PNEUMONIA"]


def save_resized(src, dst):
    img = tf.io.decode_image(tf.io.read_file(str(src)), channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224), method="bilinear")
    img = tf.cast(tf.round(tf.clip_by_value(img, 0, 255)), tf.uint8)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tf.io.write_file(str(dst), tf.io.encode_jpeg(img, quality=95))


def md5_of(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


class LocalIndex:
    """Finds a manifest image inside a local copy of the dataset (paths in the manifest are Kaggle paths).

    Tries, in order: same filename with matching MD5, any file with matching MD5 (renamed/moved files),
    the filename from the manifest's local_path column, then the same filename without an MD5 check.
    """

    EXT = {".jpg", ".jpeg", ".png"}

    def __init__(self, root):
        self.files = [p for p in Path(root).rglob("*") if p.suffix.lower() in self.EXT and not p.name.startswith("._")]
        self.by_name = defaultdict(list)
        for p in self.files:
            self.by_name[p.name].append(p)
        self._by_md5 = None
        self.unverified = 0
        print(f"Indexed {len(self.files)} images under {root}")

    def by_md5(self):
        if self._by_md5 is None:
            print("Hashing local images to match renamed files (one-off, can take a minute)...")
            self._by_md5 = {}
            for p in self.files:
                self._by_md5.setdefault(md5_of(p), p)
        return self._by_md5

    def _named(self, path_str):
        if not isinstance(path_str, str) or not path_str or path_str == "nan":
            return []
        kp = Path(path_str.replace("\\", "/"))
        tail = [x.lower() for x in kp.parts[-3:]]
        return sorted(self.by_name.get(kp.name, []), key=lambda p: [x.lower() for x in p.parts[-3:]] != tail)

    def find(self, row):
        md5 = row.get("md5")
        md5 = md5 if isinstance(md5, str) and md5 else None
        for c in self._named(row.get("path")) + self._named(row.get("local_path")):
            if md5 is None or md5_of(c) == md5:
                return c
        if md5 and md5 in self.by_md5():
            return self.by_md5()[md5]
        for c in self._named(row.get("local_path")) + self._named(row.get("path")):
            self.unverified += 1               # same name, different bytes (e.g. the file was re-saved)
            return c
        return None


def pick_source(row, index=None):
    if index is not None:
        found = index.find(row)
        if found is None:
            raise FileNotFoundError(f"{row['path']} (md5 {row.get('md5')}) not found under --data-root")
        return found
    local = Path(str(row.get("local_path", "")))
    return local if local.exists() else Path(row["path"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default=None,
                    help="folder with a local copy of the full dataset (when not running on Kaggle)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--demo-out", default="demo_uploads")
    ap.add_argument("--train-per-class", type=int, default=300)
    ap.add_argument("--val-per-class", type=int, default=100)
    ap.add_argument("--test-per-class", type=int, default=150)
    ap.add_argument("--demo-per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    index = LocalIndex(args.data_root) if args.data_root else None
    if args.data_root and Path(args.out).resolve() == Path(args.data_root).resolve():
        raise SystemExit("--out must be a new folder, not the dataset folder itself")
    plan = [("train", args.train_per_class, Path(args.out) / "train"),
            ("val", args.val_per_class, Path(args.out) / "val"),
            ("test", args.test_per_class, Path(args.out) / "test"),
            ("new_data", args.demo_per_class, Path(args.demo_out))]
    for split, n, out_dir in plan:
        for label in CLASSES:
            sub = df[(df["split"] == split) & (df["label"] == label)]
            sub = sub.sample(n=min(n, len(sub)), random_state=args.seed)
            for i, (_, row) in enumerate(sub.iterrows()):
                save_resized(pick_source(row, index), out_dir / label / f"{row['md5'][:12]}.jpg")
            print(f"{split:9s} {label:10s} {len(sub):4d} images -> {out_dir / label}")
    if index is not None and index.unverified:
        print(f"Note: {index.unverified} images matched by filename only (their bytes differ from the manifest's MD5).")


if __name__ == "__main__":
    main()