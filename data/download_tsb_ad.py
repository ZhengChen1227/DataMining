"""下载并转换 TSB-AD 到本仓库的标准格式。

设计原则：宁可明确失败，也不要静默生成错的数据。
如果自动识别失败，脚本会打印它实际找到了什么，让你按 data/README.md
里的结构手动放好，而不是悄悄跳过某个子集。

用法::

    python data/download_tsb_ad.py --root data/raw/tsb-ad
    python data/download_tsb_ad.py --source ../third_party/TSB-AD --root data/raw/tsb-ad
    python data/download_tsb_ad.py --no-convert
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_URL = "https://github.com/TheDatumOrg/TSB-AD"
DEFAULT_CLONE = Path("third_party/TSB-AD")

TRAIN_TOKENS = ("train",)
TEST_TOKENS = ("test",)
LABEL_TOKENS = ("test_label", "label", "labels", "anomaly")


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def clone_repo(dest: Path) -> Path:
    if dest.exists():
        print(f"已存在，跳过克隆: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", REPO_URL, str(dest)])
    return dest


def _classify(name: str) -> Optional[str]:
    """把一个文件名分类成 train / test / label / 无关。"""
    stem = Path(name).stem.lower()
    if stem.endswith("_label") or "label" in stem:
        return "label"
    if any(f"_{t}" in stem or stem == t for t in TRAIN_TOKENS):
        return "train"
    if any(f"_{t}" in stem or stem == t for t in TEST_TOKENS):
        return "test"
    return None


def _read_any(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if path.suffix in (".csv", ".txt", ".tsv"):
        import pandas as pd

        sep = "\t" if path.suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep).to_numpy()
    raise ValueError(f"不支持的文件类型: {path}")


def _extract_label(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim > 1:
        # 多列时取任一列里出现过异常的判定，通常是最后一列
        array = array[:, -1] if array.shape[1] > 1 else array[:, 0]
    return (array.ravel() > 0).astype(np.int8)


def _extract_features(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array[:, None].astype(np.float32)
    return array.astype(np.float32)


def discover(source: Path) -> Dict[str, Dict[str, Path]]:
    """扫描数据目录，把文件按子集名分组。

    支持两种组织方式：
    1. 每个子集一个目录，目录内含 train/test/label 文件；
    2. 所有文件平铺在一个目录，靠 `{subset}_train.npy` 这类前缀区分。
    """
    groups: Dict[str, Dict[str, Path]] = {}

    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".npy", ".csv", ".tsv", ".txt"):
            continue
        kind = _classify(path.name)
        if kind is None:
            continue

        # 方式 1：目录名作为子集名
        if (path.parent / f"train{path.suffix}").exists() or _directory_looks_complete(path.parent):
            subset = path.parent.name
        else:
            # 方式 2：前缀作为子集名
            stem = path.stem
            subset = None
            for token in ("_train", "_test_label", "_test", "_label"):
                if stem.endswith(token):
                    subset = stem[: -len(token)]
                    break
            subset = subset or path.parent.name

        entry = groups.setdefault(subset, {})
        entry.setdefault(kind, path)

    return {k: v for k, v in groups.items() if "train" in v and "test" in v}


def _directory_looks_complete(directory: Path) -> bool:
    names = {p.stem.lower() for p in directory.iterdir() if p.is_file()}
    has_train = any(n.endswith("train") or n == "train" for n in names)
    has_test = any(("test" in n) and ("label" not in n) for n in names)
    return has_train and has_test


def convert(groups: Dict[str, Dict[str, Path]], root: Path, overwrite: bool = False) -> int:
    written = 0
    for subset, files in sorted(groups.items()):
        target = root / subset
        if target.exists() and not overwrite:
            print(f"  跳过（已存在）: {subset}")
            continue
        target.mkdir(parents=True, exist_ok=True)

        train = _extract_features(_read_any(files["train"]))
        test = _extract_features(_read_any(files["test"]))

        if "label" in files:
            label = _extract_label(_read_any(files["label"]))
        elif test.shape[1] == train.shape[1] + 1:
            # 部分数据把标签放在测试文件最后一列
            label = (test[:, -1] > 0).astype(np.int8)
            test = test[:, :-1]
        else:
            print(f"  跳过（找不到标签）: {subset}")
            continue

        if len(label) != len(test):
            print(
                f"  跳过（标签长度 {len(label)} 与测试长度 {len(test)} 不一致）: {subset}"
            )
            continue
        if test.shape[1] != train.shape[1]:
            print(
                f"  跳过（通道数不一致 {train.shape[1]} vs {test.shape[1]}）: {subset}"
            )
            continue

        np.save(target / "train.npy", train.astype(np.float32))
        np.save(target / "test.npy", test.astype(np.float32))
        np.save(target / "test_label.npy", label.astype(np.int8))
        (target / "meta.json").write_text(
            json.dumps(
                {
                    "name": subset,
                    "source": "TSB-AD",
                    "n_channels": int(train.shape[1]),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "anomaly_ratio": float(label.mean()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  已写入: {subset}  C={train.shape[1]} T_train={len(train)} "
              f"T_test={len(test)} 异常占比={label.mean():.4f}")
        written += 1
    return written


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="下载并转换 TSB-AD")
    parser.add_argument("--root", default="data/raw/tsb-ad", help="输出目录（标准格式）")
    parser.add_argument("--clone", default=str(DEFAULT_CLONE), help="官方仓库克隆位置")
    parser.add_argument("--source", default=None, help="已有的官方仓库路径，给出后跳过克隆")
    parser.add_argument("--no-convert", action="store_true", help="只克隆不转换")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的子集")
    args = parser.parse_args(list(argv) if argv is not None else None)

    source = Path(args.source) if args.source else clone_repo(Path(args.clone))
    if args.no_convert:
        print(f"\n已克隆到 {source}，未做转换。按 data/README.md 的结构手动整理后再运行本脚本。")
        return 0

    print(f"\n扫描 {source} ...")
    groups = discover(source)
    if not groups:
        print(
            "\n没有找到可以自动识别的数据。脚本实际找到的文件里，"
            "没有任何文件名同时满足「有 train、有 test」的成对条件。\n"
            "请检查：\n"
            "  1) 官方仓库是否需要先运行它自己的数据下载脚本"
            "（通常位于 Datasets/ 或 README 的数据获取一节）；\n"
            "  2) 数据是否解压完成；\n"
            "  3) 命名是否为 {subset}_train.npy / {subset}_test.npy / "
            "{subset}_test_label.npy。\n"
            "确认后按 data/README.md 的结构放到 data/raw/tsb-ad/ 下即可，"
            "本脚本不是唯一入口。"
        )
        return 2

    print(f"识别到 {len(groups)} 个子集，开始转换 ...")
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    written = convert(groups, root, overwrite=args.overwrite)

    print(f"\n完成：写入 {written} 个子集到 {root}")
    if written == 0:
        return 1
    print("校验命令：python -c \"from src.datasets.tsb_ad import list_subsets; "
          f"print(list_subsets('{root}'))\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())