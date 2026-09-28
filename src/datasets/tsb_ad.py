"""TSB-AD 数据加载。

期望的目录结构（由 `data/download_tsb_ad.py` 生成）::

    data/raw/tsb-ad/
        001_NAB_id_1_Facility_tr_10000_1st_2016-01-01/
            train.npy          # (T_train, C) float
            test.npy           # (T_test, C) float
            test_label.npy     # (T_test,) 0/1
            meta.json          # 可选，{"name":..., "source":...}

如果官方仓库的落地格式不同，只需要改本文件的 `_load_one`，
其余模块全部依赖 `Bundle` / `Series` 抽象，不受影响。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .base import Bundle, Series

TRAIN_PATTERNS = ("train.npy", "*_train.npy", "train.csv", "*_train.csv")
TEST_PATTERNS = ("test.npy", "*_test.npy", "test.csv", "*_test.csv")
LABEL_PATTERNS = (
    "test_label.npy",
    "*_test_label.npy",
    "label.npy",
    "test_labels.npy",
    "test_label.csv",
    "*_test_label.csv",
    "label.csv",
)
LABEL_COLUMNS = ("Label", "label", "is_anomaly", "anomaly", "target")
NON_FEATURE_COLUMNS = ("timestamp", "time", "date", "datetime", "index")
LABEL_SUFFIXES = ("_label", "_labels", "_anomaly")


def _find(sub_dir: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        hits = sorted(sub_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def _read_table(path: Path, want_label: bool = False) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path))
    if path.suffix in (".csv", ".txt", ".tsv"):
        import pandas as pd

        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if want_label:
            for col in LABEL_COLUMNS:
                if col in df.columns:
                    return df[col].to_numpy()
            raise ValueError(f"{path}: 找不到标签列，已尝试 {LABEL_COLUMNS}")
        drop = [c for c in df.columns if str(c).lower() in NON_FEATURE_COLUMNS]
        drop += [c for c in df.columns if str(c).endswith(LABEL_SUFFIXES)]
        drop += [c for c in LABEL_COLUMNS if c in df.columns]
        return df.drop(columns=drop).to_numpy()
    raise ValueError(f"不支持的文件类型: {path}")


def _load_one(sub_dir: Path) -> Series:
    train_path = _find(sub_dir, TRAIN_PATTERNS)
    test_path = _find(sub_dir, TEST_PATTERNS)
    label_path = _find(sub_dir, LABEL_PATTERNS)

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"{sub_dir}: 缺少训练或测试文件。"
            f"已尝试 {TRAIN_PATTERNS} / {TEST_PATTERNS}"
        )

    train = _read_table(train_path).astype(np.float32)
    test = _read_table(test_path).astype(np.float32)

    if label_path is not None:
        label = np.asarray(_read_table(label_path, want_label=True)).ravel()
        label = (label > 0).astype(np.int8)
    else:
        # 部分 TSB-AD 子集把标签放在测试文件的最后一列
        raw = _read_table(test_path)
        if raw.shape[1] != test.shape[1] and raw.shape[1] == test.shape[1] + 1:
            label = (raw[:, -1] > 0).astype(np.int8)
        else:
            raise FileNotFoundError(f"{sub_dir}: 找不到异常标签")

    meta = {}
    meta_path = sub_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("name", sub_dir.name)
    meta.setdefault("path", str(sub_dir))

    if label.shape[0] != test.shape[0]:
        raise ValueError(f"{sub_dir}: 标签长度 {label.shape[0]} 与测试长度 {test.shape[0]} 不一致")

    return Series(
        name=sub_dir.name,
        train=train,
        train_label=np.zeros(train.shape[0], dtype=np.int8),
        test=test,
        test_label=label,
        meta=meta,
    )


def list_subsets(root: str | Path) -> List[str]:
    root = Path(root)
    if not root.exists():
        return []
    names = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _find(child, TRAIN_PATTERNS) and _find(child, TEST_PATTERNS):
            names.append(child.name)
    return names


def load_tsb_ad(
    root: str | Path,
    subsets: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    min_anomaly_ratio: float = 0.0,
) -> Bundle:
    root = Path(root)
    available = list_subsets(root)
    if not available:
        raise FileNotFoundError(
            f"{root} 下没有找到任何数据集子目录。\n"
            f"请先运行 `python data/download_tsb_ad.py`，"
            f"或检查 data/README.md 里的目录结构说明。"
        )

    if subsets:
        chosen = [s for s in available if s in set(subsets)]
        missing = set(subsets) - set(chosen)
        if missing:
            raise KeyError(f"以下子集不存在: {sorted(missing)}")
    else:
        chosen = available
    if limit is not None:
        chosen = chosen[: int(limit)]

    series: List[Series] = []
    for name in chosen:
        s = _load_one(root / name)
        if s.anomaly_ratio < min_anomaly_ratio:
            continue
        series.append(s)

    if not series:
        raise RuntimeError("过滤后没有任何序列，请放宽 min_anomaly_ratio 或 subsets")

    return Bundle(name=root.name, series=series, meta={"root": str(root), "n_subsets": len(series)})