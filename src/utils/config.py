"""轻量配置系统：多个 YAML 深合并 + 命令行 `-o key=value` 覆盖。

用法::

    from src.utils.config import load_config

    cfg = load_config(
        "configs/base.yaml",
        "configs/model/iforest.yaml",
        overrides=["drift.name=temporal", "eval.buffer_sizes=[0,10,100]"],
    )
    assert cfg.model.name == "iforest"
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import yaml


class Config(dict):
    """支持 `cfg.data.name` 点号访问的字典。"""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(f"配置中不存在字段: {item}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def deep_update(base: Config, patch: Dict[str, Any]) -> Config:
    """把 patch 递归合并进 base，标量以 patch 为准。"""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = _wrap(value)
    return base


def _parse_scalar(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def set_by_path(cfg: Config, dotted: str, value: Any) -> None:
    """按 `a.b.c=value` 写入嵌套配置，中间层级不存在时自动创建。"""
    parts = dotted.split(".")
    node: Any = cfg
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = Config()
        node = node[part]
    node[parts[-1]] = _parse_scalar(value) if isinstance(value, str) else value


def load_config(*paths: Optional[str], overrides: Optional[Sequence[str]] = None) -> Config:
    cfg = Config()
    for path in paths:
        if path is None:
            continue
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"配置文件顶层必须是字典: {p}")
        deep_update(cfg, _wrap(data))

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"覆盖项必须形如 key=value，收到: {item!r}")
        key, value = item.split("=", 1)
        set_by_path(cfg, key.strip(), value.strip())
    return cfg


def dump_config(cfg: Config, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, allow_unicode=True, sort_keys=False)
    return path


def copy_tree(src: str | Path, dst: str | Path) -> None:
    """把本次实验用到的配置目录整体快照到产物目录里。"""
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)