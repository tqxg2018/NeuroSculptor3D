"""Tiny YAML config helper: `load_config(path, overrides=["train.max_lr=1e-4", ...])`."""
import copy
from types import SimpleNamespace

import yaml


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    return ns


def _parse_value(value: str):
    v = yaml.safe_load(value)
    if isinstance(v, str):          # PyYAML reads e.g. "1e-4" (no decimal point) as a string
        try:
            return float(v)
        except ValueError:
            pass
    return v


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for ov in overrides or ():
        key, value = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)
    return _to_ns(cfg)
