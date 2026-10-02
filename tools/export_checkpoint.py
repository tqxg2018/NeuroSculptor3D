"""Maintainer tool: convert a training checkpoint into the release format (safetensors + config.json).

    python tools/export_checkpoint.py outputs/ss_sc_sub-0001/model_epoch200.pth release/ss_sc_sub-0001
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
os.makedirs(dst, exist_ok=True)
# clone: the rotary-embedding buffer is shared by all prior layers, which safetensors cannot store
save_file({k: v.detach().clone().contiguous() for k, v in ck["model_state_dict"].items()}, os.path.join(dst, "model.safetensors"),
          metadata={"epoch": str(ck.get("epoch"))})
cfg = ck["config"]
cfg["trellis"]["image_ckpt"] = "JeffreyXiang/TRELLIS-image-large"   # local paths of the training machine -> HF id
cfg.pop("output_dir", None)
with open(os.path.join(dst, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print(dst, sum(v.numel() for v in ck["model_state_dict"].values()) / 1e6, "M parameters")
