"""Load the four frozen checkpoints and their comparable target layers."""

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    run_dir: str
    kind: str
    layer_name: str


SPECS = (
    ModelSpec("ResNet18", "resnet18_patient_clean_seed24", "resnet18", "layer4[-1]"),
    ModelSpec("ResNet50", "resnet50_patient_clean_seed24", "resnet50", "layer4[-1]"),
    ModelSpec("Improved1_Original", "attention_resnet50_patient_clean_seed24",
              "original", "backbone[-1][-1]"),
    ModelSpec("Improved2_SpatialFixed", "attention_resnet50_spatial_fixed_patient_clean_seed24",
              "spatial_fixed", "backbone[-1][-1]"),
)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instantiate(spec):
    if spec.kind == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        layer = model.layer4[-1]
    elif spec.kind == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        layer = model.layer4[-1]
    else:
        from model import CustomModel, CustomModelSpatialFixed
        model = (CustomModel if spec.kind == "original" else CustomModelSpatialFixed)(2)
        layer = model.backbone[-1][-1]
    return model, layer


def load_models(device):
    manifest_hash = sha256(ROOT / "data" / "metadata" / "image_manifest.csv")
    loaded = {}
    reference_stats = None
    for spec in SPECS:
        directory = ROOT / "runs" / spec.run_dir
        config_path = directory / "run_config.json"
        checkpoint_path = directory / "best_model.pth"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["class_to_idx"] != {"HCC": 0, "Hemangioma": 1}:
            raise ValueError("Unexpected class mapping for " + spec.name)
        if config["manifest_sha256"] != manifest_hash:
            raise ValueError("Training manifest differs for " + spec.name)
        if reference_stats is None:
            reference_stats = config["normalization"]
        elif reference_stats != config["normalization"]:
            raise ValueError("Normalization differs across models")
        model, layer = instantiate(spec)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        loaded[spec.name] = {
            "spec": spec,
            "model": model,
            "target_layer": layer,
            "config": config,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": sha256(checkpoint_path),
            "best_epoch": int(checkpoint["epoch"]),
        }
    return loaded, reference_stats
