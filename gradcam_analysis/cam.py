"""Standard logit-based Grad-CAM using PyTorch forward and tensor hooks."""

import numpy as np
import torch
from torch.nn import functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activation = None
        self.gradient = None
        self.handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, module, inputs, output):
        self.activation = output
        self.gradient = None
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient):
        self.gradient = gradient

    def close(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def generate(self, image_tensor, target_class):
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM accepts one image per call")
        if target_class not in (0, 1):
            raise ValueError("Target class must be HCC=0 or Hemangioma=1")
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(image_tensor)
            if logits.shape != (1, 2):
                raise ValueError("Unexpected model output shape")
            probabilities = torch.softmax(logits.detach(), dim=1)[0]
            logits[0, target_class].backward()
        if self.activation is None or self.gradient is None:
            raise ValueError("Missing activation or target-class gradient")
        activation = self.activation
        gradient = self.gradient
        if activation.ndim != 4 or activation.shape != gradient.shape:
            raise ValueError("Activation and gradient shape mismatch")
        if min(activation.shape[-2:]) < 2:
            raise ValueError("Target layer has no spatial resolution")
        if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
            raise ValueError("Nonfinite activation or gradient")
        gradient_l1 = float(gradient.detach().abs().sum().item())
        if gradient_l1 <= 0:
            raise ValueError("Target-class gradient is zero")

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        raw = F.relu((weights * activation).sum(dim=1, keepdim=True))
        raw = F.interpolate(raw, size=image_tensor.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0].detach().cpu().numpy()
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite Grad-CAM map")
        low, high = float(raw.min()), float(raw.max())
        if high <= low or high <= 0:
            raise ValueError("All-zero or constant Grad-CAM map")
        normalized = ((raw - low) / (high - low)).astype(np.float32)
        return {
            "cam": normalized,
            "probabilities": probabilities.cpu().numpy(),
            "logits": logits.detach().cpu().numpy()[0],
            "activation_shape": list(activation.shape),
            "gradient_l1": gradient_l1,
            "raw_cam_min": low,
            "raw_cam_max": high,
        }
