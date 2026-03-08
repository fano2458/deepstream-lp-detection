"""YOLO26 → DeepStream-compatible ONNX export utilities.

This module patches the Ultralytics YOLO26 detection head so its output tensor
matches the format expected by the custom NvDsInferParseYolo parser:
    [batch, num_detections, 6]  where the last dim is [x1, y1, x2, y2, score, class_id]

Usage (CLI):
    python -m src.export -w best.pt --simplify --dynamic
"""

from __future__ import annotations

import os
import sys
import types
import warnings
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect, v10Detect
import ultralytics.utils
import ultralytics.models.yolo
import ultralytics.utils.tal as _tal_module

# Compatibility shim: some versions expose ultralytics.yolo as an alias
sys.modules.setdefault("ultralytics.yolo", ultralytics.models.yolo)
sys.modules.setdefault("ultralytics.yolo.utils", ultralytics.utils)


# ---------------------------------------------------------------------------
# Patch dist2bbox to drop the xywh branch (DeepStream needs xyxy output)
# ---------------------------------------------------------------------------

def _dist2bbox(
    distance: torch.Tensor,
    anchor_points: torch.Tensor,
    xywh: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """Convert distance predictions to xyxy bounding boxes.

    Args:
        distance: Predicted distances [lt, rb] from anchor centres.
        anchor_points: Anchor centre coordinates.
        xywh: Ignored; always returns xyxy for DeepStream compatibility.
        dim: Dimension along which to split/concatenate.

    Returns:
        Tensor of xyxy bounding boxes.
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)


_tal_module.dist2bbox.__code__ = _dist2bbox.__code__


# ---------------------------------------------------------------------------
# Custom output layer
# ---------------------------------------------------------------------------

class DeepStreamOutput(nn.Module):
    """Reshape YOLO26 detections to the NvDsInferParseYolo expected layout.

    Input shape:  [batch, 6, num_detections]  (raw detection head output)
    Output shape: [batch, num_detections, 6]  ([x1, y1, x2, y2, score, class_id])
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


# ---------------------------------------------------------------------------
# Detection head forward patch
# ---------------------------------------------------------------------------

def _forward_deepstream(self: Detect, x: list[torch.Tensor]) -> torch.Tensor:
    """Patched forward for the Detect head that is compatible with ONNX tracing.

    Replaces the default forward so that the exported graph does not contain
    any control-flow branches that the ONNX exporter cannot handle.

    Args:
        self: The Detect module instance (bound via types.MethodType).
        x: List of feature maps from the neck.

    Returns:
        Raw detection tensor before DeepStreamOutput post-processing.
    """
    x_detach = [xi.detach() for xi in x]
    if hasattr(self, "inference"):
        one2one = [
            torch.cat(
                (self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1
            )
            for i in range(self.nl)
        ]
        return self.inference(one2one)
    one2one = self.forward_head(x_detach, **self.one2one)
    return self._inference(one2one)


# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------

def prepare_model(weights: str | Path, device: torch.device, fuse: bool = True) -> nn.Module:
    """Load and patch a YOLO26 model for DeepStream ONNX export.

    Steps:
      1. Load the Ultralytics YOLO checkpoint.
      2. Deep-copy the underlying nn.Module (removes Ultralytics wrapper).
      3. Freeze parameters and switch to eval/float mode.
      4. Optionally fuse Conv+BN layers for inference efficiency.
      5. Patch Detect heads with a tracing-compatible forward.
      6. Patch C2f modules to use the split-friendly forward.

    Args:
        weights: Path to the ``.pt`` checkpoint file.
        device: Torch device to load the model onto.
        fuse: Whether to fuse Conv+BN layers before export.

    Returns:
        Patched nn.Module ready for ``torch.onnx.export``.
    """
    model = YOLO(str(weights))
    model = deepcopy(model.model).to(device)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()

    if fuse:
        model = model.fuse()

    for _name, m in model.named_modules():
        if isinstance(m, (Detect, v10Detect)):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
            if m.__class__.__name__ == "Detect":
                m.forward = types.MethodType(_forward_deepstream, m)
        elif isinstance(m, C2f):
            m.forward = m.forward_split

    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    img_size: tuple[int, int],
    batch: int,
    opset: int,
    dynamic: bool,
    simplify: bool,
    device: torch.device,
) -> None:
    """Export a patched YOLO26 model to ONNX with a DeepStreamOutput head.

    Args:
        model: Patched model from :func:`prepare_model`.
        output_path: Destination ``.onnx`` file path.
        img_size: ``(height, width)`` of the inference input.
        batch: Static batch size (ignored when ``dynamic=True``).
        opset: ONNX opset version.
        dynamic: If ``True``, export with a dynamic batch dimension.
        simplify: If ``True``, run onnxslim after export.
        device: Torch device for the dummy input tensor.
    """
    full_model = nn.Sequential(model, DeepStreamOutput())
    dummy = torch.zeros(batch, 3, *img_size, device=device)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    suppress_warnings()
    torch.onnx.export(
        full_model,
        dummy,
        str(output_path),
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    if simplify:
        import onnxslim

        model_onnx = onnx.load(str(output_path))
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, str(output_path))


def write_labels(model: nn.Module, output_path: str | Path = "labels.txt") -> None:
    """Write class names to a labels file for DeepStream.

    Args:
        model: The patched YOLO26 model (must have a ``names`` dict attribute).
        output_path: Destination path for the labels text file.
    """
    names: dict[int, str] = getattr(model, "names", {})
    if not names:
        return
    with open(output_path, "w", encoding="utf-8") as f:
        for name in names.values():
            f.write(f"{name}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def suppress_warnings() -> None:
    """Suppress noisy tracer/deprecation warnings during ONNX export."""
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> Namespace:
    """Parse CLI arguments for the export script.

    Returns:
        Parsed argument namespace.

    Raises:
        RuntimeError: If the weights file does not exist or conflicting
            dynamic/static batch options are given.
    """
    parser = ArgumentParser(description="Export YOLO26 .pt → DeepStream-compatible ONNX")
    parser.add_argument("-w", "--weights", required=True, type=str, help="Path to .pt checkpoint")
    parser.add_argument(
        "-s", "--size", nargs="+", type=int, default=[640],
        help="Input image size [H W] or single value for square (default: 640)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--simplify", action="store_true", help="Run onnxslim after export")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic batch dimension")
    parser.add_argument("--batch", type=int, default=1, help="Static batch size (default: 1)")
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise RuntimeError(f"Weights file not found: {args.weights}")
    if args.dynamic and args.batch > 1:
        raise RuntimeError("--dynamic and --batch > 1 are mutually exclusive")
    return args


def main(args: Namespace | None = None) -> None:
    """Run the full YOLO26 → ONNX export pipeline.

    Args:
        args: Pre-parsed namespace; if ``None``, arguments are read from
            ``sys.argv`` via :func:`parse_args`.
    """
    if args is None:
        args = parse_args()

    weights = Path(args.weights)
    img_size: tuple[int, int] = (args.size[0], args.size[-1])
    onnx_out = weights.with_suffix(".onnx")

    print(f"\nExporting {weights} → {onnx_out}")

    device = torch.device("cpu")
    model = prepare_model(weights, device)

    write_labels(model, output_path=weights.parent / "labels.txt")
    export_onnx(
        model=model,
        output_path=onnx_out,
        img_size=img_size,
        batch=args.batch,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        device=device,
    )
    print(f"Done: {onnx_out}\n")


if __name__ == "__main__":
    main()
