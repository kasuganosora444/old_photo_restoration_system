from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GFPGAN_MODEL_PATH = PROJECT_ROOT / "models" / "gfpgan" / "GFPGANv1.4.pth"
REALESRGAN_MODEL_PATH = PROJECT_ROOT / "models" / "realesrgan" / "RealESRGAN_x4plus.pth"


class RestorationError(RuntimeError):
    """Raised when a selected restoration stage cannot run."""


@dataclass(frozen=True)
class RestorationOptions:
    face_enhancement: bool = True
    upscaling: bool = True
    scale: int = 2
    tile_size: int = 256


def get_model_status() -> dict[str, bool]:
    """Return the availability of local weights and Python inference packages."""
    return {
        "gfpgan_weight": GFPGAN_MODEL_PATH.is_file(),
        "realesrgan_weight": REALESRGAN_MODEL_PATH.is_file(),
        "gfpgan_package": find_spec("gfpgan") is not None,
        "realesrgan_package": find_spec("realesrgan") is not None,
    }


def restore_photo(
    image: Image.Image | None,
    options: RestorationOptions,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Image.Image, list[str]]:
    """Apply the selected GFPGAN and Real-ESRGAN restoration stages."""
    if image is None:
        raise RestorationError("请先选择一张照片。")
    if not options.face_enhancement and not options.upscaling:
        raise RestorationError("请至少启用一个修复模块。")
    if options.scale not in (2, 4):
        raise RestorationError("清晰化倍率仅支持 2 倍或 4 倍。")

    restored = ImageOps.exif_transpose(image).convert("RGB")
    completed_steps: list[str] = []

    if options.face_enhancement:
        _report(progress_callback, "正在进行 GFPGAN 人脸修复...")
        restored = _enhance_faces(restored)
        completed_steps.append("GFPGAN 人脸修复")

    if options.upscaling:
        _report(progress_callback, f"正在进行 Real-ESRGAN {options.scale} 倍清晰化...")
        restored = _upscale_image(restored, options.scale, options.tile_size)
        completed_steps.append(f"Real-ESRGAN {options.scale} 倍清晰化")

    _report(progress_callback, "修复完成。")
    return restored, completed_steps


def _report(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _ensure_available(package: str, model_path: Path) -> None:
    if not model_path.is_file():
        raise RestorationError(f"未找到模型权重：{model_path}")
    if find_spec(package) is None:
        raise RestorationError(
            f"未安装 {package} 推理依赖。请先按照 README 在 Conda 环境中安装项目依赖。"
        )


@lru_cache(maxsize=1)
def _get_face_restorer():
    _ensure_available("gfpgan", GFPGAN_MODEL_PATH)
    from gfpgan import GFPGANer

    return GFPGANer(
        model_path=str(GFPGAN_MODEL_PATH),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )


def _enhance_faces(image: Image.Image) -> Image.Image:
    bgr_image = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    _, _, restored_bgr = _get_face_restorer().enhance(
        bgr_image,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.5,
    )
    if restored_bgr is None:
        return image
    return Image.fromarray(restored_bgr[:, :, ::-1])


@lru_cache(maxsize=4)
def _get_upsampler(tile_size: int):
    _ensure_available("realesrgan", REALESRGAN_MODEL_PATH)
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    return RealESRGANer(
        scale=4,
        model_path=str(REALESRGAN_MODEL_PATH),
        model=model,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )


def _upscale_image(image: Image.Image, scale: int, tile_size: int) -> Image.Image:
    bgr_image = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    output_bgr, _ = _get_upsampler(tile_size).enhance(bgr_image, outscale=scale)
    return Image.fromarray(output_bgr[:, :, ::-1])
