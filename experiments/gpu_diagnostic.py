"""Diagnose WSL GPU compute and OpenGL rendering support.

Run headlessly with ``python -m experiments.gpu_diagnostic`` or pass
``--show-window`` to display a short hardware-rendered OpenGL animation.
If PyTorch is installed, the diagnostic also performs a CUDA matrix operation.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one diagnostic check."""

    name: str
    status: str
    detail: str


def check_wsl() -> CheckResult:
    """Report whether the process has WSL GPU and display integration."""
    release = platform.release()
    is_wsl = "microsoft" in release.lower() or "WSL_DISTRO_NAME" in os.environ
    dxg = os.path.exists("/dev/dxg")
    display = os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")
    status = "PASS" if is_wsl and dxg and display else "WARN"
    detail = f"WSL={is_wsl}, /dev/dxg={dxg}, display={display or 'missing'}"
    return CheckResult("WSL integration", status, detail)


def check_nvidia() -> CheckResult:
    """Query the Windows NVIDIA driver exposed to WSL."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return CheckResult("NVIDIA driver", "SKIP", "nvidia-smi was not found")

    command = [
        executable,
        "--query-gpu=name,driver_version,compute_cap,memory.total",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "nvidia-smi returned an error"
        return CheckResult("NVIDIA driver", "FAIL", detail)
    return CheckResult("NVIDIA driver", "PASS", completed.stdout.strip())


def check_pytorch() -> CheckResult:
    """Run a small CUDA operation when PyTorch is available."""
    try:
        import torch
    except ImportError:
        return CheckResult(
            "PyTorch CUDA",
            "SKIP",
            "PyTorch is not installed in this interpreter",
        )

    if not torch.cuda.is_available():
        detail = f"torch={torch.__version__}, built for CUDA={torch.version.cuda}"
        return CheckResult("PyTorch CUDA", "FAIL", detail)

    try:
        left = torch.randn((512, 512), device="cuda")
        right = torch.randn((512, 512), device="cuda")
        result = left @ right
        torch.cuda.synchronize()
        detail = (
            f"torch={torch.__version__}, device={torch.cuda.get_device_name(0)}, "
            f"result mean={result.mean().item():.6f}"
        )
    except Exception as error:  # CUDA errors vary by driver and PyTorch version.
        return CheckResult("PyTorch CUDA", "FAIL", repr(error))
    return CheckResult("PyTorch CUDA", "PASS", detail)


def _gl_string(gl: ctypes.CDLL, parameter: int) -> str:
    gl.glGetString.argtypes = [ctypes.c_uint]
    gl.glGetString.restype = ctypes.c_char_p
    value = gl.glGetString(parameter)
    return value.decode(errors="replace") if value else "unknown"


def check_opengl(show_window: bool, seconds: float) -> CheckResult:
    """Create a WSLg OpenGL context and optionally show an animation."""
    try:
        import pygame
    except ImportError:
        return CheckResult("OpenGL rendering", "SKIP", "pygame-ce is not installed")

    try:
        pygame.display.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if not show_window:
            flags |= pygame.HIDDEN
        pygame.display.set_mode((640, 360), flags)

        gl = ctypes.CDLL("libGL.so.1")
        vendor = _gl_string(gl, 0x1F00)  # GL_VENDOR
        renderer = _gl_string(gl, 0x1F01)  # GL_RENDERER
        version = _gl_string(gl, 0x1F02)  # GL_VERSION
        gl.glClearColor.argtypes = [
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        gl.glClear.argtypes = [ctypes.c_uint]

        start = time.monotonic()
        running = True
        while running and time.monotonic() - start < seconds:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            phase = (time.monotonic() - start) * 2.0
            red = (1.0 + math.sin(phase)) / 2.0
            gl.glClearColor(red, 0.2, 1.0 - red, 1.0)
            gl.glClear(0x00004000)  # GL_COLOR_BUFFER_BIT
            pygame.display.flip()
            pygame.time.wait(16)
    except Exception as error:
        return CheckResult("OpenGL rendering", "FAIL", repr(error))
    finally:
        pygame.display.quit()

    detail = f"vendor={vendor}; renderer={renderer}; version={version}"
    software_renderers = ("llvmpipe", "softpipe", "software rasterizer")
    if any(name in renderer.lower() for name in software_renderers):
        detail += "; software rendering detected (try GALLIUM_DRIVER=d3d12)"
        return CheckResult("OpenGL rendering", "FAIL", detail)
    return CheckResult("OpenGL rendering", "PASS", detail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show a short animated OpenGL window instead of testing invisibly.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="How long to render the OpenGL test (default: 3 seconds).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seconds <= 0:
        raise SystemExit("--seconds must be greater than zero")

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    results = [
        check_wsl(),
        check_nvidia(),
        check_pytorch(),
        check_opengl(show_window=args.show_window, seconds=args.seconds),
    ]
    for result in results:
        print(f"[{result.status:4}] {result.name}: {result.detail}")

    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
