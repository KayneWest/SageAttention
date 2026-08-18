"""Regression checks for PyTorch CUDA stream correctness.

PyTorch extensions must launch kernels on the caller's current CUDA stream.
A launch that omits the fourth execution-configuration argument silently uses
CUDA stream 0 and can race with tensors produced on a non-default stream.
"""

from __future__ import annotations

import re
from pathlib import Path


CSRC = Path(__file__).resolve().parents[1] / "csrc"
CUDA_LAUNCH = re.compile(r"<<<.*>>>")


def test_all_cuda_launches_use_an_explicit_stream() -> None:
    launches: list[str] = []
    offenders: list[str] = []

    for path in sorted(CSRC.rglob("*.cu")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not CUDA_LAUNCH.search(line):
                continue
            location = f"{path.relative_to(CSRC)}:{line_number}: {line.strip()}"
            launches.append(location)
            if "stream" not in line:
                offenders.append(location)

    assert launches, "no CUDA kernel launches found; the audit is not exercising csrc"
    assert not offenders, "CUDA launches missing an explicit stream:\n" + "\n".join(
        offenders
    )

