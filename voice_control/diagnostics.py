"""
Runtime and memory diagnostics for the voice-control pipeline.

Written for the evaluation section of the project report — we need
concrete numbers for wall time, CPU time, peak process RSS, and peak
Python heap usage, and eyeballing them from logs doesn't cut it.

Everything here is stdlib (tracemalloc + resource + time.perf_counter),
so there's no extra dependency to install. Two ways to use it:

    from voice_control.diagnostics import measure, diagnose

    @measure("classify")
    def hot_path(...): ...

    with diagnose("load_model") as d:
        model = VoiceCommandCNN(...)
    sys.stdout.write(d.format_line() + "\\n")

Run `python -m voice_control.diagnostics` to get a one-shot report on the current
checkpoint — loading cost, preprocessing cost, forward-pass cost, and
end-to-end latency, each with memory deltas.
"""

from __future__ import annotations

import functools
import logging
import resource
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, ParamSpec, TypeVar


logger = logging.getLogger(__name__)


P = ParamSpec("P")
R = TypeVar("R")


BYTES_PER_KIB = 1024
RSS_UNIT_BYTES_DARWIN = 1
RSS_UNIT_BYTES_LINUX = BYTES_PER_KIB


@dataclass
class DiagnosticResult:
    """
    One diagnostic sample.

    `python_peak_bytes` is whatever tracemalloc saw inside the block —
    that's Python-allocated memory only. Torch tensors on CPU live
    partly in C-extension allocators that tracemalloc can't see, so for
    model workloads the number is an under-estimate. Use `rss_delta_bytes`
    as the upper bound for "how much extra RAM did this cost the process".
    """

    label: str
    wall_seconds: float
    cpu_seconds: float
    python_peak_bytes: int
    rss_delta_bytes: int

    def format_line(self) -> str:
        """Single-line summary safe to dump to stdout or a log widget."""
        return (
            f"[{self.label}] "
            f"wall={self.wall_seconds * 1000:.2f}ms "
            f"cpu={self.cpu_seconds * 1000:.2f}ms "
            f"py_peak={_format_bytes(self.python_peak_bytes)} "
            f"rss_delta={_format_bytes(self.rss_delta_bytes)}"
        )


def _get_process_rss_bytes() -> int:
    """
    Return the process peak RSS in bytes.

    `ru_maxrss` has an infamous cross-platform unit gotcha: macOS
    reports bytes, Linux reports kibibytes, BSDs do their own thing.
    We normalise to bytes here so callers never have to think about it.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = RSS_UNIT_BYTES_DARWIN if sys.platform == "darwin" else RSS_UNIT_BYTES_LINUX
    return raw * unit


def _format_bytes(n: int) -> str:
    """Human-readable IEC formatter. Handles negative deltas for deallocation."""
    sign = "-" if n < 0 else ""
    value = float(abs(n))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < BYTES_PER_KIB:
            return f"{sign}{value:.1f}{unit}"
        value /= BYTES_PER_KIB
    return f"{sign}{value:.1f}TiB"


@contextmanager
def diagnose(label: str = "block") -> Iterator[DiagnosticResult]:
    """
    Context manager that times a block and records memory deltas.

    Yields a DiagnosticResult populated on exit. If tracemalloc was
    already tracing (e.g. a caller has it on for their own reasons),
    we leave it running instead of stomping their state.

    @param label: Short name for the block, shows up in format_line().
    """
    result = DiagnosticResult(
        label=label,
        wall_seconds=0.0,
        cpu_seconds=0.0,
        python_peak_bytes=0,
        rss_delta_bytes=0,
    )
    tracemalloc_started_here = not tracemalloc.is_tracing()
    if tracemalloc_started_here:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()

    rss_before = _get_process_rss_bytes()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        yield result
    finally:
        result.wall_seconds = time.perf_counter() - wall_start
        result.cpu_seconds = time.process_time() - cpu_start
        _, result.python_peak_bytes = tracemalloc.get_traced_memory()
        if tracemalloc_started_here:
            tracemalloc.stop()
        result.rss_delta_bytes = _get_process_rss_bytes() - rss_before


def measure(
    label: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that wraps a function in `diagnose` and writes the result
    to stdout on every call.

    Meant for ad-hoc profiling — drop it on a hot path while you're
    collecting numbers, pull it off when you're done. For finer-grained
    measurements inside a function, use the `diagnose` context manager.

    @param label: Optional label override. Defaults to fn.__qualname__.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        display = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with diagnose(display) as result:
                value = fn(*args, **kwargs)
            logger.info(result.format_line())
            return value

        return wrapper

    return decorator


def report_model_footprint(model, label: str = "model") -> int:
    """
    Count parameters and sum their actual byte footprint.

    Uses `element_size() * numel()` per tensor instead of the naive
    `params * 4` so mixed-precision or int8 checkpoints are reported
    correctly. Returns the total byte count so the caller can feed it
    into the end-of-run summary.
    """
    total_bytes = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()

    logger.info(
        f"[{label}] params={total_params:,} bytes={_format_bytes(total_bytes)}"
    )
    return total_bytes


def report_checkpoint_size(path: str, label: str = "checkpoint") -> int:
    """Report the on-disk size of a checkpoint file in bytes."""
    import os

    size = os.path.getsize(path)
    logger.info(f"[{label}] path={path} size={_format_bytes(size)}")
    return size


def _run_cli_report() -> None:
    """
    One-shot diagnostic run used by `python diagnostics.py`.

    Produces the numbers needed for the evaluation section:
      - checkpoint size on disk
      - model parameter count and in-memory bytes
      - model load cost (time + RSS delta)
      - preprocess cost for a single 1s window
      - model forward cost for a single window
      - end-to-end cost (preprocess + forward), averaged over 100 runs
    """
    import torch

    from voice_control.audio.processing import NUM_SAMPLES, SAMPLE_RATE, get_mel_transform, preprocess
    from voice_control.config import InferenceConfig
    from voice_control.model import VoiceCommandCNN

    config = InferenceConfig()

    report_checkpoint_size(config.model_path)

    with diagnose("load_checkpoint") as load_diag:
        checkpoint = torch.load(
            config.model_path, map_location="cpu", weights_only=True
        )
        model = VoiceCommandCNN(num_classes=len(checkpoint["labels"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    logger.info(load_diag.format_line())

    report_model_footprint(model)

    mel_transform = get_mel_transform()
    dummy_waveform = torch.randn(1, NUM_SAMPLES)

    with diagnose("preprocess_single") as pre_diag:
        mel = preprocess(dummy_waveform, SAMPLE_RATE, mel_transform)
    logger.info(pre_diag.format_line())

    mel_batch = mel.unsqueeze(0)
    with torch.no_grad():
        for _ in range(10):
            model(mel_batch)

        with diagnose("forward_single") as fwd_diag:
            model(mel_batch)
        logger.info(fwd_diag.format_line())

        iterations = 100
        with diagnose(f"end_to_end_x{iterations}") as e2e_diag:
            for _ in range(iterations):
                mel_i = preprocess(dummy_waveform, SAMPLE_RATE, mel_transform)
                model(mel_i.unsqueeze(0))
        logger.info(e2e_diag.format_line())

        per_call_ms = (e2e_diag.wall_seconds / iterations) * 1000
        logger.info(
            f"[end_to_end_avg] per_call={per_call_ms:.2f}ms over {iterations} runs"
        )


if __name__ == "__main__":
    from voice_control.log_config import configure_logging

    configure_logging()
    _run_cli_report()
