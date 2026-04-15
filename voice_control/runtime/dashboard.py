"""Live terminal dashboard for the inference loop.

Three threads write into this thing — the audio callback, the classify
worker, and rich's own refresh thread reading it back — so every
mutator takes the lock. `__rich__` snapshots state under the same lock
and rebuilds the layout from scratch on each tick. That's cheap at 10 Hz
and avoids any cleverness about partial updates.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from voice_control.diagnostics import format_bytes, get_process_rss_bytes


MAX_SAMPLES = 500
RECENT_LIMIT = 6
BAR_WIDTH = 30


@dataclass(frozen=True)
class FireEvent:
    """One keypress we actually dispatched, with the numbers we want to show."""

    label: str
    confidence: float
    latency_ms: float


class Dashboard:
    """Live dashboard state + rich renderable. All mutators are thread-safe."""

    def __init__(
        self,
        *,
        console: Console,
        model_path: str,
        val_acc: float,
        epoch: int,
        device: str,
        commands: Sequence[str],
        confidence_threshold: float,
    ) -> None:
        self._console = console
        self._lock = threading.Lock()

        self._model_path = model_path
        self._val_acc = val_acc
        self._epoch = epoch
        self._device = device
        self._commands = list(commands)
        self._confidence_threshold = confidence_threshold

        self._status = "STARTING"
        self._status_color = "yellow"
        self._mic_device = "—"
        self._sample_rate = 0
        self._vad_threshold = 0.0
        self._mic_level = 0.0

        self._classifications = 0
        self._fires = 0
        self._preprocess_ms: deque[float] = deque(maxlen=MAX_SAMPLES)
        self._forward_ms: deque[float] = deque(maxlen=MAX_SAMPLES)
        self._end_to_end_ms: deque[float] = deque(maxlen=MAX_SAMPLES)
        self._recent: deque[FireEvent] = deque(maxlen=RECENT_LIMIT)
        self._startup_rss_bytes = get_process_rss_bytes()

    def set_status(self, status: str, color: str = "green") -> None:
        """Swap the status badge at the top of the status panel."""
        with self._lock:
            self._status = status
            self._status_color = color

    def set_mic_config(
        self, device_label: str, sample_rate: int, vad_threshold: float
    ) -> None:
        """Remember which mic we ended up on, so the header can show it."""
        with self._lock:
            self._mic_device = device_label
            self._sample_rate = sample_rate
            self._vad_threshold = vad_threshold

    def set_mic_level(self, level: float) -> None:
        """Update the level meter with the latest RMS reading."""
        with self._lock:
            self._mic_level = level

    def record_classification(self, preprocess_ms: float, forward_ms: float) -> None:
        """Add one preprocess + forward timing pair from a classify call."""
        with self._lock:
            self._classifications += 1
            self._preprocess_ms.append(preprocess_ms)
            self._forward_ms.append(forward_ms)

    def record_fire(self, label: str, confidence: float, latency_ms: float) -> None:
        """A keypress was dispatched — add its end-to-end latency and log it."""
        with self._lock:
            self._fires += 1
            self._end_to_end_ms.append(latency_ms)
            self._recent.append(FireEvent(label, confidence, latency_ms))

    def __rich__(self) -> Layout:
        """Called by rich.Live on every tick — returns a freshly-built layout."""
        layout = Layout(name="root")
        layout.split_column(
            Layout(self._render_header(), name="header", size=5),
            Layout(self._render_status(), name="status", size=5),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(self._render_stats(), name="stats", ratio=3),
            Layout(self._render_recent(), name="recent", ratio=2),
        )
        return layout

    def _render_header(self) -> Panel:
        with self._lock:
            commands = list(self._commands)
            device = self._device
            val_acc = self._val_acc
            epoch = self._epoch
            threshold = self._confidence_threshold
            model_path = self._model_path

        text = Text()
        text.append("voice-control ", style="bold cyan")
        text.append(f"· device={device} · ", style="dim")
        text.append(f"val_acc={val_acc:.4f}", style="green")
        text.append(f" · epoch={epoch} · ", style="dim")
        text.append(f"threshold={threshold:.2f}", style="magenta")
        text.append("\n")
        text.append(f"model: {model_path}\n", style="dim")
        text.append("commands: ", style="dim")
        text.append(" ".join(commands), style="bold magenta")
        return Panel(text, border_style="cyan", padding=(0, 1))

    def _render_status(self) -> Panel:
        with self._lock:
            status = self._status
            color = self._status_color
            device = self._mic_device
            sample_rate = self._sample_rate
            vad = self._vad_threshold
            level = self._mic_level

        scale = max(vad * 5, 1e-9)
        normalized = min(max(level / scale, 0.0), 1.0)
        filled = int(normalized * BAR_WIDTH)
        bar_style = "cyan" if level >= vad else "blue"

        text = Text()
        text.append("status  ", style="dim")
        text.append(f"{status:<10}", style=f"bold {color}")
        text.append("   mic  ", style="dim")
        text.append(f"{device} @ {sample_rate}Hz\n", style="white")
        text.append("level   ", style="dim")
        text.append("█" * filled, style=bar_style)
        text.append("░" * (BAR_WIDTH - filled), style="dim")
        text.append(f"  {level:.5f}  ", style="white")
        text.append(f"(vad={vad:.5f})", style="dim")
        return Panel(text, border_style="blue", padding=(0, 1))

    def _render_stats(self) -> Panel:
        with self._lock:
            n_class = self._classifications
            n_fire = self._fires
            pre = self._stats(self._preprocess_ms)
            fwd = self._stats(self._forward_ms)
            e2e = self._stats(self._end_to_end_ms)
            startup = self._startup_rss_bytes

        peak = get_process_rss_bytes()
        delta = peak - startup

        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_column(justify="right")
        table.add_column(justify="right")
        table.add_column(justify="right")

        table.add_row(
            Text("stage", style="bold dim"),
            Text("n", style="bold dim"),
            Text("p50", style="bold dim"),
            Text("p95", style="bold dim"),
            Text("max", style="bold dim"),
        )
        table.add_row(Text("preprocess", style="white"), *pre)
        table.add_row(Text("forward", style="white"), *fwd)
        table.add_row(Text("end-to-end", style="white"), *e2e)
        table.add_row("", "", "", "", "")
        table.add_row(
            Text("classifications", style="dim"),
            Text(str(n_class), style="white"),
            "",
            "",
            "",
        )
        table.add_row(
            Text("fires", style="dim"),
            Text(str(n_fire), style="white"),
            "",
            "",
            "",
        )
        table.add_row(
            Text("memory", style="dim"),
            Text(format_bytes(peak), style="white"),
            Text(f"Δ+{format_bytes(delta)}", style="dim"),
            "",
            "",
        )
        return Panel(table, title="stats", border_style="green", padding=(0, 1))

    def _render_recent(self) -> Panel:
        with self._lock:
            recent = list(self._recent)

        if not recent:
            body = Align.center(
                Text("waiting for speech…", style="dim italic"),
                vertical="middle",
            )
            return Panel(
                body, title="recent commands", border_style="magenta"
            )

        table = Table.grid(padding=(0, 2), expand=True)
        table.add_column(style="bold magenta", justify="left")
        table.add_column(style="green", justify="right")
        table.add_column(style="dim", justify="right")
        for fire in reversed(recent):
            table.add_row(
                fire.label,
                f"{fire.confidence:.3f}",
                f"{fire.latency_ms:.1f}ms",
            )
        return Panel(
            table,
            title="recent commands",
            border_style="magenta",
            padding=(0, 1),
        )

    def _stats(
        self, samples: deque[float]
    ) -> tuple[Text, Text, Text, Text]:
        n = len(samples)
        if n == 0:
            dash = Text("—", style="dim")
            return Text("0", style="dim"), dash, dash, dash
        ordered = sorted(samples)
        p50 = ordered[n // 2]
        p95 = ordered[min(int(n * 0.95), n - 1)]
        top = ordered[-1]
        return (
            Text(str(n), style="white"),
            Text(f"{p50:.2f}ms", style="green"),
            Text(f"{p95:.2f}ms", style="yellow"),
            Text(f"{top:.2f}ms", style="red"),
        )

    def print_summary(self) -> None:
        """Print the closing summary once Live has torn down the live region."""
        with self._lock:
            pre = self._stats(self._preprocess_ms)
            fwd = self._stats(self._forward_ms)
            e2e = self._stats(self._end_to_end_ms)
            startup = self._startup_rss_bytes
            fires = self._fires
            classifications = self._classifications

        peak = get_process_rss_bytes()
        delta = peak - startup

        table = Table(
            title="session summary",
            title_style="bold cyan",
            border_style="cyan",
            expand=False,
        )
        table.add_column("stage", style="white")
        table.add_column("n", justify="right")
        table.add_column("p50", justify="right", style="green")
        table.add_column("p95", justify="right", style="yellow")
        table.add_column("max", justify="right", style="red")
        table.add_row("preprocess", *pre)
        table.add_row("forward", *fwd)
        table.add_row("end-to-end", *e2e)

        self._console.print()
        self._console.print(table)
        self._console.print(
            f"[dim]classifications[/dim] {classifications}   "
            f"[dim]fires[/dim] {fires}   "
            f"[dim]memory[/dim] startup={format_bytes(startup)} "
            f"peak={format_bytes(peak)} delta={format_bytes(delta)}"
        )
