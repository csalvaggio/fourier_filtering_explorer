#!/usr/bin/env python3
"""Interactive classroom exploration of discrete frequency-domain filtering.

Python 3.10+, NumPy 2+, Matplotlib 3.9+. Run this script directly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
import matplotlib.pyplot as plt

LENGTH = 4.0
SAMPLE_COUNTS = (32, 64, 128, 256, 512)
SIGNALS = (
    "Rectangle", "Triangle", "Sine (integer cycles)", "Gaussian",
    "Sawtooth", "Two rectangles", "Rectangle train",
    "Sine (noninteger cycles)", "Chirp (nonperiodic)",
)
FILTERS = ("Low-pass", "High-pass", "Band-pass", "Band-reject")
FAMILIES = ("Ideal", "Gaussian", "Butterworth")
WINDOWS = ("None", "Hann", "Hamming")

# Matplotlib registers its own shortcuts on each figure. Some overlap with
# our teaching controls (f=fullscreen, v=forward, o=zoom, l=y-scale, etc.).
# Remove only our keys from Matplotlib's defaults, leaving other shortcuts
# such as Ctrl+S/Cmd+S to save figures intact.
EXPLORER_KEYS = frozenset(
    tuple(str(i) for i in range(1, 10))
    + ("f", "t", "v", "up", "down", "b", "B", "o", "O", "w", "W",
       "[", "]", "left", "right", "+", "=", "shift+=", "-",
       "l", "r", "home", "q", "escape")
)


def reserve_explorer_keys() -> None:
    """Keep Matplotlib's built-in keymaps from handling explorer keys."""
    for name in plt.rcParams:
        if name.startswith("keymap."):
            bindings = plt.rcParams[name]
            if any(binding in EXPLORER_KEYS for binding in bindings):
                plt.rcParams[name] = [binding for binding in bindings
                                      if binding not in EXPLORER_KEYS]


@dataclass(frozen=True)
class State:
    n: int = 256
    signal: int = 0
    width: float = 1.3
    separation: float = 1.6
    repetitions: int = 4
    cycles: float = 5.0
    fractional_cycles: float = 5.5
    shift: int = 0
    filter_type: int = 0
    family: int = 0
    cutoff: float = 2.0  # cycles per x-unit
    bandwidth: float = 1.0  # full width for band filters, cycles per x-unit
    order: int = 4
    window: int = 0
    log_spectrum: bool = False


def coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-LENGTH / 2, LENGTH / 2, n, endpoint=False)
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=LENGTH / n))
    return x, freq


def sample_signal(state: State) -> np.ndarray:
    """Signals sampled on [-2, 2); shifts are integer circular shifts."""
    n = state.n
    x, _ = coordinates(n)
    w = np.clip(state.width, 2 * LENGTH / n, 0.92 * LENGTH)
    if state.signal == 0:
        y = (np.abs(x) < w / 2).astype(float)
        y[np.isclose(np.abs(x), w / 2, atol=1e-12)] = 0.5
    elif state.signal == 1:
        y = np.maximum(0, 1 - 2 * np.abs(x) / w)
    elif state.signal == 2:
        y = np.sin(2 * np.pi * state.cycles * np.arange(n) / n)
    elif state.signal == 3:
        sigma = max(w / 3, LENGTH / n)
        y = sum(np.exp(-0.5 * ((x + k * LENGTH) / sigma) ** 2) for k in (-1, 0, 1))
        y /= y.max()
    elif state.signal == 4:
        y = 2 * np.mod(state.cycles * np.arange(n) / n, 1.0) - 1
    elif state.signal == 5:
        half_sep = min(state.separation, LENGTH - w) / 2
        y = np.maximum((np.abs(x - half_sep) < w / 4).astype(float),
                       (np.abs(x + half_sep) < w / 4).astype(float))
    elif state.signal == 6:
        phase = np.mod(state.repetitions * np.arange(n) / n, 1.0)
        y = (phase < min(w / LENGTH, 0.95)).astype(float)
    elif state.signal == 7:
        y = np.sin(2 * np.pi * state.fractional_cycles * np.arange(n) / n)
    elif state.signal == 8:
        t = np.arange(n) / n
        y = np.sin(2 * np.pi * (1.5 * t + 10.0 * t * t))
    else:
        raise ValueError("Unknown signal")
    return np.roll(y, state.shift)


def windowing_function(n: int, kind: int) -> np.ndarray:
    if kind == 0:
        return np.ones(n)
    if kind == 1:
        return np.hanning(n)
    if kind == 2:
        return np.hamming(n)
    raise ValueError("Unknown window")


def filter_transfer_function(frequencies: np.ndarray, state: State) -> np.ndarray:
    """Real, even transfer response with unity passbands; f in cycles/unit."""
    a = np.abs(frequencies)
    c = state.cutoff
    bw = max(state.bandwidth, 1e-12)
    lo, hi = max(0.0, c - bw / 2), c + bw / 2
    order = state.order

    def low(cut: float) -> np.ndarray:
        if cut <= 0:
            return (a == 0).astype(float)
        if state.family == 0:
            return (a <= cut).astype(float)
        if state.family == 1:
            return np.exp(-0.5 * (a / cut) ** 2)
        return 1.0 / np.sqrt(1.0 + (a / cut) ** (2 * order))

    def high(cut: float) -> np.ndarray:
        if cut <= 0:
            return (a > 0).astype(float)
        if state.family == 0:
            return (a >= cut).astype(float)
        if state.family == 1:
            return 1.0 - np.exp(-0.5 * (a / cut) ** 2)
        # Butterworth high-pass with zero DC (not just 1 - low-pass magnitude).
        out = np.zeros_like(a)
        nz = a > 0
        out[nz] = 1.0 / np.sqrt(1.0 + (cut / a[nz]) ** (2 * order))
        return out

    if state.filter_type == 0:
        return low(c)
    if state.filter_type == 1:
        return high(c)
    if state.filter_type == 2:
        return high(lo) * low(hi)
    if state.filter_type == 3:
        return 1.0 - high(lo) * low(hi)
    raise ValueError("Unknown filter")


def compute(state: State) -> dict[str, np.ndarray | float]:
    x, f = coordinates(state.n)
    original = sample_signal(state)
    window_values = windowing_function(state.n, state.window)
    windowed_input = original * window_values
    spectrum = np.fft.fftshift(np.fft.fft(windowed_input)) / state.n
    filter_response = filter_transfer_function(f, state)
    filtered_spectrum = spectrum * filter_response
    filtered_signal = np.fft.ifft(np.fft.ifftshift(filtered_spectrum) * state.n).real
    return dict(x=x, f=f, original=original, window_values=window_values, windowed_input=windowed_input,
                spectrum=spectrum, filter_response=filter_response, filtered_spectrum=filtered_spectrum,
                filtered_signal=filtered_signal,
                rms=np.sqrt(np.mean((filtered_signal - windowed_input) ** 2)))


def normalized(state: State) -> State:
    nyquist = state.n / (2 * LENGTH)
    return replace(state, width=float(np.clip(state.width, 2 * LENGTH / state.n, 0.95 * LENGTH)),
                   separation=float(np.clip(state.separation, 0, LENGTH)),
                   repetitions=int(np.clip(state.repetitions, 1, state.n // 4)),
                   cycles=float(np.clip(round(state.cycles), 1, state.n // 4)),
                   fractional_cycles=float(np.clip(state.fractional_cycles, 0.5, state.n / 3)),
                   cutoff=float(np.clip(state.cutoff, 0.0, nyquist)),
                   bandwidth=float(np.clip(state.bandwidth, 0.5, 2 * nyquist)),
                   order=int(np.clip(state.order, 1, 12)))


class Explorer:
    def __init__(self, state: State | None = None):
        reserve_explorer_keys()
        self.state = normalized(state or State())
        self.fig, axes = plt.subplots(3, 2, figsize=(14, 9.1))
        self.axes = axes.ravel()
        self.fig.subplots_adjust(top=0.79, bottom=0.09, hspace=0.61, wspace=0.25,
                                 left=0.085, right=0.97)
        self.fig.canvas.manager.set_window_title("Fourier Filtering Explorer")
        self.fig.text(0.5, 0.975, "Fourier Filtering Explorer", ha="center", va="top",
                      fontsize=19, weight="bold")
        self.instructions = self.fig.text(
            0.5, 0.931,
            "1–9 signal  |  f filter  |  t response  |  v window  |  ↑/↓ cutoff  |  b/B bandwidth  |  o/O order\n"
            "w/W width  |  [/] separation or repetitions  |  ←/→ shift  |  +/- samples  |  l dB  |  r reset  |  q quit",
            ha="center", va="top", fontsize=10, linespacing=1.6,
            family="monospace")
        self.detail = self.fig.text(0.5, 0.849, "", ha="center", va="center", fontsize=11)
        self.status = self.fig.text(0.5, 0.023, "", ha="center", va="center",
                                    family="monospace", fontsize=10)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.refresh()

    def on_key(self, event):
        key = event.key
        s = self.state
        if key in ("q", "escape"):
            plt.close(self.fig)
            return
        if key in ("r", "home"):
            self.state = State(n=s.n)
            self.refresh()
            return
        if key in tuple(str(i) for i in range(1, 10)):
            s = replace(s, signal=int(key) - 1)
        elif key == "f":
            s = replace(s, filter_type=(s.filter_type + 1) % len(FILTERS))
        elif key == "t":
            s = replace(s, family=(s.family + 1) % len(FAMILIES))
        elif key == "v":
            s = replace(s, window=(s.window + 1) % len(WINDOWS))
        elif key == "up":
            s = replace(s, cutoff=s.cutoff + 0.25)
        elif key == "down":
            s = replace(s, cutoff=s.cutoff - 0.25)
        elif key in ("b", "B"):
            s = replace(s, bandwidth=s.bandwidth + (0.5 if key == "B" else -0.5))
        elif key in ("o", "O"):
            s = replace(s, order=s.order + (1 if key == "O" else -1))
        elif key in ("w", "W"):
            delta = (LENGTH / s.n) * (2 if key == "W" else -2)
            if s.signal == 7:
                s = replace(s, fractional_cycles=s.fractional_cycles + (0.25 if key == "W" else -0.25))
            elif s.signal in (2, 4):
                s = replace(s, cycles=s.cycles + (1 if key == "W" else -1))
            else:
                s = replace(s, width=s.width + delta)
        elif key in ("[", "]"):
            if s.signal == 5:
                s = replace(s, separation=s.separation + (-0.125 if key == "[" else 0.125))
            elif s.signal == 6:
                s = replace(s, repetitions=(max(1, s.repetitions // 2) if key == "[" else s.repetitions * 2))
            else:
                return
        elif key == "left":
            s = replace(s, shift=s.shift - 1)
        elif key == "right":
            s = replace(s, shift=s.shift + 1)
        elif key in ("+", "=", "shift+="):
            s = replace(s, n=min(SAMPLE_COUNTS[-1], s.n * 2))
        elif key == "-":
            s = replace(s, n=max(SAMPLE_COUNTS[0], s.n // 2))
        elif key == "l":
            s = replace(s, log_spectrum=not s.log_spectrum)
        else:
            return
        self.state = normalized(s)
        self.refresh()

    def draw_cutoff_references(self, axis):
        """Show the actual cutoff boundary/boundaries on a frequency axis."""
        s = self.state
        if s.filter_type < 2:
            edges = (s.cutoff,)
        else:
            edges = (max(0.0, s.cutoff - s.bandwidth / 2),
                     s.cutoff + s.bandwidth / 2)
        # Band edges can lie beyond Nyquist: leave these off-screen rather
        # than misrepresenting the effective response.
        nyquist = s.n / (2 * LENGTH)
        for edge in sorted(set(edges)):
            if edge > nyquist:
                continue
            for position in ((0.0,) if edge == 0 else (-edge, edge)):
                axis.axvline(position, color="tab:red" if s.filter_type >= 2 else "0.40",
                             ls=":" if s.filter_type >= 2 else "--", lw=1.0,
                             alpha=0.75, zorder=1)

    def refresh(self):
        s = self.state
        d = compute(s)
        x, f = d["x"], d["f"]
        original, windowed_input = d["original"], d["windowed_input"]
        spec, output_spec = d["spectrum"], d["filtered_spectrum"]
        out, filter_response = d["filtered_signal"], d["filter_response"]
        for ax in self.axes:
            ax.clear()
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=9)
        ax = self.axes
        ax[0].plot(x, original, color="tab:blue", lw=2)
        ax[0].set_title(SIGNALS[s.signal])
        ax[0].set_ylabel("Amplitude")
        ax[1].plot(x, original, color="0.73", lw=1.5, label="Original function")
        ax[1].plot(x, windowed_input, color="tab:orange", lw=2, label="Windowed input")
        ax[1].set_title(f"Analysis window / Windowing function: {WINDOWS[s.window]}")
        ax[1].legend(loc="upper right", fontsize=8)
        ax[1].set_ylabel("Amplitude")
        def mag(z):
            a = np.abs(z)
            return 20 * np.log10(np.maximum(a, 1e-8)) if s.log_spectrum else a
        def plot_spectrum(axis, values, title, color):
            axis.vlines(f, mag(values), 0 if not s.log_spectrum else -160,
                        color=color, linewidth=1.0)
            axis.plot(f, mag(values), ".", color=color, markersize=3)
            axis.set_title(title)
            axis.set_ylabel("Magnitude (dB)" if s.log_spectrum else "|DFT| / N")
            if s.log_spectrum:
                axis.set_ylim(-165, max(5, float(np.max(mag(values))) + 6))
        plot_spectrum(ax[2], spec, "Original spectrum", "tab:blue")
        ax[3].plot(f, filter_response, color="tab:green", lw=2, marker=".", markersize=3)
        ax[3].set_ylim(-0.08, 1.08)
        ax[3].set_ylabel("Filter response H(f)")
        ax[3].set_title(f"{FAMILIES[s.family]} {FILTERS[s.filter_type]} response")
        plot_spectrum(ax[4], output_spec, "Filtered spectrum", "tab:purple")
        for frequency_axis in (ax[2], ax[3], ax[4]):
            self.draw_cutoff_references(frequency_axis)
        ax[5].plot(x, windowed_input, color="0.62", lw=1.8, ls="--", label="Windowed input")
        ax[5].plot(x, out, color="tab:red", lw=2, label="Filtered output")
        ax[5].set_title(f"Filtered {SIGNALS[s.signal]}")
        ax[5].set_ylabel("Amplitude")
        ax[5].legend(loc="upper right", fontsize=8)
        for i, a in enumerate(ax):
            a.set_xlabel("x" if i in (0, 1, 5) else "Frequency (cycles per x-unit)")
            if i in (0, 1, 5):
                a.set_xlim(-LENGTH / 2, LENGTH / 2)
            else:
                a.set_xlim(-s.n / (2 * LENGTH), s.n / (2 * LENGTH))
        self.detail.set_text(
            f"{SIGNALS[s.signal]}   |   {FAMILIES[s.family]} {FILTERS[s.filter_type]}   |   "
            f"Cutoff = {s.cutoff:.2f} cycles/unit"
            + (f"   |   Bandwidth = {s.bandwidth:.2f}" if s.filter_type >= 2 else "")
            + (f"   |   Order = {s.order}" if s.family == 2 else ""))
        self.status.set_text(
            f"N = {s.n}  |  Δf = {1 / LENGTH:g} cycles/unit  |  Nyquist = {s.n / (2 * LENGTH):g}  |  "
            f"Window = {WINDOWS[s.window]}  |  RMS change = {d['rms']:.4f}")
        self.fig.canvas.draw_idle()


def main() -> None:
    # Matplotlib keeps weak references to bound event callbacks. Keep the
    # Explorer alive for the entire GUI event loop or its keyboard handler
    # may be garbage-collected while the figure is still displayed.
    explorer = Explorer()
    plt.show()


if __name__ == "__main__":
    main()
