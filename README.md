# Fourier Filtering Explorer

An interactive, keyboard-driven classroom demonstration of **frequency-domain filtering**, built with NumPy and Matplotlib. Designed as a companion to [Fourier Series Explorer](https://github.com/csalvaggio/fourier_series_explorer): the same standalone-script approach, synchronized six-panel layout, and keyboard-focused teaching workflow.

## Installation

Python **3.10+**, NumPy **2.0+**, Matplotlib **3.9+**, and a desktop graphical Matplotlib backend are required.

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 fourier_filtering_explorer.py
```

Click inside the figure to give it keyboard focus. Resize or maximize with your desktop window controls.

## Six synchronized plots

The application displays these plots in a 3 × 2 arrangement:

1. **[Selected function name]** — the original sampled signal (for example, **Rectangle**).
2. **Analysis window / Windowing function: [window name]** — the original signal and the **Windowed input** overlaid.
3. **Original spectrum** — the magnitude spectrum of the windowed input, before filtering.
4. **[Response family] [Filter type] response** — the frequency-domain transfer function; for example, **Ideal Low-pass response**.
5. **Filtered spectrum** — the magnitude spectrum after multiplication by the filter response.
6. **Filtered [selected function name]** — the filtered output overlaid with the **Windowed input**.

**Original spectrum** means *before filtering*, not necessarily *before windowing*. When Hann or Hamming is selected, the spectrum is calculated from the windowed input.

The three frequency-domain plots share the full **±Nyquist** axis range, which changes with the sample count. Dashed vertical references mark low-pass/high-pass cutoffs; dotted references mark the positive and negative edges for band-pass/band-reject filters. Edges above Nyquist lie outside the displayed range. Individual stems indicate discrete DFT bins, not a continuous Fourier transform. Press `l` to toggle linear and dB magnitude spectra.

## Keyboard controls

| Key | Action |
|---|---|
| `1`–`9`, `0` | Choose input function |
| `f` | Cycle low-pass / high-pass / band-pass / band-reject |
| `t` | Cycle ideal / Gaussian / Butterworth responses |
| `v` | Cycle no window / Hann / Hamming |
| `↑` / `↓` | Increase / decrease cutoff or band center, by 0.25 cycles per x-unit |
| `b` / `B` | Decrease / increase band width by 0.5 cycles per x-unit |
| `o` / `O` | Decrease / increase Butterworth order (1–12) |
| `w` / `W` | Decrease / increase signal width; sine/sawtooth cycle count; noninteger sine frequency |
| `[` / `]` | Decrease / increase separation for two rectangles, or halve / double pulse-train repetition count |
| `←` / `→` | Circularly shift the input by one sample |
| `+` / `-` | Double / halve sample count (32–512) |
| `l` | Toggle linear versus dB magnitude spectrum |
| `r` or `Home` | Restore defaults (retain sample count) |
| `q` or `Esc` | Quit |

The default is a 256-sample rectangle and ideal low-pass with cutoff 2 cycles per x-unit. The domain is `[-2, 2)` (length 4), so DFT bins are separated by **0.25 cycles per x-unit**. Cutoff adjustments therefore occur in exact DFT-bin increments, including zero: at zero cutoff, low-pass passes only DC and high-pass removes only DC (for all response families).

## Input functions

| Key | Signal | Teaching purpose |
|---|---|---|
| `1` | Rectangle | Sinc-like spectrum; ringing caused by an ideal low-pass |
| `2` | Triangle | Faster decay of high frequencies |
| `3` | Integer-cycle sine | Spectral line concentrated at the matching DFT bins |
| `4` | Periodized Gaussian | Smooth shapes and compact frequency-domain energy |
| `5` | Integer-period sawtooth | Harmonics from jumps |
| `6` | Two rectangles | Interference patterns and their changes with separation |
| `7` | Rectangle train | Discrete harmonic comb and duty cycle |
| `8` | Noninteger-cycle sine | Spectral leakage from endpoint mismatch |
| `9` | Chirp | Time-varying frequency content, broad spectrum, endpoint mismatch |
| `0` | Signal with periodic noise | Smooth underlying signal with eight cycles of coherent sinusoidal interference across the record |

**Note on periodicity:** A DFT *always* treats its finite input sequence as one period of a periodic extension. A noninteger-cycle sinusoid has a discontinuity in that extension, giving leakage across many DFT bins. The chirp is also intentionally nonperiodic across the sampled record. Windowing can reduce sidelobe leakage but widens the central spectral lobe and changes amplitudes.

## Filtering convention

For `N` equally spaced samples over length `L = 4`, the code uses

\[
X_k = \frac{1}{N}\sum_{n=0}^{N-1} x_n e^{-i 2\pi kn/N},
\qquad f_k = \frac{k}{L}
\]

with the displayed spectrum FFT-shifted so DC appears at the center. Filtering is multiplication by a **real, even**, zero-phase transfer function, followed by an inverse transform.

- **Ideal low-pass:** 1 for `|f| <= cutoff`, 0 outside. High-pass is 1 for `|f| >= cutoff`. The equality boundary is included for both; at a bin exactly on the cutoff both pass that bin.
- **Gaussian low-pass:** `exp(-0.5*(|f|/cutoff)^2)`. The displayed **cutoff is the Gaussian scale parameter**, not a universal -3 dB frequency. Gaussian high-pass is `1 - low-pass`.
- **Butterworth low-pass:** `1/sqrt(1+(|f|/cutoff)^(2*order))`, Butterworth high-pass: `1/sqrt(1+(cutoff/|f|)^(2*order))` for nonzero `f`, with DC = 0.
- **Band-pass:** the product of the high-pass edge at `center - bandwidth/2` and the low-pass edge at `center + bandwidth/2`.
- **Band-reject:** 1 minus that band-pass response.

The band filter products are pedagogical combinations of low- and high-pass magnitude responses; they are **not** specialized classical Butterworth band transformations, and the nominal edges are not generally the composite response's exact -3 dB points. All three filter families multiply FFT bins directly, yielding circular (periodic-boundary) filtering. Thus even nonperiodic records are implicitly periodically extended.

**Critical window interpretation:** `windowed_input = original × window_values` is the input to the filter. Therefore the output is compared to the **windowed signal**, not to the unwindowed original. Applying a Hann/Hamming window is not a free way to remove leakage: it modifies the signal itself. A separate workflow (windowed spectrum analysis of a signal, without physically windowing the signal before reconstruction) is another valid application of windows, but not what this explorer implements.

The RMS change on the status line is `rms(filtered_signal - windowed_input)`, **not** an error metric against the original unwindowed function.

## Demonstrations for class

1. **Gibbs-like ringing:** Select `1`, ideal low-pass, reduce cutoff with `↓`. Compare with Gaussian and Butterworth using `t`.
2. **High-pass edge emphasis:** Press `f` once from default and adjust cutoff.
3. **Windowing and leakage:** Select `8` then toggle `l` to dB view and `v` to compare no window, Hann and Hamming. Adjust the noninteger sine frequency with `w` / `W`.
4. **Select a band of harmonics:** Select `7`; switch to band-pass (`f` twice from reset) and adjust center with arrows, width with `b`/`B`.
5. **Change filter order:** Select Butterworth via `t`, then use `o`/`O` to contrast slow and sharp transition bands.
6. **Circular filtering at boundaries:** Select `0` for a reflectance line scan (or `8`/`9` for other nonperiodic examples), then examine filtering near the interval endpoints.
7. **Remove periodic interference:** Select `n` for a smooth signal with an eight-cycle sinusoid at **2 cycles per x-unit** (DFT bins ±8). Set ideal **Band-reject**, center cutoff **2.00**, bandwidth **1.00**, and window **None**. Both interfering spectral lines are removed while most of the underlying signal is retained. Switch to **Band-pass** with the same settings to isolate the interference along with any underlying energy in the selected band.

## Tests

Run the standard-library unittest suite (no additional test packages required):

```bash
python3 -m unittest -v test_fourier_filtering_explorer.py
```

Tests cover sample identity for all signals, nonperiodic line-scan generation, bin-aligned interference, window definitions, real/even filter responses, spectral leakage of integer vs. noninteger sinusoids, and GUI-independent keyboard callbacks. They use Matplotlib's noninteractive `Agg` backend. A desktop smoke test remains useful to verify focus and display geometry with your macOS/Linux GUI backend.

## Files

- `fourier_filtering_explorer.py`: standalone application
- `test_fourier_filtering_explorer.py`: tests
- `requirements.txt`: direct dependencies
- `README.md`: instructions and mathematical notes

## Scope

This is a one-dimensional, finite-record DFT demonstration. Filtering assumes periodic extension of the sampled record and does not perform padding to approximate linear convolution.
