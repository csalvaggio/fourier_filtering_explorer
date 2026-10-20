"""Run with: python3 -m unittest -v test_fourier_filtering_explorer.py"""
import unittest
import gc
import weakref
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
from fourier_filtering_explorer import (
    State, Explorer, main, FILTERS, FAMILIES, SIGNALS, compute, normalized,
    sample_signal, filter_transfer_function, coordinates, windowing_function,
    EXPLORER_KEYS, reserve_explorer_keys, SAMPLE_COUNTS,
)


class FilteringTests(unittest.TestCase):
    def test_identity_all_signals(self):
        for n in (32, 128, 512):
            for k in range(len(SIGNALS)):
                s = State(n=n, signal=k, filter_type=0, family=0, cutoff=n / 8)
                d = compute(s)
                np.testing.assert_allclose(d['filtered_signal'], d['original'], atol=1e-12)

    def test_window_definitions(self):
        n = 128
        np.testing.assert_allclose(windowing_function(n, 0), 1)
        self.assertAlmostEqual(windowing_function(n, 1)[0], 0)
        self.assertAlmostEqual(windowing_function(n, 2)[0], 0.08)
        for kind in (1, 2):
            d = compute(State(window=kind, cutoff=32))
            np.testing.assert_allclose(d['filtered_signal'], d['windowed_input'], atol=1e-12)

    def test_even_real_filters(self):
        _, f = coordinates(256)
        for family in range(len(FAMILIES)):
            for kind in range(len(FILTERS)):
                h = filter_transfer_function(f, State(family=family, filter_type=kind))
                self.assertTrue(np.all(np.isfinite(h)))
                self.assertTrue(np.all((h >= 0) & (h <= 1)))
                for j, val in enumerate(f):
                    match = np.where(np.isclose(f, -val))[0]
                    if len(match):
                        self.assertAlmostEqual(h[j], h[match[0]], places=12)
                self.assertLess(np.max(np.abs(np.imag(compute(State(family=family, filter_type=kind))['filtered_signal']))), 1e-12)

    def test_low_high_extremes(self):
        _, f = coordinates(256)
        low = filter_transfer_function(f, State(filter_type=0, family=0, cutoff=1))
        high = filter_transfer_function(f, State(filter_type=1, family=0, cutoff=1))
        np.testing.assert_allclose(low + high, 1 + (np.abs(f) == 1))

    def test_zero_cutoff_dc_only_and_sample_bounds(self):
        self.assertEqual(SAMPLE_COUNTS, (32, 64, 128, 256, 512))
        self.assertEqual(normalized(State(cutoff=-2)).cutoff, 0)
        _, f = coordinates(256)
        for family in range(len(FAMILIES)):
            low = filter_transfer_function(f, State(family=family, cutoff=0))
            high = filter_transfer_function(f, State(family=family, cutoff=0, filter_type=1))
            np.testing.assert_array_equal(low, (f == 0).astype(float))
            np.testing.assert_array_equal(high, (f != 0).astype(float))

    def test_display_labels_and_sample_controls(self):
        app = Explorer(State(n=512, signal=5, filter_type=3))
        try:
            titles = [ax.get_title() for ax in app.axes]
            self.assertEqual(titles[0], 'Two rectangles')
            self.assertEqual(titles[1], 'Analysis window / Windowing function: None')
            self.assertEqual(titles[2], 'Original spectrum')
            self.assertIn('Band-reject', titles[3])
            self.assertEqual(titles[4], 'Filtered spectrum')
            self.assertEqual(titles[5], 'Filtered Two rectangles')
            self.assertEqual(app.axes[5].get_legend().get_texts()[0].get_text(), 'Windowed input')
            app.on_key(SimpleNamespace(key='+'))
            self.assertEqual(app.state.n, 512)
            for _ in range(6):
                app.on_key(SimpleNamespace(key='-'))
            self.assertEqual(app.state.n, 32)
            app.state = normalized(State(cutoff=0.25))
            app.on_key(SimpleNamespace(key='down'))
            self.assertEqual(app.state.cutoff, 0)
        finally:
            plt.close(app.fig)

    def test_full_nyquist_axes_and_cutoff_references(self):
        for n in SAMPLE_COUNTS:
            for filter_type, expected in ((0, [-2.0, 2.0]),
                                          (1, [-2.0, 2.0]),
                                          (2, [-3.0, -1.0, 1.0, 3.0]),
                                          (3, [-3.0, -1.0, 1.0, 3.0])):
                app = Explorer(State(n=n, filter_type=filter_type,
                                     cutoff=2.0, bandwidth=2.0))
                try:
                    nyquist = n / (2 * 4.0)
                    for index in (2, 3, 4):
                        np.testing.assert_allclose(app.axes[index].get_xlim(),
                                                   [-nyquist, nyquist])
                        references = sorted(float(line.get_xdata()[0])
                                            for line in app.axes[index].lines
                                            if line.get_linestyle() in ("--", ":"))
                        np.testing.assert_allclose(references, expected)
                    app.on_key(SimpleNamespace(key='+'))
                    for index in (2, 3, 4):
                        self.assertAlmostEqual(app.axes[index].get_xlim()[1],
                                               app.state.n / 8)
                finally:
                    plt.close(app.fig)

    def test_leakage(self):
        a = np.abs(compute(State(signal=2))['spectrum'])
        b = np.abs(compute(State(signal=7))['spectrum'])
        self.assertLess(np.count_nonzero(a > 1e-9), 5)
        self.assertGreater(np.count_nonzero(b > 1e-4), 20)

    def test_main_retains_keyboard_handler_during_show(self):
        # GUI callbacks hold a weak reference to Explorer.on_key.
        # Verify main() retains Explorer across an event-loop collection.
        def simulate_event_loop():
            gc.collect()
            fig = plt.gcf()
            callbacks = fig.canvas.callbacks.callbacks.get('key_press_event', {})
            handlers = [ref() for ref in callbacks.values()]
            self.assertTrue(any(getattr(handler, '__self__', None).__class__ is Explorer
                                for handler in handlers if handler is not None))
            for handler in handlers:
                if getattr(handler, '__self__', None).__class__ is Explorer:
                    handler(SimpleNamespace(key='v'))
                    self.assertEqual(handler.__self__.state.window, 1)
                    break
            plt.close(fig)

        with patch('fourier_filtering_explorer.plt.show', side_effect=simulate_event_loop):
            main()

    def test_no_matplotlib_shortcut_collisions(self):
        # Verify all explorer bindings are reserved, including conflicts
        # introduced by the Matplotlib defaults (f, v, o, l, arrows, r).
        original = {name: list(plt.rcParams[name]) for name in plt.rcParams
                    if name.startswith('keymap.')}
        try:
            plt.rcParams['keymap.fullscreen'] = ['f', 'ctrl+f']
            plt.rcParams['keymap.forward'] = ['right', 'v']
            plt.rcParams['keymap.zoom'] = ['o']
            plt.rcParams['keymap.yscale'] = ['l']
            reserve_explorer_keys()
            for name in original:
                self.assertFalse(set(plt.rcParams[name]) & EXPLORER_KEYS, name)
            self.assertIn('ctrl+f', plt.rcParams['keymap.fullscreen'])
            self.assertIn('ctrl+s', plt.rcParams['keymap.save'])
        finally:
            for name, keys in original.items():
                plt.rcParams[name] = keys

    def test_keyboard(self):
        app = Explorer()
        for key in ('8', 'v', 'v', 'f', 't', 'up', 'down', 'w', 'W', 'left', 'right',
                    'b', 'B', 'o', 'O', 'l', '9', '+', '-', 'r'):
            app.on_key(SimpleNamespace(key=key))
        self.assertEqual(app.state.signal, 0)
        self.assertEqual(app.state.window, 0)
        plt.close(app.fig)


if __name__ == '__main__':
    unittest.main()
