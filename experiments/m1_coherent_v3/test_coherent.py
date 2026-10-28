"""Preservation and repair invariants on generated trajectories, not CPR accuracy claims."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from algorithm import Config, features, repair_bursts, stabilize  # noqa: E402

spec = importlib.util.spec_from_file_location("accepted_fast", ROOT / "experiments/m1_fast_v1/algorithm.py")
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)


def inputs():
    p = np.repeat(np.array([[j * 3.0, j * 6.0] for j in range(17)])[None], 120, axis=0)
    p[:, 11:13, 1] += 80
    p += np.arange(120)[:, None, None] * 0.1
    return p, np.full((120, 17), 0.95)


class CoherentTests(unittest.TestCase):
    def test_common_offset_is_repaired_with_unchanged_confidence(self):
        p, q = inputs()
        bad = p.copy()
        bad[50:54, 9] += [35, 25]
        q[50:54, 9] = 0.25
        a, b, _, audit = stabilize(bad, q)
        np.testing.assert_allclose(a, p, atol=1e-10)
        np.testing.assert_array_equal(b, q)
        self.assertEqual(audit["short_burst_points_corrected"], 4)

    def test_high_confidence_and_weak_smooth_motion_are_not_changed(self):
        p, q = inputs()
        for weak in [False, True]:
            if weak:
                q[50:54, 9] = 0.25
            a, _, _, audit = stabilize(p, q)
            np.testing.assert_array_equal(a, p)
            self.assertEqual(audit["short_burst_points_corrected"], 0)

    def test_curved_weak_motion_is_not_interpreted_as_common_offset(self):
        p, q = inputs()
        p[50:54, 9] += [[30, 0], [0, 30], [-30, 0], [0, -30]]
        q[50:54, 9] = 0.25
        a, changed, _ = repair_bursts(p, q, 100.0, Config())
        np.testing.assert_array_equal(a, p)
        self.assertFalse(changed.any())

    def test_long_weak_interval_is_not_interpolated(self):
        p, q = inputs()
        p[50:56, 9] += [35, 25]
        q[50:56, 9] = 0.25
        a, _, _, audit = stabilize(p, q)
        np.testing.assert_array_equal(a, p)
        self.assertEqual(audit["short_burst_points_corrected"], 0)

    def test_long_missing_interval_remains_unavailable(self):
        p, q = inputs()
        p[50:60, 9], q[50:60, 9] = 0, 0
        a, _, supported, audit = stabilize(p, q)
        self.assertFalse(supported[50:60, 9].any())
        self.assertTrue(np.isnan(a[50:60, 9]).all())
        self.assertFalse(audit["reconstructed_points_are_observations"])

    def test_disabled_addition_is_numerically_equivalent_to_accepted_m1(self):
        p, q = inputs()
        rng = np.random.default_rng(20261027)
        p += rng.normal(0, 3, p.shape)
        q[:] = rng.uniform(0.1, 1, q.shape)
        a, b, _ = features(p, q, Config(burst_enabled=False))
        x, y, _ = old.features(p, q)
        np.testing.assert_array_equal(a, x)
        np.testing.assert_array_equal(b, y)

    def test_inputs_immutable_and_descriptor_layout_unchanged(self):
        p, q = inputs()
        before = p.copy(), q.copy()
        a, b, _ = features(p, q)
        self.assertEqual(a.shape, (271,))
        self.assertEqual(b.shape, (176,))
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())
        np.testing.assert_array_equal(p, before[0])
        np.testing.assert_array_equal(q, before[1])

    def test_invalid_inputs_are_rejected(self):
        p, q = inputs()
        p[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            stabilize(p, q)


if __name__ == "__main__":
    unittest.main(verbosity=2)
