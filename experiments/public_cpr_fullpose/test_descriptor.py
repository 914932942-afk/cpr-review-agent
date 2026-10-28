import unittest
import numpy as np
from descriptor import describe, hand_features, spectrum_features, sample_indices


def fixture():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(200, 68, 2))
    points[:, 5] = [0., 0.]; points[:, 6] = [1., 0.]
    points[:, 11] = [0., 2.]; points[:, 12] = [1., 2.]
    return points, np.ones((200, 68))


class DescriptorContracts(unittest.TestCase):
    def test_descriptor_dimension_finiteness_and_uniform_sample_scope(self):
        p, q = fixture()
        x, audit = describe(p, q)
        self.assertEqual(x.shape, (726,))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(sample_indices(200)[[0, -1]], [0, 199])
        self.assertEqual(len(sample_indices(200)), 16)

    def test_hand_features_are_translation_and_scale_invariant(self):
        p, q = fixture(); idx = sample_indices(len(p))
        a, _ = hand_features(p[idx], q[idx])
        b, _ = hand_features(p[idx]*8+np.array([200, -100]), q[idx])
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    def test_missing_hands_do_not_fabricate_finger_shape(self):
        p, q = fixture(); q[:, 26:] = 0.
        result, _ = hand_features(p[sample_indices(200)], q[sample_indices(200)])
        np.testing.assert_array_equal(result[408:], np.zeros(224))

    def test_original_time_spectrum_distinguishes_known_synthetic_rates(self):
        p, q = fixture(); t = np.arange(200)/25
        p[:, 0, 1] = np.sin(2*np.pi*t)
        a = spectrum_features(p[:, :17], q[:, :17])
        p[:, 0, 1] = np.sin(2*np.pi*2*t)
        b = spectrum_features(p[:, :17], q[:, :17])
        self.assertAlmostEqual(a[5], 1., places=2)
        self.assertAlmostEqual(b[5], 2., places=2)

    def test_invalid_layout_and_nonfinite_values_fail_closed(self):
        p, q = fixture()
        with self.assertRaises(ValueError):
            hand_features(p[:, :17], q[:, :17])
        p[0, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            hand_features(p, q)


if __name__ == '__main__':
    unittest.main(verbosity=2)
