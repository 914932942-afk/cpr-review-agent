import unittest
import numpy as np
from mechanics import describe, DIMENSION


class MechanicsTests(unittest.TestCase):
    def fixture(self):
        p=np.tile(np.arange(34).reshape(17,2),(200,1,1)).astype(float)
        p[:,5:7]=[[0,0],[5,0]]; p[:,11:13]=[[0,10],[5,10]]
        return p,np.ones((200,17))

    def test_translation_scale_invariance_and_shape(self):
        p,q=self.fixture(); a=describe(p,q); b=describe(p*30+100,q)
        self.assertEqual(a.shape,(DIMENSION,)); np.testing.assert_allclose(a,b,atol=1e-5)
        self.assertTrue(np.isfinite(a).all())

    def test_preserves_global_bounce_lost_by_framewise_root_subtraction(self):
        p,q=self.fixture(); a=describe(p,q)
        p[:,:,1]+=np.sin(np.arange(200)/5)[:,None]*3
        b=describe(p,q)
        self.assertGreater(np.linalg.norm(b[24:92]-a[24:92]),.1)

    def test_unobservable_does_not_invent_movement(self):
        p,q=self.fixture(); q[:]=0
        np.testing.assert_array_equal(describe(p,q),np.zeros(DIMENSION))


if __name__=='__main__': unittest.main()
