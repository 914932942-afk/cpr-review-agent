from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import compact_head as c
import optimize_dense as d


class CompactContracts(unittest.TestCase):
    def test_linear_probabilities_and_bad_feature_contract(self):
        component={'kind':'linear','mixture_weight':1.,'mean':torch.zeros(783),'scale':torch.ones(783),'weight1':torch.ones((12,783))*.001,'bias1':torch.zeros(12)}
        model=c.CompactHead({'components':[component]})
        p=model.probability(np.ones((3,783)))
        np.testing.assert_allclose(p,1/(1+np.exp(-.783)),atol=1e-7)
        for bad in [np.ones((1,512)),np.full((1,783),np.nan)]:
            with self.assertRaises(ValueError):model.probability(bad)

    def test_mlp_export_relu_and_mixture_are_numerically_consistent(self):
        torch.manual_seed(1);a=torch.nn.Linear(783,32);b=torch.nn.Linear(32,12)
        component={'kind':'mlp','mixture_weight':1.,'mean':torch.zeros(783),'scale':torch.ones(783),'weight1':a.weight.detach(),'bias1':a.bias.detach(),'weight2':b.weight.detach(),'bias2':b.bias.detach()}
        x=torch.randn(8,783)
        with torch.inference_mode():expected=b(a(x).relu()).sigmoid().numpy()
        np.testing.assert_allclose(c.CompactHead({'components':[component]}).probability(x.numpy()),expected,atol=1e-6)
        self.assertLess(c.count_fields([component]),70000)

    def test_wrong_profile_protocol_and_oversized_head_are_rejected(self):
        state={'schema':'cpr-compact-head-v1','label_ids':list(c.LABEL_IDS),'source_revision':c.REVISION,'feature_profile_hash':'profile','protocol_sha256':'protocol','components':[{'kind':'linear','weights':torch.zeros(70001)}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head.pt';torch.save(state,path)
            for profile,protocol in [('wrong','protocol'),('profile','wrong'),('profile','protocol')]:
                with self.assertRaises(ValueError):c.load(path,profile,protocol)

    def test_fixed_data_reservations_and_gates_are_preserved(self):
        p=d.o.checked(d.BASE)
        self.assertEqual(p['split_counts'],{'train':738,'validation':196,'test':386})
        self.assertEqual(len(p['raw_test_reservations']),26)
        self.assertEqual(p['gates']['composite_macro_f1'],.8)

    def test_macro_f1_does_not_discard_false_positives(self):
        y=np.eye(11,dtype=bool);self.assertEqual(d.macro_f1(y,y),1.)
        self.assertLess(d.macro_f1(np.ones_like(y),y),.2)


if __name__=='__main__':unittest.main(verbosity=2)
