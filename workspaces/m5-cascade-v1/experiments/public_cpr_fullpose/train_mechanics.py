"""Public-label-only tilt/motion experts, selected before raw-video regression."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from mechanics import describe, DIMENSION, CODE
from train_head import o
from head import count_fields
from quality_head import export_classifier, PortableHead
from train_rgb_experts_v3 import binary_metrics

DATA = CODE/'data/public_cpr'
ROOT = DATA/'body-mechanics-v2'


def prepare():
    if ROOT.exists(): raise FileExistsError('Preserve mechanics experiment')
    source = DATA/'rgb-evaluation-v4/record-splits.json'
    rows = o.parent.read(source)
    files = [Path(__file__).resolve(), Path(__file__).with_name('mechanics.py')]
    o.write(ROOT/'protocol.json', {'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': o.parent.digest(source), 'implementation_sources': {p.relative_to(CODE).as_posix(): o.parent.digest(p) for p in files},
        'fit': 'Original train738 only, thresholds selected on validation196. Entire test386 including raw26 untouched.',
        'targets': [5, 6], 'features': 'arm/image-axis24; clip-centered global vertical68, hip/knee/ankle16 and interframe motion68',
        'grid': 'balanced logistic C=.003,.03,.3,3; HistGB32 max_leaf7 min_leaf8,16; threshold .1.. .9',
        'selection': 'highest minimum primary/supplemental validation F1, then overall F1, then specificity; no raw26 choices',
        'no_new_labels': True, 'no_backbone_training': True, 'dimension': DIMENSION})
    vectors = []
    for row in rows:
        key = hashlib.sha256(row['record_id'].encode()).hexdigest()
        with np.load(DATA/'fullpose-v2/skeletons'/(key+'.npz'), allow_pickle=False) as f:
            vectors.append(describe(f['keypoint'][0, :, :17], f['keypoint_score'][0, :, :17]))
    np.savez_compressed(ROOT/'features.npz', features=np.stack(vectors),
        targets=np.asarray([[label in r['original_labels'] for label in o.parent.LABEL_IDS] for r in rows], bool),
        splits=np.asarray([r['application_split'] for r in rows]), record_ids=np.asarray([r['record_id'] for r in rows]))
    o.write(ROOT/'features-integrity.json', {'sha256': o.parent.digest(ROOT/'features.npz'), 'records': len(rows)})
    print('Mechanics features prepared', len(rows), flush=True)


def train():
    p = o.parent.read(ROOT/'protocol.json'); o.parent.check_hashes(p['implementation_sources'])
    integrity = o.parent.read(ROOT/'features-integrity.json')
    if o.parent.digest(ROOT/'features.npz') != integrity['sha256']: raise ValueError('Features changed')
    if (ROOT/'experts.pt').exists(): raise FileExistsError('Preserve selected mechanics heads')
    with np.load(ROOT/'features.npz', allow_pickle=False) as f:
        x, y, splits, ids = [f[k].copy() for k in ['features','targets','splits','record_ids']]
    train_mask, val_mask = splits == 'train', splits == 'validation'
    domain = np.asarray([str(k).startswith('Sup') for k in ids])
    weights = np.asarray([1/max(int((domain[train_mask] == v).sum()), 1) for v in domain[train_mask]])
    weights *= len(weights)/weights.sum()
    experts = []; selection = []
    for label in [5, 6]:
        indices = np.arange(24) if label == 5 else np.arange(24, DIMENSION)
        column = list(o.parent.LABEL_IDS).index(label)
        xf, xv = x[train_mask][:, indices], x[val_mask][:, indices]
        yf, yv = y[train_mask, column], y[val_mask, column]
        candidates = []
        mean, scale = xf.mean(0), xf.std(0); scale[scale < 1e-6] = 1
        for strength in [.003, .03, .3, 3.]:
            with threadpool_limits(limits=2):
                model = LogisticRegression(C=strength, class_weight='balanced', solver='liblinear', max_iter=2000).fit(np.clip((xf-mean)/scale,-6,6), yf, sample_weight=weights)
            component = {'kind':'linear','indices':torch.from_numpy(indices), 'mean':torch.from_numpy(mean), 'scale':torch.from_numpy(scale),
                'weight':torch.from_numpy(model.coef_[0]),'bias':float(model.intercept_[0])}
            candidates.append((f'linear{strength}', component, model.predict_proba(np.clip((xv-mean)/scale,-6,6))[:,1]))
        for leaf in [8, 16]:
            with threadpool_limits(limits=2):
                model = HistGradientBoostingClassifier(max_iter=32, max_leaf_nodes=7, min_samples_leaf=leaf,
                    class_weight='balanced', l2_regularization=1, early_stopping=False, random_state=20261021).fit(xf,yf,sample_weight=weights)
            classifier = export_classifier(model)
            candidates.append((f'tree32_leaf{leaf}', {'kind':'tree','indices':torch.from_numpy(indices),'classifier':classifier}, model.predict_proba(xv)[:,1]))
        measured = []
        for name, state, probability in candidates:
            for threshold in np.linspace(.1,.9,33):
                pred = probability >= threshold
                metric = binary_metrics(pred, yv)
                domain_f1 = [binary_metrics(pred[domain[val_mask] == v], yv[domain[val_mask] == v])['f1'] for v in [False, True]]
                measured.append({'name':name,'threshold':float(threshold),'metric':metric,'worst_domain_f1':min(domain_f1)})
        chosen = max(measured, key=lambda r:(r['worst_domain_f1'],r['metric']['f1'],r['metric']['specificity'],-abs(r['threshold']-.5)))
        component = next(c for name,c,probability in candidates if name == chosen['name'])
        experts.append({'label_id':label, 'threshold':chosen['threshold'], **component})
        selection.append({'label_id':label,'selected':chosen,'all_candidates':measured})
        print('Mechanics expert selected',label,chosen,flush=True)
    state = {'experts':experts, 'protocol_sha256':o.parent.digest(ROOT/'protocol.json'), 'scalar_fields':count_fields(experts)}
    torch.save(state, ROOT/'experts.pt')
    o.write(ROOT/'selection.json', {'experts':selection,'sha256':o.parent.digest(ROOT/'experts.pt'), 'scalar_fields':state['scalar_fields'], 'test_used_for_fit':False})


def predict(x, state):
    output = {}
    for e in state['experts']:
        indices = e['indices'].numpy(); values = x[:, indices]
        if e['kind'] == 'tree':
            probability = PortableHead({'input_dimension':len(indices),'classifiers':[e['classifier']]}).probability(values)[:,0]
        else:
            z = np.clip((values-e['mean'].numpy())/e['scale'].numpy(),-6,6)
            probability = 1/(1+np.exp(-np.clip(z @ e['weight'].numpy()+e['bias'],-60,60)))
        output[e['label_id']] = (probability, probability >= e['threshold'])
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','train']); globals()[parser.parse_args().stage]()
