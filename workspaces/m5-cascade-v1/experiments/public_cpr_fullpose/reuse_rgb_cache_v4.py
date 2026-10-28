"""Byte-identical replay of already label-free RGB features under unchanged runtime."""
import json
import shutil
from rgb_features_v3 import CODE, digest
from prepare_features import write_new


def main():
    data=CODE/'data/public_cpr'; original=data/'rgb-domain-features-v3'; target=data/'rgb-domain-features-v4'
    old=json.loads((original/'profile.json').read_text(encoding='utf-8'))
    new=json.loads((target/'profile.json').read_text(encoding='utf-8'))
    if old['profile']!=new['profile'] or old['model_source_sha256']!=new['model_source_sha256']:
        raise ValueError('RGB transform or backbone changed, fresh raw extraction required')
    runtime='experiments/public_cpr_fullpose/rgb_features_v3.py'
    if old['implementation_sources'][runtime]!=new['implementation_sources'][runtime] or digest(CODE/runtime)!=old['implementation_sources'][runtime]:
        raise ValueError('RGB runtime source changed')
    manifest=json.loads((original/'raw26/manifest.json').read_text(encoding='utf-8'))
    if manifest['labels_read'] or manifest['failures'] or len(manifest['records'])!=26:
        raise ValueError('Original label-free raw cache incomplete')
    folder=target/'raw26'
    if folder.exists(): raise FileExistsError('Preserve RGB cache replay')
    folder.mkdir()
    write_new(folder/'replay-plan.json',{'same_frozen_runtime':True,'source_manifest_sha256':digest(original/'raw26/manifest.json'),
        'new_development_profile_sha256':digest(target/'profile.json'),'labels_read':False,'source_runtime_sha256':digest(CODE/runtime)})
    for row in manifest['records']:
        source=original/'raw26'/row['feature_file']
        if digest(source)!=row['feature_sha256']: raise ValueError('Original raw features changed')
        dest=folder/row['feature_file']; shutil.copyfile(source,dest)
        if digest(dest)!=row['feature_sha256']: raise ValueError('Copy integrity mismatch')
    manifest['cache_reuse']='Byte-identical unchanged label-free perception; body/RGB processing was already executed from actual26 MP4 files'
    write_new(folder/'manifest.json',manifest)
    print('Replayed26 label-free RGB caches with unchanged backbone and transform',flush=True)


if __name__=='__main__': main()
