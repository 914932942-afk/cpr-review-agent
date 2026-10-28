"""Atomic offline-candidate selection after quality, cold replay, tests and report QA."""
from datetime import datetime,timezone
import json
import os
from head import CODE
from train_head import o


def main():
    data=CODE/'data/public_cpr';root=data/'rgb-evaluation-v15';report=CODE/'test-results/public-cpr-fullpose-v1'
    record=o.parent.read(root/'acceptance-record.json');qa=o.parent.read(report/'final-report-qa.json')
    if not record['raw_video_quality_passed'] or not record['tests']['passed'] or not record['cold_replay']['passed'] or not qa['passed']:
        raise ValueError('Every acceptance gate and report QA must pass')
    manifest=root/'bundle-manifest-v3.json'
    if o.parent.digest(manifest)!=record['manifest_sha256']:raise ValueError('Accepted runnable bundle changed')
    bundle=o.parent.read(manifest);o.parent.check_hashes(bundle['pinned_sources_and_assets'])
    pointer=data/'current-offline-candidate.json'
    if o.parent.digest(pointer)!=record['pointer_before_sha256']:raise ValueError('Concurrent candidate update; preserve current pointer')
    value={'schema':'cpr-offline-candidate-v2','candidate_id':'single-view-cpr-v15',
        'purpose':'Accepted offline actual-video candidate; no Web/Agent or production release',
        'protocol':'data/public_cpr/rgb-evaluation-v15/protocol.json',
        'manifest':'data/public_cpr/rgb-evaluation-v15/bundle-manifest-v3.json',
        'entrypoint':'experiments/public_cpr_fullpose/infer_candidate_v2.py',
        'report':'test-results/public-cpr-fullpose-v1/真实视频优化验收-v15.html',
        'public_keypoint_gates_passed':None,'base_public_keypoint_gates_passed':True,
        'hybrid386_rgb_benchmark_evaluated':False,'composite_raw_video_evaluated':False,
        'raw_video_smoke_passed':True,'full_raw_acceptance_complete':True,'raw_video_acceptance_performed':True,
        'raw_video_acceptance_status':'passed','raw_video_execution_passed':True,'raw_video_quality_passed':True,
        'raw_video_acceptance_protocol':'data/public_cpr/rgb-evaluation-v15/protocol.json',
        'raw_video_acceptance_report':'test-results/public-cpr-fullpose-v1/真实视频优化验收-v15.html',
        'raw_video_acceptance_evaluation':'data/public_cpr/rgb-evaluation-v15/acceptance-record.json',
        'active_scalar_fields_including_fallback':69755,'scope':'Same26-video single-view ch0 engineering regression; not independent blind, cross-person or clinical evidence',
        'runtime_release_approved':False,'selected_utc':datetime.now(timezone.utc).isoformat(),
        'previous_pointer_backup':'data/public_cpr/rgb-evaluation-v15/pointer-before-promotion.json'}
    temporary=pointer.with_name('current-offline-candidate.v15-update.tmp')
    o.write(temporary,value);os.replace(temporary,pointer)
    if o.parent.read(pointer)!=value:raise AssertionError('Candidate selection did not persist')
    o.write(root/'promotion-record.json',{'selected_utc':value['selected_utc'],'pointer_before_sha256':record['pointer_before_sha256'],
        'pointer_after_sha256':o.parent.digest(pointer),'manifest_sha256':record['manifest_sha256'],
        'report_sha256':o.parent.digest(report/'真实视频优化验收-v15.html'),'report_qa_sha256':o.parent.digest(report/'final-report-qa.json'),
        'original_application_source_unchanged':True,'runtime_release_approved':False})
    print('Offline candidate v15 selected. Raw quality/cold replay/contracts passed. Web/Agent release remains disabled.',flush=True)


if __name__=='__main__':main()
