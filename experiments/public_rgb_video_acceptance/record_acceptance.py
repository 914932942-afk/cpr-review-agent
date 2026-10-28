"""Run final contracts and record acceptance metadata, without replacing v4."""
from datetime import datetime, timezone
import io
from pathlib import Path
import unittest

from common import CODE, DATA, OUTPUT, REPORT, read, digest, write_new, verify_lock


def main():
    protocol, lock = verify_lock(include_videos=True)
    result = read(OUTPUT/'evaluation.json')
    scoring_lock = read(OUTPUT/'scoring-lock.json')
    for source, expected in scoring_lock['implementation_sources'].items():
        if digest(CODE/source) != expected:
            raise ValueError('Frozen acceptance scoring source changed: '+source)
    if (OUTPUT/'recorded-acceptance.json').exists():
        raise FileExistsError('Acceptance was already recorded')
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py')
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    summary = {'tests_run': tests.testsRun, 'failures': len(tests.failures), 'errors': len(tests.errors),
               'skipped': len(tests.skipped), 'passed': tests.wasSuccessful()}
    write_new(REPORT/'contract-tests.json', summary)
    (REPORT/'contract-tests.txt').write_text(stream.getvalue(), encoding='utf-8')
    if not summary['passed']:
        raise AssertionError('Acceptance contracts failed; do not change the pointer metadata')
    pointer_path = DATA/'current-offline-candidate.json'
    if digest(pointer_path) != lock['pointer_before_sha256']:
        raise ValueError('Concurrent pointer update; will not overwrite')
    pointer = read(pointer_path)
    status = 'passed' if result['raw_quality_passed'] else ('failed_quality' if result['raw_execution_passed'] else 'failed_execution')
    updated = {**pointer, 'purpose': 'Frozen v4 offline candidate; raw-video acceptance recorded; not production release',
        'raw_video_acceptance_performed': True, 'raw_video_acceptance_status': status,
        'full_raw_acceptance_complete': result['completed_records'] == 26,
        'raw_video_execution_passed': result['raw_execution_passed'],
        'raw_video_quality_passed': result['raw_quality_passed'],
        'raw_video_acceptance_protocol': (OUTPUT/'acceptance-protocol.json').relative_to(CODE).as_posix(),
        'raw_video_acceptance_report': (REPORT/'26段真实视频验收报告.html').relative_to(CODE).as_posix(),
        'raw_video_acceptance_evaluation': (OUTPUT/'evaluation.json').relative_to(CODE).as_posix(),
        'runtime_release_approved': False}
    for key in ['manifest', 'protocol', 'entrypoint']:
        if updated[key] != pointer[key]:
            raise AssertionError('Recording acceptance must not replace the model or entrypoint')
    temporary = pointer_path.with_name('current-offline-candidate.acceptance-update.tmp')
    write_new(temporary, updated)
    temporary.replace(pointer_path)
    final = read(pointer_path)
    if final != updated:
        raise AssertionError('Pointer persistence failed')
    integrity = {'passed': True, 'protected_files_checked': len(lock['protected_files']),
        'video_hashes_checked': 26, 'model_thresholds_and_inputs_unchanged': True,
        'original_application_source_files_unchanged': 19, 'candidate_model_and_entrypoint_unchanged': True,
        'only_pointer_acceptance_metadata_updated': True, 'runtime_release_approved': False}
    write_new(REPORT/'source-integrity.json', integrity)
    record = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'tests': summary, 'integrity': integrity,
        'completed_records': result['completed_records'], 'failed_records': result['failed_records'],
        'raw_execution_passed': result['raw_execution_passed'], 'raw_quality_passed': result['raw_quality_passed'],
        'evaluation_sha256': digest(OUTPUT/'evaluation.json'), 'pointer_after_sha256': digest(pointer_path),
        'pointer_before_backup': (OUTPUT/'candidate-pointer-before.json').relative_to(CODE).as_posix(),
        'runtime_release_approved': False}
    write_new(OUTPUT/'recorded-acceptance.json', record)
    message = ('v4固化及26段真实视频验收已完成。\n'
        f'完成分类：{result["completed_records"]}/26；处理失败：{result["failed_records"]}；'
        f'完整标签匹配：{result["raw_exact_matches"]}/26。\n'
        f'执行链路：{"通过" if result["raw_execution_passed"] else "未通过"}；'
        f'识别质量：{"通过" if result["raw_quality_passed"] else "未通过"}。\n'
        f'{summary["tests_run"]}项技术及结果校验通过。模型、阈值、输入处理、应用源码均未改变。\n'
        '仅更新当前候选的验收状态元数据；仍未批准接入Web/Agent。\n'
        '本批没有组合错误RGB，不是独立盲测，也不代表跨人或临床效果。\n')
    (REPORT/'验收结论.txt').write_text(message, encoding='utf-8')
    print(message, flush=True)


if __name__ == '__main__':
    main()
