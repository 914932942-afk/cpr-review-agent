"""Serial isolated video jobs. The inference command receives no ground truth."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import CODE, OUTPUT, read, digest, within_code, verify_lock, validate_result, write_new


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    protocol, lock = verify_lock(include_videos=True)
    jobs = read(OUTPUT/'jobs.json')
    mutex = OUTPUT/'runner.lock'
    with mutex.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps({'pid': os.getpid(), 'started_utc': now()}))
    try:
        for index, job in enumerate(jobs, 1):
            folder = OUTPUT/'runs'/job['key']
            status = folder/'attempt.json'
            if status.exists():
                existing = read(status)
                for item in existing.get('artifacts', []):
                    if digest(folder/item['name']) != item['sha256']:
                        raise ValueError('Prior attempt artifact changed: '+job['record_id'])
                print(f'[{index}/26] already recorded {job["record_id"]}: {existing["status"]}', flush=True)
                continue
            if folder.exists():
                raise FileExistsError('Unfinished attempt requires inspection; will not overwrite: '+str(folder))
            folder.mkdir(parents=True)
            if digest(within_code(job['video'])) != job['video_sha256']:
                raise ValueError('Video hash changed')
            started = time.perf_counter()
            result = {'record_id': job['record_id'], 'video_sha256': job['video_sha256'],
                      'started_utc': now(), 'attempt_number': 1, 'ground_truth_supplied_to_inference': False}
            command = [sys.executable, '-X', 'utf8', str(CODE/protocol['entrypoint']), '--video', str(CODE/job['video']),
                       '--output', str(folder/'inference.json'), '--manifest', str(CODE/protocol['manifest'])]
            print(f'[{index}/26] START {job["record_id"]} ({job["expected_frames"]} frames)', flush=True)
            try:
                env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUTF8': '1'}
                with (folder/'process.txt').open('x', encoding='utf-8') as log:
                    process = subprocess.run(command, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=protocol['timeout_seconds_per_record'])
                result['exit_code'] = process.returncode
                if process.returncode != 0:
                    raise RuntimeError('Video subprocess failed; inspect process.txt')
                analysis = read(folder/'inference.json')
                validate_result(job, analysis, lock['manifest_sha256'])
                if not (folder/'inference.npz').is_file():
                    raise ValueError('Missing frozen feature/prediction output')
                result.update({'status': 'completed', 'decoded_frames': analysis['decoded_frames'],
                               'analysis_seconds': analysis['analysis_seconds'], 'cuda_peak_allocated_bytes': analysis['cuda_peak_allocated_bytes']})
            except Exception as error:
                result.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
            result['wall_seconds'] = time.perf_counter()-started
            result['finished_utc'] = now()
            result['artifacts'] = [{'name': p.name, 'sha256': digest(p), 'bytes': p.stat().st_size}
                                   for p in sorted(folder.iterdir()) if p.is_file()]
            write_new(status, result)
            print(f'[{index}/26] {result["status"].upper()} {job["record_id"]} wall={result["wall_seconds"]:.1f}s', flush=True)
        verify_lock(include_videos=True)
        attempts = [read(OUTPUT/'runs'/job['key']/'attempt.json') for job in jobs]
        write_new(OUTPUT/'batch-complete.json', {'finished_utc': now(), 'attempts': len(attempts),
                  'completed': sum(r['status'] == 'completed' for r in attempts), 'failed': sum(r['status'] == 'failed' for r in attempts),
                  'model_and_video_integrity_after_run': True, 'release_approved': False})
        print('BATCH COMPLETE; public labels have not been used for inference', flush=True)
    finally:
        if mutex.is_file() and read(mutex).get('pid') == os.getpid():
            mutex.unlink()


if __name__ == '__main__':
    main()
