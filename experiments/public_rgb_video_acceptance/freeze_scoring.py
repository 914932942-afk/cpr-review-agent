"""Freeze scoring/report contracts while video jobs are still running."""
from datetime import datetime, timezone
from pathlib import Path
from common import CODE, OUTPUT, digest, write_new, verify_lock


def main():
    verify_lock()
    if (OUTPUT/'batch-complete.json').exists() or (OUTPUT/'evaluation.json').exists():
        raise ValueError('Freeze scoring before completion, not after looking at results')
    folder = Path(__file__).parent
    names = ['summarize.py', 'record_acceptance.py', 'test_preflight.py', 'test_results.py', 'freeze_scoring.py']
    write_new(OUTPUT/'scoring-lock.json', {'schema': 'cpr-video-scoring-lock-v1',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'implementation_sources': {(folder/name).relative_to(CODE).as_posix(): digest(folder/name) for name in names},
        'purpose': 'Scoring, failure denominators, metrics, checks and report frozen before batch completion'})
    print('Post-run scoring sources frozen before batch completion', flush=True)


if __name__ == '__main__':
    main()
