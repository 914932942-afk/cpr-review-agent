"""Compare the new run with the prior v4 smoke; never alter recognition."""
import numpy as np
from common import CODE, OUTPUT, REPORT, read, write_new


def main():
    old = CODE/'test-results/public-rgb-optimization-v4/raw-video-smoke.json'
    new = OUTPUT/'runs/CPR_Dataset_S0--C13_S0--r12--ch0/inference.json'
    before, after = read(old), read(new)
    with np.load(old.with_suffix('.npz'), allow_pickle=False) as a, np.load(new.with_suffix('.npz'), allow_pickle=False) as b:
        checks = {'same_video_bytes': before['video_sha256'] == after['video_sha256'],
            'same_model_manifest': before['manifest_sha256'] == after['manifest_sha256'],
            'same_number_of_frames': before['decoded_frames'] == after['decoded_frames'],
            'same_predicted_error_ids': before['predicted_original_label_ids'] == after['predicted_original_label_ids'],
            'all_frozen_features_bitwise_equal': bool(np.array_equal(a['features'], b['features'])),
            'all_scores_bitwise_equal': bool(np.array_equal(a['scores'], b['scores']))}
        result = {'checks': checks, 'passed': all(checks.values()),
            'feature_max_absolute_difference': float(np.abs(a['features']-b['features']).max()),
            'score_max_absolute_difference': float(np.abs(a['scores']-b['scores']).max()),
            'only_old_and_new_results_compared': True, 'no_inference_rerun_or_threshold_change': True}
    write_new(REPORT/'previous-v4-smoke-replay.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
