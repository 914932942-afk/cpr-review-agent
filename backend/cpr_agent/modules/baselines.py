"""Adapters around existing behavior, not four new research algorithms."""

from .contracts import Assessment, Decision, MLResult, Prediction


class BaselineCalibration:
    def assess(self, request):
        predictions = {label: p for p in request.predictions for label in p.supported_label_ids}
        states = {
            label: (
                "not_evaluated"
                if label not in predictions
                else "model_flagged"
                if label in predictions[label].flagged_label_ids
                else "not_flagged"
            )
            for label in request.target_label_ids
        }
        return Assessment(
            states=states,
            source_prediction_ids=[p.prediction_id for p in request.predictions],
            calibration_profile="baseline-uncalibrated-v1",
            calibration_status="uncalibrated",
        )


class BaselineScheduling:
    def choose(self, observation):
        if "run_full_pipeline" in observation.available_actions:
            return Decision(
                action="run_full_pipeline",
                profile_id="full-v15",
                reason="Fixed baseline full pipeline; no gain estimator is connected.",
            )
        return Decision(action="STOP", reason="Use existing results or stop after completed/failed analysis.")


def prediction_from_arrays(request, arrays, labels, decision_key="eligible"):
    return Prediction(
        prediction_id=request.execution_id,
        binding=request.binding,
        supported_label_ids=list(labels),
        flagged_label_ids=[int(k) for k, yes in zip(labels, arrays[decision_key][0]) if yes],
        score_type="uncalibrated_score",
        scores={int(k): float(arrays["scores"][0, i]) for i, k in enumerate(labels)},
        thresholds={int(k): float(arrays["thresholds"][0, i]) for i, k in enumerate(labels)},
    )


class BaselineRGB:
    def predict(self, request):
        import numpy as np
        import torch
        from pipeline import frozen, rgb_readout

        if request.sampling_profile != "uniform16-v15":
            raise ValueError("RGB_PROFILE_NOT_SUPPORTED")
        indices = frozen.sample_indices(len(request.points))
        frames = frozen.read_video_samples(request.video_path, len(request.points))
        runtime = frozen.RGBRuntime()
        rgb, audit = runtime.extract(frames, request.points[indices], request.confidence[indices])
        del runtime
        torch.cuda.empty_cache()
        shape, hand_audit = frozen.shape_features(
            frames, request.points[indices], request.confidence[indices]
        )
        arrays = rgb_readout(rgb[None], shape[None], np.asarray([hand_audit["available"]]))
        arrays.update(rgb_features=rgb[None], shape_features=shape[None])
        prediction = prediction_from_arrays(request, arrays, arrays["label_ids"].tolist())
        return MLResult(
            prediction,
            arrays,
            {
                "rgb_audit": audit,
                "hand_shape_audit": hand_audit,
                "selected_frame_indices": indices.tolist(),
                "sampling_profile": request.sampling_profile,
            },
        )


class BaselineFusion:
    def fuse(self, request):
        from pipeline import LABELS, merge_readouts

        arrays, reliability = merge_readouts(request.skeleton.arrays, request.rgb.arrays)
        return MLResult(
            prediction_from_arrays(request, arrays, LABELS, "predictions"),
            arrays,
            {"reliability_sources": reliability},
        )
