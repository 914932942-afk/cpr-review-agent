Isolated v5 optimization: not adopted.

Data: the same public train738 / validation196 / test386 protocol as v4.
Only six balanced logistic heads were fitted. Tree weights and clipping bounds
produce 18 declared configurations plus an exact v4 control. No new labels,
no backbone training, no custom data collection.

Stages (use the existing ML Conda interpreter from CODE):
  python experiments/public_rgb_robust_head/optimize_robust.py prepare
  python experiments/public_rgb_robust_head/optimize_robust.py train
  python experiments/public_rgb_robust_head/optimize_robust.py evaluate
  python experiments/public_rgb_robust_head/optimize_robust.py compare
  python experiments/public_rgb_robust_head/finalize_report.py

Existing artifacts are exclusive-write; do not rerun a stage in its frozen
directory. The source, data, gates and promotion policy were frozen before
fitting. After one validation-selected candidate failed nonregression, no
alternative candidate was selected using the test. Export refuses the failed
candidate and the current-offline-candidate.json pointer remains at v4.

The fixed engineering test is repeatedly exposed; do not claim blind-test
generalization. Public-keypoint results do not establish raw-video accuracy.
No new raw-video inference was run after this candidate was rejected.

Results: test-results/public-rgb-optimization-v5/后续优化验证.html
