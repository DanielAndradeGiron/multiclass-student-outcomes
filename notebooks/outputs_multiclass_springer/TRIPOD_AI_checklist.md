# TRIPOD+AI — where each reporting domain is addressed

Map to the official TRIPOD+AI item numbers (Collins et al., BMJ 2024) before submission.

| Domain | Content | Location |
|---|---|---|
| Title/abstract | Multiclass prediction (Dropout, Enrolled, Graduate); design; ensembles and meta-ensembles | Title, Abstract |
| Source of data | UCI 697 (Realinho et al., 2021, 2022); SHA-256 | metadata/source_manifest.json |
| Participants | Undergraduates, Polytechnic Institute of Portalegre | Table 1 |
| Outcome | Three classes at the end of the normal programme duration | Methods; Table 1 |
| Predictors | 36 predictors; type and time of availability | Table S1 |
| Sample size | n per partition and class | Fig. 1; split_manifest.json |
| Missing data | None in source; imputers inside pipeline | Table S1 |
| Data preparation | Fold-local SMOTENC, one-hot encoding and scaling | Methods |
| Analysis — data use | Locked grouped holdout; repeated nested CV; grouped stacking folds | Fig. 1 |
| Analysis — models | 6 ensembles, voting, stacking, multinomial LR reference; grids | Table S4 |
| Calibration | Temperature scaling (primary); isotonic; Dirichlet; per-class ECE; calibration slope and intercept | Tables 4, 6, S5; Fig. 3 |
| Class imbalance | SMOTENC in training folds; ROS and none (sensitivity) | Table 9 |
| Performance measures | Macro-F1, MCC, κ, AUC, per-class metrics, Brier, log-loss, ECE with CIs | Tables 2, 5–6 |
| Model comparison | Friedman/Iman–Davenport–Nemenyi; Nadeau–Bengio + Holm; Bayesian correlated t test (ROPE) | Table 3; Fig. 2 |
| Decision use | Capacity-constrained prioritisation; expected vs observed; decision curve | Table 7; Table S9; Fig. 4 |
| Fairness/subgroups | Gender, scholarship, age, displaced, debtor | Table S6; Fig. S3 |
| Explainability | SHAP by class; permutation; fidelity-gated LIME; concordance and stability | Table 8; Figs. 5–6; Tables S7–S8 |
| Limitations | Single institution; internal validation; prediction horizon | Discussion |
| Code/data availability | Notebook; versions; checksums; runtime log | metadata/ |