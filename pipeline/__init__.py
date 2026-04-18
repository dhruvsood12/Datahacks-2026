"""
pipeline — San Diego multi-label Species Distribution Model pipeline.

Stages:
    ingest     Stage 1: data loading, sensor matching, species filtering
    sampling   Stage 2: spatial thinning, target-group background sampling
    features   Stage 3: feature engineering, train-only scaler
    model      Stage 4: multi-label MLP with per-species pos_weight
    evaluation Stage 5: blocked spatial cross-validation
    inference  Stage 6: counterfactual warming predictions
"""
