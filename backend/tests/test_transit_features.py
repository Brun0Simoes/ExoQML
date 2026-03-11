import numpy as np

from exoqml.training.model import TransitMultiViewNet
from exoqml.transit_features import (
    GLOBAL_VIEW_BINS,
    LOCAL_VIEW_BINS,
    SCALAR_FEATURE_DIM,
    build_scalar_features,
    build_tce_views,
)


def test_build_tce_views_shape_and_finite() -> None:
    time = np.linspace(0.0, 30.0, 4096, dtype=np.float64)
    flux = np.ones_like(time)
    flux[(time % 5.0) < 0.15] -= 0.01

    global_view, local_view = build_tce_views(
        time=time,
        flux=flux,
        period=5.0,
        epoch=0.0,
        duration_hours=3.0,
    )

    assert global_view.shape == (GLOBAL_VIEW_BINS,)
    assert local_view.shape == (LOCAL_VIEW_BINS,)
    assert np.isfinite(global_view).all()
    assert np.isfinite(local_view).all()
    assert float(np.max(local_view)) > 0.0


def test_build_scalar_features_shape_and_range() -> None:
    features = build_scalar_features(period=10.0, duration_hours=5.0, depth_ppm=1200.0, model_snr=15.0)

    assert features.shape == (SCALAR_FEATURE_DIM,)
    assert np.isfinite(features).all()
    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0)


def test_multiview_model_forward_shape() -> None:
    model = TransitMultiViewNet(scalar_dim=SCALAR_FEATURE_DIM, base_channels=16, dropout=0.1)
    global_view = np.zeros((4, 1, GLOBAL_VIEW_BINS), dtype=np.float32)
    local_view = np.zeros((4, 1, LOCAL_VIEW_BINS), dtype=np.float32)
    scalar = np.zeros((4, SCALAR_FEATURE_DIM), dtype=np.float32)

    output = model(
        global_view=model.global_encoder.net[0].weight.new_tensor(global_view),
        local_view=model.local_encoder.net[0].weight.new_tensor(local_view),
        scalar_features=model.scalar_head[0].weight.new_tensor(scalar),
    )

    assert tuple(output.shape) == (4, 1)
