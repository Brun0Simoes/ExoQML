import numpy as np

from exoqml.training.max_train import TCERow, select_hard_negative_indices, summarize_tce_rows
from exoqml.training.model import TransitMultiViewNet
from exoqml.transit_features import (
    GLOBAL_VIEW_BINS,
    LOCAL_VIEW_BINS,
    SCALAR_FEATURE_DIM,
    build_scalar_features,
    build_tce_views,
    project_folded_relevance_to_time,
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


def test_summarize_tce_rows_counts_mixed_label_stars_once() -> None:
    rows = [
        TCERow(
            tce_id="100_1",
            star_id="100",
            label=1,
            label_name="PC",
            split="train",
            period=10.0,
            duration_hours=4.0,
            epoch=1.0,
            depth_ppm=1500.0,
            model_snr=12.0,
        ),
        TCERow(
            tce_id="100_2",
            star_id="100",
            label=0,
            label_name="AFP",
            split="train",
            period=20.0,
            duration_hours=6.0,
            epoch=2.0,
            depth_ppm=200.0,
            model_snr=5.0,
        ),
        TCERow(
            tce_id="200_1",
            star_id="200",
            label=0,
            label_name="NTP",
            split="test",
            period=30.0,
            duration_hours=3.0,
            epoch=3.0,
            depth_ppm=100.0,
            model_snr=4.0,
        ),
    ]

    summary = summarize_tce_rows(rows)

    assert summary["stars_total"] == 2
    assert summary["stars_positive"] == 1
    assert summary["stars_negative"] == 1


def test_select_hard_negative_indices_prefers_high_score_negatives() -> None:
    rows = [
        TCERow("1_1", "1", 1, "PC", "train", 10.0, 4.0, 1.0, 1000.0, 12.0),
        TCERow("2_1", "2", 0, "AFP", "train", 10.0, 4.0, 1.0, 100.0, 2.0),
        TCERow("3_1", "3", 0, "NTP", "train", 10.0, 4.0, 1.0, 100.0, 2.0),
        TCERow("4_1", "4", 0, "AFP", "train", 10.0, 4.0, 1.0, 100.0, 2.0),
        TCERow("5_1", "5", 0, "NTP", "train", 10.0, 4.0, 1.0, 100.0, 2.0),
    ]
    scores = np.array([0.95, 0.92, 0.88, 0.30, 0.10], dtype=np.float32)

    selected = select_hard_negative_indices(
        rows=rows,
        scores=scores,
        min_score=0.8,
        top_fraction=0.25,
        min_count=1,
        max_count=2,
    )

    assert selected == [1, 2]


def test_project_folded_relevance_to_time_highlights_transit_region() -> None:
    time = np.linspace(0.0, 20.0, 4096, dtype=np.float64)
    flux = np.ones_like(time)
    in_transit = (time % 5.0) < 0.12
    flux[in_transit] -= 0.015

    global_relevance = np.zeros(GLOBAL_VIEW_BINS, dtype=np.float32)
    local_relevance = np.zeros(LOCAL_VIEW_BINS, dtype=np.float32)
    global_relevance[GLOBAL_VIEW_BINS // 2] = 1.0
    local_relevance[LOCAL_VIEW_BINS // 2] = 1.0

    projected = project_folded_relevance_to_time(
        time=time,
        flux=flux,
        period=5.0,
        epoch=0.0,
        duration_hours=2.5,
        global_relevance=global_relevance,
        local_relevance=local_relevance,
    )

    assert projected.shape == flux.shape
    assert np.isfinite(projected).all()
    assert float(projected[in_transit].mean()) > float(projected[~in_transit].mean())
