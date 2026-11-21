"""Реестр пайплайнов: проверяем, что все имена резолвятся и группы непустые."""

from __future__ import annotations

from deepfakes.pipelines.registry import (
    REGISTRY,
    all_groups,
    get_pipeline_class,
    pipelines_by_group,
)


def test_registry_has_all_expected():
    must_have = {
        "col1",
        "col2",
        "col3",
        "freq2",
        "freq3",
        "freq4",
        "grad1",
        "grad3",
        "grad4",
        "tex2",
        "tex4",
        "tex6",
        "noise1",
        "noise2",
        "noise3",
        "ml6_hybrid",
        "ml7_freq_lbp",
        "ml8_anomaly",
        "rob1_fusion",
    }
    missing = must_have - set(REGISTRY)
    assert not missing, f"Missing pipelines in registry: {missing}"


def test_all_classes_resolvable():
    for name in REGISTRY:
        cls = get_pipeline_class(name)
        assert hasattr(cls, "from_cfg"), f"{name} has no from_cfg"


def test_groups_non_empty():
    groups = all_groups()
    for g in ("color", "freq", "grad", "texture", "noise", "ml", "rob"):
        assert g in groups, f"Group '{g}' is missing from registry"
        assert pipelines_by_group(g), f"Group '{g}' has no pipelines"


def test_color_group_contains_three():
    assert set(pipelines_by_group("color")) == {"col1", "col2", "col3"}


def test_freq_group_contains_three():
    assert set(pipelines_by_group("freq")) == {"freq1", "freq2", "freq3", "freq4", "freq5"}


def test_grad_group_contains_three():
    assert set(pipelines_by_group("grad")) == {"grad1", "grad2", "grad3", "grad4"}


def test_texture_group_contains_three():
    assert set(pipelines_by_group("texture")) == {"tex1", "tex2", "tex3", "tex4", "tex5", "tex6"}


def test_noise_group_contains_three():
    assert set(pipelines_by_group("noise")) == {"noise1", "noise2", "noise3", "ela"}
