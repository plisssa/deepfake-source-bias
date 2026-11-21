"""Доразметка тегом data: верные значения, ничего лишнего, существующее не трогаем."""

from __future__ import annotations

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from scripts.backfill_tags import infer_cross_data, plan

# имена метрик — ровно такие, какие run_cross_dataset записал на кластере
SRC_CHECK = "cross_flickr_real_140k-stylegan_140k__to__celeba_real_ood-stylegan_140k_valid_roc_auc"
GAN_DM = (
    "cross_wiki_real_norm_even-stylegan140k_norm__to__wiki_real_norm_odd-dff_text2img_norm_roc_auc"
)


def test_infer_cross_data_from_real_metric_names():
    assert infer_cross_data([SRC_CHECK]) == "cross_src_check"
    assert infer_cross_data([GAN_DM]) == "cross_gan_dm"


def test_infer_cross_data_refuses_ambiguity():
    """Два протокола сразу или ни одного — не угадываем."""
    assert infer_cross_data([SRC_CHECK, GAN_DM]) is None
    assert infer_cross_data(["cross_something_else_roc_auc"]) is None
    assert infer_cross_data([]) is None


@pytest.fixture()
def client(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    yield MlflowClient(tracking_uri=uri)
    mlflow.set_tracking_uri(None)


def _log(exp: str, name: str, tags: dict | None = None, metrics: dict | None = None) -> str:
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name=name) as r:
        for k, v in (tags or {}).items():
            mlflow.set_tag(k, v)
        for k, v in (metrics or {}).items():
            mlflow.log_metric(k, v)
        return r.info.run_id


def test_plan_assigns_known_provenance(client):
    neg = _log("deepfake_source_bias_neg_ctrl", "negctrl__col2__xgboost")
    rob = _log("deepfake_source_bias_robust", "robust__freq4__xgboost")
    x_src = _log("deepfake_source_bias_cross", "cross__col2__xgboost__seed42", metrics={SRC_CHECK: 0.99})
    x_dm = _log("deepfake_source_bias_cross", "cross__col2__xgboost__seed42", metrics={GAN_DM: 0.46})
    x_bad = _log("deepfake_source_bias_cross", "cross__odd", metrics={"cross_weird_roc_auc": 0.5})

    got = {run_id: value for _, run_id, _, value in plan(client)}
    assert got[neg] == "gan_only"
    assert got[rob] == "gan_only"
    assert got[x_src] == "cross_src_check"
    assert got[x_dm] == "cross_gan_dm", "одноимённые cross-прогоны различены по метрикам"
    assert got[x_bad] is None, "неопределённый остаётся без тега"


def test_plan_skips_already_tagged_and_other_experiments(client):
    tagged = _log("deepfake_source_bias_neg_ctrl", "negctrl__tex2__xgboost", tags={"data": "dm_only"})
    bench = _log("deepfake_source_bias", "col1__logreg__holdout__seed42")  # основной — не наш случай
    ids = {run_id for _, run_id, _, _ in plan(client)}
    assert tagged not in ids, "существующий тег перезаписывать нельзя"
    assert bench not in ids, "бенчмарк размечен при логировании, его не трогаем"


def test_apply_writes_tag_and_marker(client, monkeypatch):
    import sys

    from scripts import backfill_tags

    neg = _log("deepfake_source_bias_neg_ctrl", "negctrl__freq4__xgboost")
    monkeypatch.setattr(sys, "argv", ["backfill_tags", "--apply"])
    backfill_tags.main()
    tags = client.get_run(neg).data.tags
    assert tags["data"] == "gan_only"
    assert tags["data_backfilled"] == "2026-09-28", "доразмеченное должно быть отличимо"


def test_dry_run_writes_nothing(client, monkeypatch):
    import sys

    from scripts import backfill_tags

    neg = _log("deepfake_source_bias_neg_ctrl", "negctrl__col2__xgboost")
    monkeypatch.setattr(sys, "argv", ["backfill_tags"])
    backfill_tags.main()
    assert "data" not in client.get_run(neg).data.tags


# --- deep-вариант под именем класса (28.09.2026) --------------------------------

DEEP_COMPONENTS = (
    "[{'kind': 'deep', 'name': 'vit', 'cfg': {'backbone': 'vit_tiny_patch16_224'}}, "
    "{'kind': 'freq', 'name': 'hf', 'cfg': {'n_bins': 8}}]"
)
CLASSIC_COMPONENTS = "[{'kind': 'freq', 'name': 'hf'}, {'kind': 'lbp', 'name': 'lbp'}]"


def test_is_deep_variant_reads_components():
    from scripts.backfill_tags import is_deep_variant

    assert is_deep_variant({"pipeline.components": DEEP_COMPONENTS})
    assert not is_deep_variant({"pipeline.components": CLASSIC_COMPONENTS})
    assert not is_deep_variant({})


def test_deep_run_is_retagged_classic_is_not(client, monkeypatch):
    """Именно такая подмена была в таблице: 0.8979 (ViT) вместо 0.8322 (классика)."""
    import sys

    from scripts import backfill_tags

    mlflow.set_experiment("deepfake_source_bias")
    with mlflow.start_run(run_name="rob1_fusion__xgboost__holdout__seed42") as r:
        mlflow.set_tags({"pipeline": "rob1_fusion", "data": "gan_only"})
        mlflow.log_param("pipeline.components", CLASSIC_COMPONENTS)
        mlflow.log_metric("test_roc_auc", 0.8322)
        classic = r.info.run_id
    with mlflow.start_run(run_name="rob1_fusion__xgboost__holdout__seed42") as r:
        mlflow.set_tags({"pipeline": "rob1_fusion", "data": "gan_only"})
        mlflow.log_param("pipeline.components", DEEP_COMPONENTS)
        mlflow.log_metric("test_roc_auc", 0.8979)
        deep = r.info.run_id

    monkeypatch.setattr(sys, "argv", ["backfill_tags", "--apply"])
    backfill_tags.main()

    assert client.get_run(deep).data.tags["pipeline"] == "rob_deep_fusion"
    assert client.get_run(deep).data.tags["pipeline_backfilled"] == "2026-09-28"
    assert client.get_run(classic).data.tags["pipeline"] == "rob1_fusion", "классику не трогаем"


def test_pipeline_variant_falls_back_to_class_outside_hydra():
    from types import SimpleNamespace

    from deepfakes.tracking.mlflow_setup import pipeline_variant

    cfg = SimpleNamespace(pipeline=SimpleNamespace(name="rob1_fusion"))
    assert pipeline_variant(cfg) == "rob1_fusion"


def test_pipeline_variant_uses_hydra_config_name(tmp_path):
    """Под Hydra тегом должно стать имя конфига, а не имя класса."""
    import os

    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig

    from deepfakes.tracking.mlflow_setup import pipeline_variant

    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name="config", overrides=["pipeline=rob_deep_fusion"], return_hydra_config=True
        )
        HydraConfig.instance().set_config(cfg)
        assert cfg.pipeline.name == "rob1_fusion"  # класс общий
        assert pipeline_variant(cfg) == "rob_deep_fusion"  # а вариант — свой
