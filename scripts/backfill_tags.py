"""Repair dataset and pipeline tags in older MLflow tracking stores."""

from __future__ import annotations

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from deepfakes.utils.env import load_project_dotenv

load_project_dotenv()

#: эксперимент → значение data для ВСЕХ его неразмеченных прогонов
FIXED_RULES: dict[str, str] = {
    "deepfake_source_bias_neg_ctrl": "gan_only",
    "deepfake_source_bias_robust": "gan_only",
    "deepfake_source_bias_tune": "gan_only",
    "deepfake_source_bias_interpret": "gan_only",
}

#: для cross: подстрока в имени метрики → протокол. Подстроки — имена
#: источников из configs/data/cross_*.yaml, пересекаться они не могут.
CROSS_RULES: dict[str, str] = {
    "celeba_real_ood": "cross_src_check",
    "wiki_real_norm": "cross_gan_dm",
}


def infer_cross_data(metric_names: list[str]) -> str | None:
    """Протокол cross-прогона по именам его метрик; None, если неоднозначно."""
    hits = {proto for key, proto in CROSS_RULES.items() if any(key in m for m in metric_names)}
    return hits.pop() if len(hits) == 1 else None


def plan(client: MlflowClient, experiment_prefix: str = "deepfake_source_bias") -> list[tuple]:
    """Список (эксперимент, run_id, run_name, новое значение или None)."""
    out: list[tuple] = []
    for exp in client.search_experiments():
        if not exp.name.startswith(experiment_prefix + "_"):
            continue
        for run in client.search_runs([exp.experiment_id], max_results=5000):
            if run.data.tags.get("data"):
                continue  # уже размечен — не трогаем
            name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
            if exp.name in FIXED_RULES:
                value = FIXED_RULES[exp.name]
            elif exp.name == f"{experiment_prefix}_cross":
                value = infer_cross_data(list(run.data.metrics))
            else:
                value = None
            out.append((exp.name, run.info.run_id, name, value))
    return out


def is_deep_variant(params: dict) -> bool:
    """Прогон ViT-гибрида: в составе компонентов есть kind=deep.

    Такие прогоны до 28.09.2026 логировались с `pipeline=rob1_fusion` (именем
    класса) и в отчёте подменяли классический rob1_fusion."""
    comps = str(params.get("pipeline.components", ""))
    return "'kind': 'deep'" in comps or '"kind": "deep"' in comps


def plan_variant_retag(client: MlflowClient, experiment: str = "deepfake_source_bias") -> list[tuple]:
    """(run_id, run_name, статус) прогонов deep-варианта под тегом rob1_fusion."""
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    runs = client.search_runs(
        [exp.experiment_id], "tags.pipeline = 'rob1_fusion'", max_results=5000
    )
    return [
        (r.info.run_id, r.data.tags.get("mlflow.runName", ""), r.info.status)
        for r in runs
        if is_deep_variant(r.data.params)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="записать теги (иначе сухой прогон)")
    ap.add_argument("--tracking-uri", default=None)
    args = ap.parse_args()
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # --- 1. deep-вариант, записанный под именем класса -----------------------
    deep = plan_variant_retag(client)
    if deep:
        print("Переразметка pipeline: rob1_fusion → rob_deep_fusion (по params.components):")
        for run_id, name, status in deep:
            print(f"  deepfake_source_bias  {name:48} [{status}]")
        if args.apply:
            for run_id, _, _ in deep:
                client.set_tag(run_id, "pipeline", "rob_deep_fusion")
                client.set_tag(run_id, "pipeline_class", "rob1_fusion")
                client.set_tag(run_id, "pipeline_backfilled", "2026-09-28")
            print(f"  записано: {len(deep)}\n")
        else:
            print()

    # --- 2. тег data там, где его не было ------------------------------------
    todo = plan(client)
    if not todo:
        print("Неразмеченных по data прогонов нет.")
        return
    unresolved = [t for t in todo if t[3] is None]
    for exp, run_id, name, value in todo:
        mark = "→ " + value if value else "?? НЕ ОПРЕДЕЛЁН, оставлен без тега"
        print(f"  {exp:18} {name:48} {mark}")
    print(f"\nИтого: {len(todo) - len(unresolved)} к разметке, {len(unresolved)} неопределённых.")
    if not args.apply:
        print("Сухой прогон: ничего не записано. Для записи — --apply.")
        return
    n = 0
    for _, run_id, _, value in todo:
        if value:
            client.set_tag(run_id, "data", value)
            client.set_tag(run_id, "data_backfilled", "2026-09-28")
            n += 1
    print(f"Записано тегов: {n}. У каждого есть пометка data_backfilled=2026-09-28.")


if __name__ == "__main__":
    main()
