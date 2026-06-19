"""
This script is designed to scan through a directory structure containing experiment
results, extract relevant metrics from CSV files, and compile them into summary tables.
The script handles multiple experiments, models, augmentations, and seeds, and produces
both mean tables (aggregated across seeds) and individual seed tables.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

METHOD_ORDER = [
    "at_basic",
    "at_awp_basic",
    "trades_basic",
    "trades_awp_basic",
    "ikl_basic",
    "emff_basic",
    "evat_basic",
    "at_cutout",
    "at_awp_cutout",
    "trades_cutout",
    "trades_awp_cutout",
    "ikl_cutout",
    "emff_cutout",
    "evat_cutout",
    "at_autoaug",
    "at_awp_autoaug",
    "trades_autoaug",
    "trades_awp_autoaug",
    "ikl_autoaug",
    "emff_autoaug",
    "evat_autoaug",
    "at_augmix",
    "at_awp_augmix",
    "trades_augmix",
    "trades_awp_augmix",
    "ikl_augmix",
    "emff_augmix",
    "evat_augmix",
]


def method_sort_key(exp_name: str):
    name = exp_name.lower()
    name = "_".join(name.split("_")[1:])
    name = name.replace("_l2", "")

    try:
        return METHOD_ORDER.index(name)
    except ValueError:
        return 999


def extract_augmentation(experiment_name: str) -> str:
    """
    Simple function to extract augmentation type from experiment name.
    """
    if "augmix" in experiment_name.lower():
        return "Augmix"
    elif "autoaug" in experiment_name.lower():
        return "AutoAug"
    elif "cutout" in experiment_name.lower():
        return "Cutout"
    elif "basic" in experiment_name.lower():
        return "Basic"
    else:
        return "Unknown"


def find_versions(log_dir: Path) -> dict[int, Path]:
    """
    Returns:
        {version_id: metrics.csv path}
    """
    versions = {}

    for d in log_dir.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith("version_"):
            continue

        try:
            v = int(d.name.split("_")[-1])
        except ValueError:
            continue

        metrics_csv = d / "metrics.csv"
        if metrics_csv.exists():
            versions[v] = metrics_csv

    return dict(sorted(versions.items()))


def find_seed_versions(model_dir: Path):
    """
    Returns:
        {
            seed: {version_id: metrics_path}
        }
    """
    seed_versions = {}

    for seed_dir in model_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        if not seed_dir.name.startswith("s"):
            continue

        log_dir = seed_dir / "test_lightning_logs"
        if not log_dir.exists():
            continue

        versions = find_versions(log_dir)
        if versions:
            seed_versions[seed_dir.name] = versions

    return seed_versions


def iterate_experiments(root_dir: Path):
    print(f"Scanning experiments in {root_dir}...")
    for experiment_dir in root_dir.iterdir():
        if not experiment_dir.is_dir():
            continue

        experiment_name = experiment_dir.name
        augmentation = extract_augmentation(experiment_name)

        for model_dir in experiment_dir.iterdir():
            if not model_dir.is_dir():
                continue

            seed_versions = find_seed_versions(model_dir)

            yield {
                "experiment_name": experiment_name,
                "model_name": model_dir.name,
                "augmentation": augmentation,
                "seed_versions": seed_versions,
            }


def read_metrics(path: Path) -> dict:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return next(reader)  # first (and only) row


def dataset_from_exp(exp: str) -> str:
    if "cifar100" in exp.lower():
        return "CIFAR100"
    elif "cifar10" in exp.lower():
        return "CIFAR10"
    return "UNK"


def norm_from_exp(exp: str) -> str:
    if exp.lower().endswith("_l2"):
        return "L2"
    return "Linf"


def rename_experiment(exp_name: str) -> str:
    """
    Maps raw experiment folder names to paper method names.
    """
    name = exp_name.lower()

    if "at_awp" in name:
        return "AT-AWP"
    elif "evat" in name:
        return "EV-AT (Ours)"
    elif "at" in name:
        return "AT"
    elif "trades_awp" in name:
        return "TRADES-AWP"
    elif "trades" in name:
        return "TRADES"
    elif "ikl" in name:
        return "IKL-AT"
    elif "emff" in name:
        return "EMFF-TRADES"

    return exp_name


def main(root_dir: Path, output_csv: Path):

    header = [
        "experiment_name",
        "model_name",
        "augmentation",
        "clean_acc",
        "AA_acc",
        "C_acc",
        "RA_mean",
        "clean_aurc",
        "clean_augrc",
        "clean_auroc",
        "aa_aurc",
        "aa_augrc",
        "aa_auroc",
        "C_aurc",
        "C_augrc",
        "C_auroc",
        "AURC_mean",
        "AUGRC_mean",
        "AUROC_mean",
    ]

    metric_keys = header[3:]

    mean_tables = {}
    seed_tables = {}

    for item in iterate_experiments(root_dir):
        exp_raw = item["experiment_name"]
        exp = rename_experiment(exp_raw)

        model = (
            item["model_name"]
            .replace("wideresnet34_10", "WRN34-10")
            .replace("preactresnet18", "PreActResNet18")
        )
        model = model.replace("_emff", "")

        aug = item["augmentation"]
        seed_versions = item["seed_versions"]
        dataset = dataset_from_exp(exp_raw)
        norm = norm_from_exp(exp_raw)

        print(
            f"Processing {exp}, {model}, {aug} with seeds: {list(seed_versions.keys())}"
        )

        if "ablation" in exp_raw:
            print("  → Skipping ablation experiment")
            continue

        collected = []

        # ---- iterate over seeds ----
        for seed in ["s0", "s1", "s2"]:
            versions = seed_versions.get(seed, {})

            row_seed = dict.fromkeys(header, "")
            row_seed["experiment_name"] = exp
            row_seed["_exp_raw"] = exp_raw
            row_seed["model_name"] = model
            row_seed["augmentation"] = aug

            valid_seed_metrics = {}

            try:
                if "L2" in exp_raw:
                    if 0 not in versions or 3 not in versions:
                        raise ValueError("missing required versions")
                    best = read_metrics(versions[0])
                    corr = read_metrics(versions[3])
                else:
                    if 0 not in versions or 7 not in versions:
                        raise ValueError("missing required versions")
                    best = read_metrics(versions[0])
                    corr = read_metrics(versions[7])

                valid_seed_metrics = {
                    "clean_acc": float(best["nat_accuracy"]),
                    "AA_acc": float(best["AutoAttackWrapper_accuracy"]),
                    "C_acc": float(corr["nat_accuracy"]),
                    "RA_mean": (
                        float(best["nat_accuracy"])
                        + float(best["AutoAttackWrapper_accuracy"])
                    )
                    / 2,
                    "clean_aurc": float(best["nat_aurc"]),
                    "clean_augrc": float(best["nat_augrc"]),
                    "clean_auroc": float(best["nat_uncert_auroc"]),
                    "aa_aurc": float(best["AutoAttackWrapper_aurc"]),
                    "aa_augrc": float(best["AutoAttackWrapper_augrc"]),
                    "aa_auroc": float(best["AutoAttackWrapper_uncert_auroc"]),
                    "C_aurc": float(corr["nat_aurc"]),
                    "C_augrc": float(corr["nat_augrc"]),
                    "C_auroc": float(corr["nat_uncert_auroc"]),
                    "AURC_mean": (
                        float(best["nat_aurc"]) + float(best["AutoAttackWrapper_aurc"])
                    )
                    / 2,
                    "AUGRC_mean": (
                        float(best["nat_augrc"])
                        + float(best["AutoAttackWrapper_augrc"])
                    )
                    / 2,
                    "AUROC_mean": (
                        float(best["nat_uncert_auroc"])
                        + float(best["AutoAttackWrapper_uncert_auroc"])
                    )
                    / 2,
                }

                # store numeric values for aggregation
                collected.append(valid_seed_metrics)

                # fill per-seed row
                for k, v in valid_seed_metrics.items():
                    row_seed[k] = f"{v * 100:.2f}"

            except Exception:
                # expected case for unfinished seed
                print(f"  Seed {seed} incomplete → kept blank")

            seed_tables.setdefault((dataset, model, norm, seed), []).append(row_seed)

        # ---- mean aggregation ----
        row_mean = dict.fromkeys(header, "")
        row_mean["experiment_name"] = exp
        row_mean["_exp_raw"] = exp_raw
        row_mean["model_name"] = model
        row_mean["augmentation"] = aug

        if collected:
            for k in metric_keys:
                vals = [d[k] for d in collected if k in d]
                if not vals:
                    continue

                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row_mean[k] = f"{mean * 100:.2f} ± {std * 100:.2f}"

        mean_tables.setdefault((dataset, model, norm), []).append(row_mean)

    # ---- write mean tables ----
    for (dataset, model, norm), rows in mean_tables.items():
        out = output_csv.parent / f"{dataset}_{model}_{norm}_mean.csv"

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            rows = sorted(rows, key=lambda r: method_sort_key(r["_exp_raw"]))
            for r in rows:
                r.pop("_exp_raw", None)

            writer.writerows(rows)

        print(f"Saved mean table → {out}")

    # ---- write seed tables ----
    for (dataset, model, norm, seed), rows in seed_tables.items():
        out = output_csv.parent / f"{dataset}_{model}_{norm}_{seed}.csv"

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            rows = sorted(rows, key=lambda r: method_sort_key(r["_exp_raw"]))
            for r in rows:
                r.pop("_exp_raw", None)

            writer.writerows(rows)

        print(f"Saved seed table → {out}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("summary_results.csv"))
    args = parser.parse_args()

    main(args.root_dir, args.out)
