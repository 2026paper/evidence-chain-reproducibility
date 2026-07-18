#!/usr/bin/env python3
"""Evidence-bounded bridge between the 18 displayed A/B/C texts and readers.

The API side is reduced to one score vector per displayed text.  The reader
side is first reduced to the corresponding concept-by-version cells.  All
inferential resampling treats the six concepts as blocks; individual reader
answers are never treated as 18 independent texts.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.stats import pearsonr, spearmanr, trim_mean
from statsmodels.stats.multitest import multipletests


SCRIPT_VERSION = "1.0.0"
SEED = 2026071704
N_BOOTSTRAP = 10_000
QUALITY_DIMENSIONS = ["fa", "cc", "lc", "tf", "mq"]
VERSIONS = ["A", "B", "C"]
NONPARALLEL_PREPOST_CONCEPT = "adaptation"

BASE = Path(__file__).resolve().parent
PAPER_ROOT = BASE.parents[1]
ANALYSIS_SOURCE = PAPER_ROOT / "analysis" / "public_abc_cleaned_long.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_api_scores() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mapping = pd.read_csv(BASE / "mapping.csv")
    inputs = pd.read_csv(BASE / "runner_inputs.csv")
    required_mapping = {"item_id", "version", "concept_key", "visible_text_sha256"}
    if not required_mapping.issubset(mapping.columns):
        raise AssertionError(f"Mapping lacks {sorted(required_mapping - set(mapping.columns))}")
    if len(mapping) != 18 or mapping["item_id"].nunique() != 18:
        raise AssertionError("Expected exactly 18 unique displayed-text mappings")
    if set(mapping["version"]) != set(VERSIONS):
        raise AssertionError("Expected A/B/C in the mapping")
    if mapping.groupby("concept_key")["version"].nunique().ne(3).any():
        raise AssertionError("Every concept must have exactly three versions")
    if inputs["item_id"].nunique() != 18 or set(inputs["item_id"]) != set(mapping["item_id"]):
        raise AssertionError("runner_inputs.csv and mapping.csv item identities differ")

    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for path in sorted((BASE / "runs").glob("*/api_judge_scores_long.csv")):
        frame = pd.read_csv(path)
        provider = path.parent.name
        if len(frame) != 18:
            raise AssertionError(f"{provider}: expected 18 rows, observed {len(frame)}")
        if frame["item_id"].nunique() != 18:
            raise AssertionError(f"{provider}: item_id is not unique")
        if set(frame["item_id"]) != set(mapping["item_id"]):
            raise AssertionError(f"{provider}: item set differs from the fixed 18 texts")
        parsed = frame["parse_success"].astype(str).str.lower().isin(["true", "1"])
        if not parsed.all():
            raise AssertionError(f"{provider}: contains parse failures")
        for dimension in QUALITY_DIMENSIONS + ["risk"]:
            values = pd.to_numeric(frame[dimension], errors="coerce")
            if values.isna().any() or (~values.between(1, 5)).any():
                raise AssertionError(f"{provider}: invalid {dimension} scores")
            frame[dimension] = values.astype(int)
        if frame["judge_provider"].nunique() != 1:
            raise AssertionError(f"{provider}: multiple judge_provider labels")
        frames.append(frame)
        sources.append(
            {
                "provider_directory": provider,
                "path": str(path.relative_to(BASE)),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
                "parse_success": int(parsed.sum()),
            }
        )
    if len(frames) != 9:
        raise AssertionError(f"Expected nine providers, observed {len(frames)}")

    scores = pd.concat(frames, ignore_index=True)
    if scores.duplicated(["judge_provider", "item_id"]).any():
        raise AssertionError("Duplicate provider-item API rows")
    scores = scores.merge(mapping, on="item_id", how="left", validate="many_to_one")
    if scores[["version", "concept_key"]].isna().any().any():
        raise AssertionError("Unmapped API rows")
    scores["judge_quality_composite"] = scores[QUALITY_DIMENSIONS].mean(axis=1)

    def aggregate_item(group: pd.DataFrame) -> pd.Series:
        quality = group["judge_quality_composite"].to_numpy(float)
        risk = group["risk"].to_numpy(float)
        result: dict[str, Any] = {
            "n_judges": len(group),
            "api_quality_mean": np.mean(quality),
            "api_quality_median": np.median(quality),
            "api_quality_trimmed_mean": trim_mean(quality, 0.20),
            "api_quality_judge_sd": np.std(quality, ddof=1),
            "api_risk_mean": np.mean(risk),
            "api_risk_median": np.median(risk),
            "api_risk_trimmed_mean": trim_mean(risk, 0.20),
            "api_risk_judge_sd": np.std(risk, ddof=1),
        }
        for dimension in QUALITY_DIMENSIONS + ["risk"]:
            values = group[dimension].to_numpy(float)
            result[f"api_{dimension}_mean"] = np.mean(values)
            result[f"api_{dimension}_median"] = np.median(values)
        return pd.Series(result)

    panel = (
        scores.groupby(["item_id", "version", "concept_key"], observed=True, sort=True)
        .apply(aggregate_item, include_groups=False)
        .reset_index()
    )
    if len(panel) != 18 or not panel["n_judges"].eq(9).all():
        raise AssertionError("The panel reduction is not 18 texts by nine judges")
    return scores, panel, {"api_sources": sources}


def load_reader_cells() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    long = pd.read_csv(ANALYSIS_SOURCE)
    required = {
        "response_id",
        "participant_id",
        "version",
        "concept",
        "phase",
        "correct",
        "confidence_score",
        "misleading_risk_score",
    }
    if not required.issubset(long.columns):
        raise AssertionError(f"Reader data lack {sorted(required - set(long.columns))}")
    if len(long) != 2136:
        raise AssertionError(f"Expected 2,136 reader long rows, observed {len(long)}")
    if long["response_id"].nunique() != 178 or long["participant_id"].nunique() != 167:
        raise AssertionError("Reader response/participant totals differ from the cleaned analysis")
    long["phase"] = pd.to_numeric(long["phase"], errors="raise").astype(int)
    long["correct"] = pd.to_numeric(long["correct"], errors="raise").astype(int)
    pre = long.loc[long["phase"] == 0, ["response_id", "participant_id", "version", "concept", "correct"]].rename(
        columns={"correct": "pre_correct"}
    )
    post = long.loc[
        long["phase"] == 1,
        [
            "response_id",
            "participant_id",
            "version",
            "concept",
            "correct",
            "confidence_score",
            "misleading_risk_score",
        ],
    ].rename(
        columns={
            "correct": "post_correct",
            "confidence_score": "reader_confidence",
            "misleading_risk_score": "reader_perceived_risk",
        }
    )
    wide = pre.merge(
        post,
        on=["response_id", "participant_id", "version", "concept"],
        how="inner",
        validate="one_to_one",
    )
    if len(wide) != 1068:
        raise AssertionError(f"Expected 1,068 paired reader rows, observed {len(wide)}")
    for column in ["reader_confidence", "reader_perceived_risk"]:
        wide[column] = pd.to_numeric(wide[column], errors="raise")
        if (~wide[column].between(1, 5)).any():
            raise AssertionError(f"Reader {column} lies outside 1--5")
    wide["accuracy_change"] = wide["post_correct"] - wide["pre_correct"]
    cells = (
        wide.groupby(["version", "concept"], observed=True, sort=True)
        .agg(
            reader_n_responses=("accuracy_change", "size"),
            reader_n_participant_clusters=("participant_id", "nunique"),
            pre_accuracy=("pre_correct", "mean"),
            post_accuracy=("post_correct", "mean"),
            accuracy_change=("accuracy_change", "mean"),
            reader_confidence_mean=("reader_confidence", "mean"),
            reader_perceived_risk_mean=("reader_perceived_risk", "mean"),
        )
        .reset_index()
        .rename(columns={"concept": "concept_key"})
    )
    expected_n = {"A": 60, "B": 60, "C": 58}
    for version, expected in expected_n.items():
        observed = cells.loc[cells["version"] == version, "reader_n_responses"]
        if len(observed) != 6 or not observed.eq(expected).all():
            raise AssertionError(f"Unexpected reader cell counts for version {version}")
    return wide, cells, {
        "reader_source": {
            "path": str(ANALYSIS_SOURCE.relative_to(PAPER_ROOT)),
            "sha256": sha256_file(ANALYSIS_SOURCE),
            "long_rows": int(len(long)),
            "paired_response_concept_rows": int(len(wide)),
            "responses": int(wide["response_id"].nunique()),
            "participant_clusters": int(wide["participant_id"].nunique()),
        }
    }


def ordered_cells(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    concepts = sorted(frame["concept_key"].unique().tolist())
    expected = pd.MultiIndex.from_product([concepts, VERSIONS], names=["concept_key", "version"])
    indexed = frame.set_index(["concept_key", "version"]).reindex(expected).reset_index()
    if indexed.isna().any().any():
        missing = indexed.loc[indexed.isna().any(axis=1), ["concept_key", "version"]]
        raise AssertionError(f"Missing ordered cells: {missing.to_dict('records')}")
    return indexed, concepts


def reader_cluster_bootstrap(
    wide: pd.DataFrame, ordered: pd.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # Order resampling units by their analysis-visible response content rather
    # than by the participant label. Privacy-preserving rekeying must not alter
    # a fixed-seed bootstrap merely because opaque cluster names changed.
    signature_columns = [
        "version",
        "concept",
        "pre_correct",
        "post_correct",
        "reader_confidence",
        "reader_perceived_risk",
    ]
    signatures: dict[str, str] = {}
    for participant, group in wide.groupby("participant_id", observed=True, sort=False):
        rows = group[signature_columns].sort_values(signature_columns).to_dict("records")
        payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        signatures[str(participant)] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    participants = sorted(
        wide["participant_id"].astype(str).unique().tolist(),
        key=lambda participant: (signatures[participant], participant),
    )
    cell_keys = list(zip(ordered["concept_key"], ordered["version"]))
    cell_lookup = {key: index for index, key in enumerate(cell_keys)}
    participant_lookup = {value: index for index, value in enumerate(participants)}
    outcomes = {
        "pre_accuracy": "pre_correct",
        "post_accuracy": "post_correct",
        "accuracy_change": "accuracy_change",
        "reader_confidence_mean": "reader_confidence",
        "reader_perceived_risk_mean": "reader_perceived_risk",
    }
    counts = np.zeros((len(participants), len(cell_keys)), dtype=float)
    sums = {name: np.zeros_like(counts) for name in outcomes}
    grouped = wide.groupby(["participant_id", "concept", "version"], observed=True, sort=False)
    for (participant, concept, version), group in grouped:
        i = participant_lookup[participant]
        j = cell_lookup[(concept, version)]
        counts[i, j] = len(group)
        for output_name, source_column in outcomes.items():
            sums[output_name][i, j] = group[source_column].astype(float).sum()
    if (counts.sum(axis=0) == 0).any():
        raise AssertionError("Reader bootstrap has an empty cell")

    weights = rng.multinomial(
        len(participants), np.repeat(1.0 / len(participants), len(participants)), size=N_BOOTSTRAP
    ).astype(float)
    denominators = weights @ counts
    if (denominators == 0).any():
        raise AssertionError("A participant-cluster bootstrap draw has an empty cell")
    boot = {name: (weights @ matrix) / denominators for name, matrix in sums.items()}
    return boot, {
        "method": "nonparametric participant-cluster bootstrap, retaining all responses within a participant cluster",
        "seed": SEED,
        "draws": N_BOOTSTRAP,
        "participant_clusters": len(participants),
        "cluster_order": "SHA-256 of analysis-visible response content; invariant to opaque participant rekeying",
    }


def add_reader_bootstrap_intervals(
    cells: pd.DataFrame, reader_boot: dict[str, np.ndarray]
) -> pd.DataFrame:
    result = cells.copy()
    for outcome, samples in reader_boot.items():
        result[f"{outcome}_bootstrap_se"] = np.nanstd(samples, axis=0, ddof=1)
        result[f"{outcome}_ci_low"] = np.nanquantile(samples, 0.025, axis=0)
        result[f"{outcome}_ci_high"] = np.nanquantile(samples, 0.975, axis=0)
    return result


def center_blocks(values: np.ndarray) -> np.ndarray:
    blocks = np.asarray(values, dtype=float).reshape(-1, 3)
    return (blocks - blocks.mean(axis=1, keepdims=True)).reshape(-1)


def rowwise_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def permutation_indices(n_concepts: int) -> np.ndarray:
    local_permutations = np.asarray(list(itertools.permutations(range(3))), dtype=np.int16)
    combinations = np.asarray(
        list(itertools.product(range(len(local_permutations)), repeat=n_concepts)),
        dtype=np.int16,
    )
    indices = np.empty((len(combinations), n_concepts * 3), dtype=np.int16)
    for concept_index in range(n_concepts):
        indices[:, 3 * concept_index : 3 * concept_index + 3] = (
            3 * concept_index + local_permutations[combinations[:, concept_index]]
        )
    return indices


def pairwise_concordance(x: np.ndarray, y: np.ndarray) -> tuple[float, int, int, int]:
    scores: list[float] = []
    concordant = discordant = tied = 0
    for xb, yb in zip(x.reshape(-1, 3), y.reshape(-1, 3)):
        for left, right in [(0, 1), (0, 2), (1, 2)]:
            product = (xb[left] - xb[right]) * (yb[left] - yb[right])
            if product > 0:
                concordant += 1
                scores.append(1.0)
            elif product < 0:
                discordant += 1
                scores.append(0.0)
            else:
                tied += 1
                scores.append(0.5)
    return float(np.mean(scores)), concordant, discordant, tied


def association(
    cells: pd.DataFrame,
    reader_boot: dict[str, np.ndarray],
    predictor: str,
    outcome: str,
    concepts: list[str],
    concept_draws: np.ndarray,
    exact_indices: np.ndarray,
    family: str,
    aggregation: str,
    status: str,
) -> dict[str, Any]:
    all_concepts = cells["concept_key"].drop_duplicates().tolist()
    positions = [
        all_concepts.index(concept) * 3 + version_index
        for concept in concepts
        for version_index in range(3)
    ]
    x = cells[predictor].to_numpy(float)[positions]
    y = cells[outcome].to_numpy(float)[positions]
    x_centered = center_blocks(x)
    y_centered = center_blocks(y)
    if np.std(x_centered) == 0 or np.std(y_centered) == 0:
        raise AssertionError(f"Constant centered variable for {predictor} versus {outcome}")
    observed_r = float(pearsonr(x_centered, y_centered).statistic)
    observed_spearman = float(spearmanr(x_centered, y_centered).statistic)
    slope = float(np.sum(x_centered * y_centered) / np.sum(x_centered**2))

    permuted_x = x_centered[exact_indices]
    fixed_y = np.broadcast_to(y_centered, permuted_x.shape)
    permuted_r = rowwise_correlation(permuted_x, fixed_y)
    exact_p = float(np.mean(np.abs(permuted_r) >= abs(observed_r) - 1e-12))

    outcome_boot = reader_boot[outcome][:, positions]
    outcome_boot_centered = (
        outcome_boot.reshape(N_BOOTSTRAP, len(concepts), 3)
        - outcome_boot.reshape(N_BOOTSTRAP, len(concepts), 3).mean(axis=2, keepdims=True)
    ).reshape(N_BOOTSTRAP, -1)
    gather = (concept_draws[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(
        N_BOOTSTRAP, -1
    )
    x_boot = x_centered[gather]
    y_boot = np.take_along_axis(outcome_boot_centered, gather, axis=1)
    r_boot = rowwise_correlation(x_boot, y_boot)
    slope_boot = np.divide(
        np.sum(x_boot * y_boot, axis=1),
        np.sum(x_boot**2, axis=1),
        out=np.full(N_BOOTSTRAP, np.nan),
        where=np.sum(x_boot**2, axis=1) > 0,
    )
    valid_r = r_boot[np.isfinite(r_boot)]
    valid_slope = slope_boot[np.isfinite(slope_boot)]
    concordance, concordant, discordant, tied = pairwise_concordance(x, y)
    return {
        "family": family,
        "status": status,
        "aggregation": aggregation,
        "predictor": predictor,
        "outcome": outcome,
        "n_text_cells": len(positions),
        "n_concept_blocks": len(concepts),
        "concepts": "|".join(concepts),
        "centered_pearson_r": observed_r,
        "centered_spearman_r": observed_spearman,
        "within_concept_slope": slope,
        "hierarchical_bootstrap_r_ci_low": float(np.quantile(valid_r, 0.025)),
        "hierarchical_bootstrap_r_ci_high": float(np.quantile(valid_r, 0.975)),
        "hierarchical_bootstrap_slope_ci_low": float(np.quantile(valid_slope, 0.025)),
        "hierarchical_bootstrap_slope_ci_high": float(np.quantile(valid_slope, 0.975)),
        "hierarchical_bootstrap_valid_draws": int(len(valid_r)),
        "exact_block_permutation_p_two_sided": exact_p,
        "exact_permutation_space": int(len(exact_indices)),
        "pairwise_concordance_fraction_ties_half": concordance,
        "pairwise_concordant": concordant,
        "pairwise_discordant": discordant,
        "pairwise_tied": tied,
    }


def holm_within_family(rows: pd.DataFrame, family: str) -> None:
    mask = rows["family"].eq(family)
    if not mask.any():
        return
    p = rows.loc[mask, "exact_block_permutation_p_two_sided"].to_numpy(float)
    rows.loc[mask, "p_holm_within_family"] = multipletests(p, method="holm")[1]


def make_report(
    path: Path,
    associations: pd.DataFrame,
    version_summary: pd.DataFrame,
    qa: dict[str, Any],
) -> None:
    primary = associations.loc[associations["family"] == "primary_all_6_concepts"].set_index("predictor")
    quality = primary.loc["api_quality_mean"]
    risk = primary.loc["api_risk_mean"]
    sensitivity = associations.loc[
        associations["family"] == "fixed_sensitivity_excluding_nonparallel_adaptation"
    ].set_index("predictor")

    def fmt(value: float, digits: int = 3) -> str:
        return f"{value:.{digits}f}"

    lines = [
        "# Reader bridge: exact displayed texts scored by the nine-judge API panel",
        "",
        "## Design and estimand",
        "",
        (
            "All nine API judges scored the 18 texts that readers actually saw (six concepts by "
            "three randomized versions). Reader answers were first reduced to concept-by-version "
            "cell estimates. The association analysis therefore has 18 text cells in six concept "
            "blocks; it does not treat 1,068 participant-concept observations as independent texts."
        ),
        "",
        (
            "The analysis reports a fixed two-endpoint primary set: (i) panel mean quality versus paired reader "
            "accuracy change and (ii) panel mean misleading-risk versus paired reader accuracy "
            "change. Two-sided exact tests enumerate all 6^6 = 46,656 independent within-concept "
            "version-label permutations. Confidence intervals use a hierarchical bootstrap that "
            "resamples both participant clusters and concept blocks."
        ),
        "",
        "## Primary results",
        "",
        (
            f"Panel quality was positively associated with reader accuracy change after within-concept "
            f"centering (r={fmt(quality['centered_pearson_r'])}, hierarchical 95% CI "
            f"[{fmt(quality['hierarchical_bootstrap_r_ci_low'])}, "
            f"{fmt(quality['hierarchical_bootstrap_r_ci_high'])}], exact two-sided p="
            f"{fmt(quality['exact_block_permutation_p_two_sided'], 4)}, Holm p="
            f"{fmt(quality['p_holm_within_family'], 4)})."
        ),
        "",
        (
            f"Panel misleading-risk was negatively associated with reader accuracy change "
            f"(r={fmt(risk['centered_pearson_r'])}, hierarchical 95% CI "
            f"[{fmt(risk['hierarchical_bootstrap_r_ci_low'])}, "
            f"{fmt(risk['hierarchical_bootstrap_r_ci_high'])}], exact two-sided p="
            f"{fmt(risk['exact_block_permutation_p_two_sided'], 4)}, Holm p="
            f"{fmt(risk['p_holm_within_family'], 4)})."
        ),
        "",
        "## Fixed sensitivity and descriptive version pattern",
        "",
        (
            "The already-documented content-parallelism sensitivity excludes the adaptation "
            "pre/post pair because its two questions are scientifically related but not equivalent. "
            f"The corresponding quality association was r={fmt(sensitivity.loc['api_quality_mean', 'centered_pearson_r'])} "
            f"(exact p={fmt(sensitivity.loc['api_quality_mean', 'exact_block_permutation_p_two_sided'], 4)}), "
            f"and the risk association was r={fmt(sensitivity.loc['api_risk_mean', 'centered_pearson_r'])} "
            f"(exact p={fmt(sensitivity.loc['api_risk_mean', 'exact_block_permutation_p_two_sided'], 4)})."
        ),
        "",
        "Version-level means are descriptive because there are only three version bundles:",
        "",
        "| Version | API quality | API risk | Pre accuracy | Post accuracy | Accuracy change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in version_summary.iterrows():
        lines.append(
            f"| {row['version']} | {row['api_quality_mean']:.3f} | {row['api_risk_mean']:.3f} | "
            f"{row['pre_accuracy']:.3f} | {row['post_accuracy']:.3f} | {row['accuracy_change']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary of the claim",
            "",
            (
                "This closes the previous text-identity gap: the API panel and readers are now linked "
                "through the same 18 visible texts. It is still a small, purposively selected six-concept "
                "bridge, not a held-out calibration set or evidence of generalization to new concepts. "
                "The absence of a no-exposure retest arm also means accuracy change cannot be interpreted "
                "as a pure causal learning effect."
            ),
            "",
            "## Quality assurance",
            "",
            f"- API rows: {qa['api_rows']} ({qa['providers']} providers x {qa['texts']} texts); parse success: {qa['api_parse_success']}/{qa['api_rows']}.",
            f"- Reader data: {qa['reader_responses']} responses, {qa['reader_participant_clusters']} participant clusters, {qa['reader_paired_rows']} paired participant-concept rows.",
            f"- Analysis cells: {qa['analysis_cells']} texts in {qa['concept_blocks']} concept blocks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    scores, panel, api_meta = load_api_scores()
    wide, reader_cells, reader_meta = load_reader_cells()
    cells = panel.merge(reader_cells, on=["version", "concept_key"], validate="one_to_one")
    cells, concepts = ordered_cells(cells)
    if len(concepts) != 6 or len(cells) != 18:
        raise AssertionError("Bridge analysis requires six concepts by three versions")

    reader_rng = np.random.default_rng(SEED)
    reader_boot, reader_boot_meta = reader_cluster_bootstrap(wide, cells, reader_rng)
    cells = add_reader_bootstrap_intervals(cells, reader_boot)

    concept_draw_rng = np.random.default_rng(SEED + 1)
    concept_draws = {
        6: concept_draw_rng.integers(0, 6, size=(N_BOOTSTRAP, 6)),
        5: concept_draw_rng.integers(0, 5, size=(N_BOOTSTRAP, 5)),
    }
    exact_indices = {6: permutation_indices(6), 5: permutation_indices(5)}
    without_adaptation = [c for c in concepts if c != NONPARALLEL_PREPOST_CONCEPT]
    if len(without_adaptation) != 5:
        raise AssertionError("Fixed adaptation sensitivity did not retain five concepts")

    specifications = [
        # Two fixed primary tests.
        ("primary_all_6_concepts", "api_quality_mean", "accuracy_change", concepts, "mean", "primary"),
        ("primary_all_6_concepts", "api_risk_mean", "accuracy_change", concepts, "mean", "primary"),
        # The already-established content-parallelism sensitivity.
        (
            "fixed_sensitivity_excluding_nonparallel_adaptation",
            "api_quality_mean",
            "accuracy_change",
            without_adaptation,
            "mean",
            "fixed_sensitivity",
        ),
        (
            "fixed_sensitivity_excluding_nonparallel_adaptation",
            "api_risk_mean",
            "accuracy_change",
            without_adaptation,
            "mean",
            "fixed_sensitivity",
        ),
        # Aggregation robustness; not additional confirmatory hypotheses.
        ("aggregation_sensitivity", "api_quality_median", "accuracy_change", concepts, "median", "sensitivity"),
        ("aggregation_sensitivity", "api_risk_median", "accuracy_change", concepts, "median", "sensitivity"),
        (
            "aggregation_sensitivity",
            "api_quality_trimmed_mean",
            "accuracy_change",
            concepts,
            "20pct_trimmed_mean",
            "sensitivity",
        ),
        (
            "aggregation_sensitivity",
            "api_risk_trimmed_mean",
            "accuracy_change",
            concepts,
            "20pct_trimmed_mean",
            "sensitivity",
        ),
        # Complete fixed secondary set; all are reported.
        ("secondary_all_6_concepts", "api_quality_mean", "post_accuracy", concepts, "mean", "secondary"),
        ("secondary_all_6_concepts", "api_risk_mean", "post_accuracy", concepts, "mean", "secondary"),
        (
            "secondary_all_6_concepts",
            "api_quality_mean",
            "reader_confidence_mean",
            concepts,
            "mean",
            "secondary",
        ),
        (
            "secondary_all_6_concepts",
            "api_risk_mean",
            "reader_perceived_risk_mean",
            concepts,
            "mean",
            "secondary",
        ),
    ]
    rows = []
    for family, predictor, outcome, included, aggregation, status in specifications:
        k = len(included)
        rows.append(
            association(
                cells,
                reader_boot,
                predictor,
                outcome,
                included,
                concept_draws[k],
                exact_indices[k],
                family,
                aggregation,
                status,
            )
        )
    associations = pd.DataFrame(rows)
    associations["p_holm_within_family"] = np.nan
    holm_within_family(associations, "primary_all_6_concepts")
    holm_within_family(associations, "secondary_all_6_concepts")

    version_summary = (
        cells.groupby("version", observed=True, sort=True)
        .agg(
            concepts=("concept_key", "nunique"),
            reader_n_responses=("reader_n_responses", "first"),
            api_quality_mean=("api_quality_mean", "mean"),
            api_risk_mean=("api_risk_mean", "mean"),
            pre_accuracy=("pre_accuracy", "mean"),
            post_accuracy=("post_accuracy", "mean"),
            accuracy_change=("accuracy_change", "mean"),
            reader_confidence_mean=("reader_confidence_mean", "mean"),
            reader_perceived_risk_mean=("reader_perceived_risk_mean", "mean"),
        )
        .reset_index()
    )

    merged_columns = [
        "item_id",
        "judge_provider",
        "paper_model_label",
        "judge_model_requested",
        "actual_api_model",
        "judge_model_returned",
        "judge_model",
        *QUALITY_DIMENSIONS,
        "risk",
        "judge_quality_composite",
        "parse_success",
        "retry_count",
        "http_status",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "system_fingerprint_or_backend_id",
        "version",
        "concept_key",
        "source_item_id",
        "material_no",
        "question_code",
        "visible_text_sha256",
    ]
    merged_columns = [column for column in merged_columns if column in scores.columns]
    score_output = scores[merged_columns].sort_values(["concept_key", "version", "judge_provider"])

    qa = {
        "status": "PASS",
        "api_rows": int(len(scores)),
        "providers": int(scores["judge_provider"].nunique()),
        "texts": int(scores["item_id"].nunique()),
        "api_parse_success": int(
            scores["parse_success"].astype(str).str.lower().isin(["true", "1"]).sum()
        ),
        "reader_responses": int(wide["response_id"].nunique()),
        "reader_participant_clusters": int(wide["participant_id"].nunique()),
        "reader_paired_rows": int(len(wide)),
        "analysis_cells": int(len(cells)),
        "concept_blocks": int(len(concepts)),
        "version_cell_counts": cells.groupby("version")["reader_n_responses"].first().astype(int).to_dict(),
        "score_range_valid": bool(
            scores[QUALITY_DIMENSIONS + ["risk"]].apply(lambda column: column.between(1, 5).all()).all()
        ),
        "privacy": "Only already-pseudonymized reader data are read; no respondent-level data are written by this script.",
    }

    outputs = {
        "api_scores_merged_162.csv": score_output,
        "reader_bridge_cells_18.csv": cells,
        "reader_bridge_associations.csv": associations,
        "reader_bridge_version_summary.csv": version_summary,
    }
    for filename, frame in outputs.items():
        write_csv(BASE / filename, frame)

    results = {
        "analysis_version": SCRIPT_VERSION,
        "estimand": (
            "Within-concept association across A/B/C displayed-text versions between the nine-judge "
            "panel score and the concept-by-version reader endpoint"
        ),
        "inference": {
            "exact_test": (
                "two-sided exhaustive permutation of API version labels independently within each concept"
            ),
            "confidence_interval": (
                "hierarchical percentile bootstrap resampling privacy-safe participant clusters and concept blocks"
            ),
            "scope": (
                "the six purposively selected concepts only; not a held-out calibration or population-general claim"
            ),
        },
        "primary_tests": associations.loc[
            associations["family"] == "primary_all_6_concepts"
        ].to_dict("records"),
        "fixed_sensitivity": associations.loc[
            associations["family"]
            == "fixed_sensitivity_excluding_nonparallel_adaptation"
        ].to_dict("records"),
        "aggregation_sensitivity": associations.loc[
            associations["family"] == "aggregation_sensitivity"
        ].to_dict("records"),
        "secondary_tests": associations.loc[
            associations["family"] == "secondary_all_6_concepts"
        ].to_dict("records"),
        "version_summary": version_summary.to_dict("records"),
        "qa": qa,
    }
    write_json(BASE / "reader_bridge_analysis_results.json", results)
    make_report(BASE / "READER_BRIDGE_REPORT.md", associations, version_summary, qa)

    output_hashes = {
        filename: sha256_file(BASE / filename)
        for filename in [
            *outputs.keys(),
            "reader_bridge_analysis_results.json",
            "READER_BRIDGE_REPORT.md",
        ]
    }
    manifest = {
        "analysis_version": SCRIPT_VERSION,
        "script": {"path": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
        "sources": {
            **api_meta,
            **reader_meta,
            "mapping": {"path": "mapping.csv", "sha256": sha256_file(BASE / "mapping.csv")},
            "runner_inputs": {
                "path": "runner_inputs.csv",
                "sha256": sha256_file(BASE / "runner_inputs.csv"),
            },
        },
        "reader_bootstrap": reader_boot_meta,
        "complete_analysis_set": {
            "primary": [
                "api_quality_mean -> accuracy_change",
                "api_risk_mean -> accuracy_change",
            ],
            "fixed_sensitivity": "exclude adaptation nonparallel pre/post pair",
            "secondary": [
                "api_quality_mean -> post_accuracy",
                "api_risk_mean -> post_accuracy",
                "api_quality_mean -> reader_confidence_mean",
                "api_risk_mean -> reader_perceived_risk_mean",
            ],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "outputs": output_hashes,
        "qa": qa,
        "secret_handling": "No credential values are read or written by this analysis script.",
    }
    write_json(BASE / "reader_bridge_analysis_manifest.json", manifest)

    # Fail closed if an accidental credential-like token entered an analysis artifact.
    forbidden_fragments = ["sk-", "AIza", "AQ.", "api_key", "authorization: bearer"]
    for filename in [*outputs.keys(), "reader_bridge_analysis_results.json", "READER_BRIDGE_REPORT.md", "reader_bridge_analysis_manifest.json"]:
        text = (BASE / filename).read_text(encoding="utf-8-sig", errors="ignore").lower()
        if any(fragment.lower() in text for fragment in forbidden_fragments):
            raise AssertionError(f"Credential-like fragment detected in {filename}")

    primary = associations.loc[associations["family"] == "primary_all_6_concepts"]
    print(
        json.dumps(
            {
                "qa": qa,
                "primary": primary[
                    [
                        "predictor",
                        "centered_pearson_r",
                        "hierarchical_bootstrap_r_ci_low",
                        "hierarchical_bootstrap_r_ci_high",
                        "exact_block_permutation_p_two_sided",
                        "p_holm_within_family",
                    ]
                ].to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
