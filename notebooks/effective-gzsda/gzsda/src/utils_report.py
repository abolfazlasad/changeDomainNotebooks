"""Build Acc_s / Acc_u / H-mean and timing dataframes from experiment result dicts."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import pandas as pd

from src.our_tupl import GENERATION_POLICIES, METHOD_PREFIX

_METRIC_RE = re.compile(r"(\w+):\s+([\d.]+\s*±\s*[\d.]+)")

OUR_TUPL_METHODS = tuple(f"{METHOD_PREFIX}_{p}" for p in GENERATION_POLICIES)

METHOD_ORDER = (
    "base",
    "VisTA",
    "CCVAE",
    "our0",
    "our_GRE",
    "TUPL",
    *OUR_TUPL_METHODS,
)


def extract_metrics(text: str) -> pd.Series:
    matches = dict(_METRIC_RE.findall(text))
    return pd.Series(matches)


def results_to_dataframe(result: Mapping, method_order: Sequence[str]) -> pd.DataFrame:
    rows = [(k, m, result[m][k]) for m in result for k in result[m]]
    df = pd.DataFrame(rows, columns=["domain", "method", "values"])

    df[["seen", "unseen", "H-mean"]] = df["values"].apply(extract_metrics)
    df = df[["domain", "method", "seen", "unseen", "H-mean"]]

    df["method"] = pd.Categorical(
        df["method"],
        categories=list(method_order),
        ordered=True,
    )
    df = df.sort_values(["domain", "method"]).reset_index(drop=True)
    return df


def times_to_dataframe(
    result_time: Mapping,
    n_senario: int,
    n_trial: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": method,
                "n_senario": n_senario,
                "n_trial": n_trial,
                "total_trial": n_senario * n_trial,
                "total_time": round(total_time, 1),
            }
            for method, total_time in result_time.items()
        ]
    )


def target_method_average(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["target"] = df_copy["domain"].str.split("->").str[1].str.strip()
    target_method_avg = (
        df_copy.groupby(["target", "method"], observed=False)[["seen", "unseen", "H-mean"]]
        .apply(lambda g: g.replace(r"±.*", "", regex=True).astype(float).mean().round(2))
    )
    return target_method_avg.dropna()


def add_metric_value(
    df: pd.DataFrame,
    source_col: str = "unseen",
    dest_col: str = "H-mean_value",
) -> pd.DataFrame:
    df[dest_col] = (
        df[source_col]
        .str.split("±")
        .str[0]
        .astype(float)
    )
    return df


def best_method_per_domain(df: pd.DataFrame) -> pd.DataFrame:
    best = df.loc[df.groupby("domain")["H-mean_value"].idxmax()]
    return best[["domain", "method", "H-mean"]].reset_index(drop=True)
