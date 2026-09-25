#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import chardet as chd
except Exception:
    chd = None


DEFAULT_BEH_DATA_DIR = Path("/public/home/dingrui/fmri_analysis/data/beh")
DEFAULT_FMRI_DATA_DIR = Path("/public/home/dingrui/BIDS_DATA")
DEFAULT_DOMAIN_LS = ("emo", "soc")
EVENT_USECOLS = [
    "Emotion",
    "Condition",
    "Feeling.RT",
    "Feeling.RESP",
    "Num",
    "Choice.RT",
    "Choice.RESP",
]


def encoding_detect(filename: str) -> Optional[str]:
    if chd is None:
        return None
    with open(filename, "rb") as f:
        content = f.read(10000)
        res_chd = chd.detect(content)
        encoding = res_chd["encoding"]
    return encoding


def clean_stimulus_value(val: object) -> str:
    s = str(val).strip()
    if s.endswith(".png"):
        s = s[:-4]
    return s


def add_return_column(
    df: pd.DataFrame,
    num_col: str = "Num",
    key_col: str = "Choice.RESP",
    out_col: str = "Return",
) -> pd.DataFrame:
    key2 = {1: 0, 2: 2, 3: 3, 4: 4}
    key4 = {1: 0, 2: 4, 3: 6, 4: 8}

    df = df.copy()
    df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    df[key_col] = pd.to_numeric(df[key_col], errors="coerce")

    fb2 = df[key_col].map(key2)
    fb4 = df[key_col].map(key4)
    df[out_col] = np.where(df[num_col] == 2, fb2, np.where(df[num_col] == 4, fb4, np.nan))
    return df


def _binary_response(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    vals = pd.to_numeric(series, errors="coerce")
    out.loc[vals.isin([1, 2])] = 0.0
    out.loc[vals.isin([3, 4])] = 1.0
    return out


def _rt_keep_mask(series: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    keep = vals.notna().copy()
    if int(keep.sum()) < 3:
        return keep

    x = vals.loc[keep].to_numpy(dtype=float)
    sd = float(np.nanstd(x, ddof=0))
    if (not np.isfinite(sd)) or sd <= 0:
        return keep

    z = np.abs((x - float(np.nanmean(x))) / sd)
    keep_idx = vals.loc[keep].index
    keep.loc[keep_idx] = z < float(z_thresh)
    return keep


def _normalize_subject_id(sub: object) -> str:
    raw = str(sub).strip()
    return raw.replace("sub-", "") if raw.startswith("sub-") else raw


def load_subject_lists(
    dir_beh_data: Path = DEFAULT_BEH_DATA_DIR,
    participants_file: Optional[Path] = None,
    valid_er_file: Optional[Path] = None,
    valid_tg_file: Optional[Path] = None,
) -> Tuple[List[str], List[str], List[str], pd.DataFrame]:
    participants_file = participants_file or (Path(dir_beh_data) / "participants_in_tfmri_demographics.csv")
    valid_er_file = valid_er_file or (Path(dir_beh_data) / "sub_list_valid_4_ER.txt")
    valid_tg_file = valid_tg_file or (Path(dir_beh_data) / "sub_list_valid_4_TG.txt")

    sub_info = pd.read_csv(participants_file, sep=",", usecols=["sub_id", "gender", "age"])
    sub_ls = [_normalize_subject_id(x) for x in sub_info["sub_id"].tolist()]
    sub_valid_er = [_normalize_subject_id(x) for x in np.loadtxt(valid_er_file, dtype=str).tolist()]
    sub_valid_tg = [_normalize_subject_id(x) for x in np.loadtxt(valid_tg_file, dtype=str).tolist()]
    return sub_ls, sub_valid_er, sub_valid_tg, sub_info


def detect_paradigm_version(
    sub: str,
    dir_fmri_data: Path = DEFAULT_FMRI_DATA_DIR,
    domain_ls: Sequence[str] = DEFAULT_DOMAIN_LS,
) -> str:
    txt_files = [
        Path(dir_fmri_data)
        / f"{domain}_20250623"
        / "v1"
        / f"sub-{sub}"
        / "beh"
        / f"func_task_{domain}"
        / "run-1"
        / f"sub-{sub}_task-{domain.upper()}_events.txt"
        for domain in domain_ls
    ]
    fpath = next((p for p in txt_files if p.exists()), None)
    if fpath is None:
        raise FileNotFoundError(f"Cannot find task txt file for subject {sub}")

    encoding = encoding_detect(str(fpath)) or "utf-8"
    with fpath.open("r", encoding=encoding, errors="replace") as txt_file:
        for line in txt_file:
            if "Experiment" in line:
                parts = line.split("_")
                if len(parts) > 1:
                    return parts[1]
                break
    return ""


def load_subject_trials(
    sub: str,
    dir_fmri_data: Path = DEFAULT_FMRI_DATA_DIR,
    domain_ls: Sequence[str] = DEFAULT_DOMAIN_LS,
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for domain in domain_ls:
        fpath = (
            Path(dir_fmri_data)
            / f"{domain}_20250623"
            / "v1"
            / f"sub-{sub}"
            / "beh"
            / f"func_task_{domain}"
            / "run-1"
            / f"sub-{sub}_task-{domain.upper()}_events.tsv"
        )
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, sep="\t", usecols=EVENT_USECOLS).copy()
        df["task"] = domain.upper()
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"Cannot find task tsv file for subject {sub}")
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


def preprocess_subject_trials(df_sub_data_trials: pd.DataFrame, ver_paradigm: str) -> pd.DataFrame:
    df = df_sub_data_trials.copy()

    if "A" in str(ver_paradigm):
        df["Feeling.RESP"] = 5 - pd.to_numeric(df["Feeling.RESP"], errors="coerce")
    if "B" in str(ver_paradigm):
        df["Choice.RESP"] = 5 - pd.to_numeric(df["Choice.RESP"], errors="coerce")

    df = add_return_column(df, num_col="Num", key_col="Choice.RESP", out_col="Return")
    df["response_ER"] = _binary_response(df["Feeling.RESP"])
    df["response_TG"] = _binary_response(df["Choice.RESP"])

    col_cond_er = []
    for row in df.itertuples(index=False):
        emotion = str(getattr(row, "Emotion", ""))
        condition = str(getattr(row, "Condition", ""))
        if (not emotion.startswith("Neutral")) and condition == "Reappraisal.png":
            col_cond_er.append("rpsl")
        elif (not emotion.startswith("Neutral")) and condition == "PassiveLook.png":
            col_cond_er.append("lkng")
        else:
            col_cond_er.append("lknt")
    df["condition_ER"] = col_cond_er

    col_cond_tg = []
    col_cond_tst = []
    for row in df.itertuples(index=False):
        num = pd.to_numeric(getattr(row, "Num", np.nan), errors="coerce")
        cond_er = str(getattr(row, "condition_ER", ""))
        if cond_er == "rpsl" and num == 2:
            col_cond_tg.append("rpsl_lowTrust")
        elif cond_er == "rpsl" and num == 4:
            col_cond_tg.append("rpsl_highTrust")
        elif cond_er != "rpsl" and num == 2:
            col_cond_tg.append("pslk_lowTrust")
        else:
            col_cond_tg.append("pslk_highTrust")
        col_cond_tst.append("low" if num == 2 else "high")
    df["condition_TG"] = col_cond_tg
    df["condition_Trust"] = col_cond_tst

    df["Feeling.RT"] = pd.to_numeric(df["Feeling.RT"], errors="coerce") / 1000.0
    df["Choice.RT"] = pd.to_numeric(df["Choice.RT"], errors="coerce") / 1000.0
    df["raw_condition"] = df["Condition"].map(clean_stimulus_value)
    df["raw_emotion"] = df["Emotion"].map(clean_stimulus_value)
    df["stimulus_content"] = df.apply(
        lambda row: f"{str(row['task']).strip()}_{clean_stimulus_value(row['Condition'])}_{clean_stimulus_value(row['Emotion'])}",
        axis=1,
    )

    emot_keep = _rt_keep_mask(df["Feeling.RT"])
    choice_keep = _rt_keep_mask(df["Choice.RT"])
    valid_emot = df["Feeling.RESP"].notna() & emot_keep
    valid_choice = df["Choice.RESP"].notna() & df["Feeling.RESP"].notna() & choice_keep

    df["valid_emot"] = valid_emot.astype(bool)
    df["valid_choice"] = valid_choice.astype(bool)
    df["emot_rating"] = np.where(valid_emot, pd.to_numeric(df["Feeling.RESP"], errors="coerce"), np.nan)
    df["emot_rt"] = np.where(valid_emot, pd.to_numeric(df["Feeling.RT"], errors="coerce"), np.nan)
    df["emot_binary"] = np.where(valid_emot, pd.to_numeric(df["response_ER"], errors="coerce"), np.nan)
    df["choice_rt"] = np.where(valid_choice, pd.to_numeric(df["Choice.RT"], errors="coerce"), np.nan)
    df["choice_binary"] = np.where(valid_choice, pd.to_numeric(df["response_TG"], errors="coerce"), np.nan)
    df["return_score"] = np.where(valid_choice, pd.to_numeric(df["Return"], errors="coerce"), np.nan)
    return df


def build_er_trial_table(sub: str, df_sub_data_trials: pd.DataFrame) -> Optional[pd.DataFrame]:
    df_er = df_sub_data_trials.loc[df_sub_data_trials["valid_emot"]].copy()
    if df_er.empty:
        return None
    df_er["sub_id"] = sub
    return df_er[
        [
            "sub_id",
            "task",
            "raw_condition",
            "raw_emotion",
            "stimulus_content",
            "Feeling.RESP",
            "Feeling.RT",
            "response_ER",
            "condition_ER",
        ]
    ].rename(
        columns={
            "sub_id": "participant_id",
            "Feeling.RT": "rt",
            "condition_ER": "condition",
            "Feeling.RESP": "emot_rating",
        }
    )


def audit_behavior_subjects(
    subjects: Optional[Sequence[str]] = None,
    dir_beh_data: Path = DEFAULT_BEH_DATA_DIR,
    dir_fmri_data: Path = DEFAULT_FMRI_DATA_DIR,
    participants_file: Optional[Path] = None,
    valid_er_file: Optional[Path] = None,
    valid_tg_file: Optional[Path] = None,
    domain_ls: Sequence[str] = DEFAULT_DOMAIN_LS,
) -> pd.DataFrame:
    sub_ls, sub_valid_er, sub_valid_tg, _ = load_subject_lists(
        dir_beh_data=dir_beh_data,
        participants_file=participants_file,
        valid_er_file=valid_er_file,
        valid_tg_file=valid_tg_file,
    )

    if subjects is None:
        iter_subjects = list(sub_ls)
    else:
        wanted = {_normalize_subject_id(x) for x in subjects}
        iter_subjects = [s for s in sub_ls if s in wanted]

    valid_er_set = set(sub_valid_er)
    valid_tg_set = set(sub_valid_tg)
    rows = []

    for sub in tqdm(iter_subjects, desc="auditing subject inclusion for ER export"):
        row = {
            "subject": f"sub-{sub}",
            "listed_in_participants": True,
            "valid_subject_er": bool(sub in valid_er_set),
            "valid_subject_tg": bool(sub in valid_tg_set),
            "has_task_txt": False,
            "has_task_tsv": False,
            "paradigm_version": "",
            "n_trials_raw": 0,
            "n_trials_emo": 0,
            "n_trials_soc": 0,
            "n_trials_with_feeling_resp": 0,
            "n_trials_with_choice_resp": 0,
            "n_trials_valid_emot": 0,
            "n_trials_valid_choice": 0,
            "n_trials_invalid_emot_due_missing_resp": 0,
            "n_trials_invalid_emot_due_rt_filter": 0,
            "n_trials_invalid_choice_due_missing_resp": 0,
            "n_trials_invalid_choice_due_missing_feeling": 0,
            "n_trials_invalid_choice_due_rt_filter": 0,
            "included_in_data_4_hddm_ER": False,
            "er_exclusion_reason": "",
        }

        try:
            ver_paradigm = detect_paradigm_version(sub, dir_fmri_data=dir_fmri_data, domain_ls=domain_ls)
            row["has_task_txt"] = True
            row["paradigm_version"] = str(ver_paradigm)
        except FileNotFoundError:
            row["er_exclusion_reason"] = "missing_task_txt"
            rows.append(row)
            continue

        try:
            df_trials = load_subject_trials(sub, dir_fmri_data=dir_fmri_data, domain_ls=domain_ls)
            row["has_task_tsv"] = True
        except FileNotFoundError:
            row["er_exclusion_reason"] = "missing_task_tsv"
            rows.append(row)
            continue

        df_trials = preprocess_subject_trials(df_trials, ver_paradigm=ver_paradigm)
        row["n_trials_raw"] = int(df_trials.shape[0])
        if "task" in df_trials.columns:
            row["n_trials_emo"] = int((df_trials["task"].astype(str) == "EMO").sum())
            row["n_trials_soc"] = int((df_trials["task"].astype(str) == "SOC").sum())

        feeling_resp = pd.to_numeric(df_trials["Feeling.RESP"], errors="coerce")
        choice_resp = pd.to_numeric(df_trials["Choice.RESP"], errors="coerce")
        valid_emot = df_trials["valid_emot"].astype(bool)
        valid_choice = df_trials["valid_choice"].astype(bool)
        emot_rt_keep = _rt_keep_mask(df_trials["Feeling.RT"])
        choice_rt_keep = _rt_keep_mask(df_trials["Choice.RT"])

        row["n_trials_with_feeling_resp"] = int(feeling_resp.notna().sum())
        row["n_trials_with_choice_resp"] = int(choice_resp.notna().sum())
        row["n_trials_valid_emot"] = int(valid_emot.sum())
        row["n_trials_valid_choice"] = int(valid_choice.sum())
        row["n_trials_invalid_emot_due_missing_resp"] = int((~feeling_resp.notna()).sum())
        row["n_trials_invalid_emot_due_rt_filter"] = int((feeling_resp.notna() & ~emot_rt_keep).sum())
        row["n_trials_invalid_choice_due_missing_resp"] = int((~choice_resp.notna()).sum())
        row["n_trials_invalid_choice_due_missing_feeling"] = int((choice_resp.notna() & ~feeling_resp.notna()).sum())
        row["n_trials_invalid_choice_due_rt_filter"] = int((choice_resp.notna() & feeling_resp.notna() & ~choice_rt_keep).sum())

        if row["n_trials_valid_emot"] <= 0:
            row["er_exclusion_reason"] = "no_valid_emot_trials_after_filter"
        else:
            row["included_in_data_4_hddm_ER"] = True
            row["er_exclusion_reason"] = "included"
        rows.append(row)

    return pd.DataFrame(rows)


def collect_behavior_trials(
    subjects: Optional[Sequence[str]] = None,
    dir_beh_data: Path = DEFAULT_BEH_DATA_DIR,
    dir_fmri_data: Path = DEFAULT_FMRI_DATA_DIR,
    participants_file: Optional[Path] = None,
    valid_er_file: Optional[Path] = None,
    valid_tg_file: Optional[Path] = None,
    domain_ls: Sequence[str] = DEFAULT_DOMAIN_LS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub_ls, sub_valid_er, sub_valid_tg, _ = load_subject_lists(
        dir_beh_data=dir_beh_data,
        participants_file=participants_file,
        valid_er_file=valid_er_file,
        valid_tg_file=valid_tg_file,
    )

    if subjects is None:
        iter_subjects = list(sub_ls)
    else:
        wanted = {_normalize_subject_id(x) for x in subjects}
        iter_subjects = [s for s in sub_ls if s in wanted]

    sub_trialdat_ls_er = []
    feature_tables = []
    valid_er_set = set(sub_valid_er)
    valid_tg_set = set(sub_valid_tg)

    for sub in tqdm(iter_subjects, desc="organizing subject trials data into reusable tables"):
        try:
            ver_paradigm = detect_paradigm_version(sub, dir_fmri_data=dir_fmri_data, domain_ls=domain_ls)
            df_trials = load_subject_trials(sub, dir_fmri_data=dir_fmri_data, domain_ls=domain_ls)
        except FileNotFoundError:
            continue

        df_trials = preprocess_subject_trials(df_trials, ver_paradigm=ver_paradigm)
        df_trials["subject"] = f"sub-{sub}"
        df_trials["valid_subject_er"] = bool(sub in valid_er_set)
        df_trials["valid_subject_tg"] = bool(sub in valid_tg_set)
        feature_tables.append(
            df_trials[
                [
                    "subject",
                    "task",
                    "raw_condition",
                    "raw_emotion",
                    "stimulus_content",
                    "Num",
                    "condition_ER",
                    "condition_TG",
                    "condition_Trust",
                    "valid_subject_er",
                    "valid_subject_tg",
                    "valid_emot",
                    "valid_choice",
                    "emot_rating",
                    "emot_rt",
                    "emot_binary",
                    "choice_rt",
                    "choice_binary",
                    "return_score",
                ]
            ].copy()
        )

        df_er = build_er_trial_table(sub, df_trials)
        if df_er is not None and not df_er.empty:
            sub_trialdat_ls_er.append(df_er)

    df_er = pd.concat(sub_trialdat_ls_er, ignore_index=True) if sub_trialdat_ls_er else pd.DataFrame()
    df_feature = pd.concat(feature_tables, ignore_index=True) if feature_tables else pd.DataFrame()
    return df_er, df_feature


def collect_behavior_feature_table(
    subjects: Optional[Sequence[str]] = None,
    dir_beh_data: Path = DEFAULT_BEH_DATA_DIR,
    dir_fmri_data: Path = DEFAULT_FMRI_DATA_DIR,
    participants_file: Optional[Path] = None,
    valid_er_file: Optional[Path] = None,
    valid_tg_file: Optional[Path] = None,
    domain_ls: Sequence[str] = DEFAULT_DOMAIN_LS,
) -> pd.DataFrame:
    _, df_feature = collect_behavior_trials(
        subjects=subjects,
        dir_beh_data=dir_beh_data,
        dir_fmri_data=dir_fmri_data,
        participants_file=participants_file,
        valid_er_file=valid_er_file,
        valid_tg_file=valid_tg_file,
        domain_ls=domain_ls,
    )
    return df_feature


def export_behavior_outputs(
    out_dir: Path,
    subjects: Optional[Sequence[str]] = None,
    dir_beh_data: Path = DEFAULT_BEH_DATA_DIR,
    dir_fmri_data: Path = DEFAULT_FMRI_DATA_DIR,
    participants_file: Optional[Path] = None,
    valid_er_file: Optional[Path] = None,
    valid_tg_file: Optional[Path] = None,
    domain_ls: Sequence[str] = DEFAULT_DOMAIN_LS,
    save_feature_table: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_er, df_feature = collect_behavior_trials(
        subjects=subjects,
        dir_beh_data=dir_beh_data,
        dir_fmri_data=dir_fmri_data,
        participants_file=participants_file,
        valid_er_file=valid_er_file,
        valid_tg_file=valid_tg_file,
        domain_ls=domain_ls,
    )
    df_audit = audit_behavior_subjects(
        subjects=subjects,
        dir_beh_data=dir_beh_data,
        dir_fmri_data=dir_fmri_data,
        participants_file=participants_file,
        valid_er_file=valid_er_file,
        valid_tg_file=valid_tg_file,
        domain_ls=domain_ls,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not df_er.empty:
        df_er.to_csv(out_dir / "data_4_hddm_ER.csv", sep=",", index=False)
    if bool(save_feature_table) and not df_feature.empty:
        df_feature.to_csv(out_dir / "data_4_behavior_feature_table.csv", sep=",", index=False)
    if not df_audit.empty:
        df_audit.to_csv(out_dir / "data_4_hddm_ER_subject_audit.csv", sep=",", index=False)
    return df_er, df_feature


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare emo_final behavior tables from raw events data")
    p.add_argument("--beh-data-dir", type=Path, default=DEFAULT_BEH_DATA_DIR)
    p.add_argument("--fmri-data-dir", type=Path, default=DEFAULT_FMRI_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=Path("/public/home/dingrui/fmri_analysis/zz_analysis/roi_results_final/behavior"))
    p.add_argument("--participants-file", type=Path, default=None)
    p.add_argument("--valid-er-file", type=Path, default=None)
    p.add_argument("--valid-tg-file", type=Path, default=None)
    p.add_argument("--save-feature-table", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_behavior_outputs(
        out_dir=args.out_dir,
        subjects=None,
        dir_beh_data=args.beh_data_dir,
        dir_fmri_data=args.fmri_data_dir,
        participants_file=args.participants_file,
        valid_er_file=args.valid_er_file,
        valid_tg_file=args.valid_tg_file,
        save_feature_table=bool(args.save_feature_table),
    )


if __name__ == "__main__":
    main()
