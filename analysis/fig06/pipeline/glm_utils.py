#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glm_utils.py
Utilities for confound regression and event classification
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_complex_confounds(confounds_path):
    """
    Confound set used in the first-level model:
    1. Friston-24 (6 motion parameters, their derivatives, squares, and squared derivatives)
    2. The first 6 aCompCor components
    3. Scrubbing (motion_outlier and non_steady_state)
    """
    # Confounds = Friston-24 + aCompCor(6) + scrubbing regressors (outlier spikes).
    try:
        df = pd.read_csv(confounds_path, sep='\t')

        # A. Friston-24
        base_axes = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

        # If the standard names are absent, use the motion columns that are present.
        if not all(col in df.columns for col in base_axes):
            # Collect available trans_ or rot_ columns.
            found_cols = []
            for col in df.columns:
                if col.startswith('trans_') or col.startswith('rot_'):
                    found_cols.append(col)
            
            if len(found_cols) >= 6:
                # Use the first six columns found.
                base_axes = found_cols[:6]
                logger.warning(f"Standard motion columns were not found; using {base_axes}")
            else:
                raise ValueError(f"Not enough motion-parameter columns; found {len(found_cols)}")

        R_motion = df[base_axes].copy()
        R_deriv = R_motion.diff().fillna(0)
        R_power2 = R_motion ** 2
        R_deriv_power2 = R_deriv ** 2

        # Rename
        R_deriv.columns = [c + '_dt' for c in base_axes]
        R_power2.columns = [c + '_sq' for c in base_axes]
        R_deriv_power2.columns = [c + '_dt_sq' for c in base_axes]

        friston24 = pd.concat([R_motion, R_deriv, R_power2, R_deriv_power2], axis=1)

        # B. aCompCor (first 6 components)
        acomp_cols = df.filter(regex='^a_comp_cor_0[0-5]$')
        if acomp_cols.empty:
            # Try alternate aCompCor names.
            acomp_cols = df.filter(regex='^a_comp_cor_')
            if not acomp_cols.empty:
                acomp_cols = acomp_cols.iloc[:, :6]  # First six components
                logger.warning(f"Standard aCompCor names were not found; using {len(acomp_cols.columns)} columns")
            else:
                logger.warning("No aCompCor columns found; they will be omitted from the regression")

        # C. Scrubbing (Outliers)
        outlier_cols = df.filter(regex='motion_outlier|non_steady_state')
        if outlier_cols.empty:
            logger.warning("No outlier columns found; they will be omitted from the regression")

        # Concatenate
        final_confounds = pd.concat([friston24, acomp_cols, outlier_cols], axis=1)
        return final_confounds.fillna(0)

    except Exception as e:
        logger.error(f"Failed to read confounds: {confounds_path} | Error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


def reclassify_events(df):
    """
    Classify events into three trial types:
    1. PassiveLook with an emotion label that starts with Neutral
    2. PassiveLook with any other emotion label
    3. Reappraisal
    """
    try:
        # Map raw condition/emotion into 3 analysis trial types + a catch-all "Other".
        col_map = {c.lower(): c for c in df.columns}
        cond_col = col_map.get('condition')
        emo_col = col_map.get('emotion')

        # Return the table unchanged when required columns are missing.
        if not cond_col or not emo_col:
            logger.warning(f"Missing event-classification columns: condition={cond_col}, emotion={emo_col}; keeping the original table")
            return df

        def _get_new_type(row):
            try:
                cond = str(row[cond_col]).strip()
                emo = str(row[emo_col]).strip()

                if cond == 'Reappraisal.png':
                    return 'Reappraisal'
                elif cond == 'PassiveLook.png':
                    if emo.startswith('Neutral'):
                        return 'Passive_Neutral'
                    else:
                        return 'Passive_Emo'
                else:
                    # Any other event is labeled Other.
                    return 'Other'
            except Exception as e:
                logger.warning(f"Event classification failed for a row ({e}); labeling it Other")
                return 'Other'

        df_new = df.copy()
        df_new['trial_type'] = df_new.apply(_get_new_type, axis=1)
        return df_new
    except Exception as e:
        logger.error(f"Event classification failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        # On failure, return the original table so the pipeline can continue.
        return df


def create_lss_events(target_idx, all_events, event_name_col="Choice"):
    """
    Build an LSS event file: one target trial versus all other trials.
    Onset and duration come from Observe.OnsetTime and Feeling.FinishTime.

    Parameters:
    - target_idx: indices of the target trials (for example [3, 5, 7])
    - all_events: E-Prime event table; must contain Observe.OnsetTime and Feeling.FinishTime
    - event_name_col: retained for compatibility; not used to extract times
    """
    lss_df = all_events.copy()

    # 1. Label the target trial and pool the remaining trials.
    # LSS design: one target event vs a pooled "others" event.
    lss_df["trial_type"] = "LSS_OTHERS"
    lss_df.loc[target_idx, "trial_type"] = "LSS_TARGET"

    # --------------------------
    # 2. Onset from Observe.OnsetTime
    # --------------------------
    def get_onset(row):
        onset_field = "Observe.OnsetTime"
        if onset_field not in lss_df.columns:
            raise ValueError(f"Missing column: {onset_field}. Check the E-Prime export.")
        # Convert milliseconds to seconds.
        return row[onset_field] / 1000

    # --------------------------
    # 3. Duration = Feeling.FinishTime - Observe.OnsetTime
    # --------------------------
    def get_duration(row):
        finish_field = "Feeling.FinishTime"
        onset_field = "Observe.OnsetTime"

        for col in (finish_field, onset_field):
            if col not in lss_df.columns:
                raise ValueError(f"Missing column: {col}. Check the E-Prime export.")

        # Duration in milliseconds, with invalid values set to zero.
        duration_ms = row[finish_field] - row[onset_field]
        if pd.isna(duration_ms) or duration_ms <= 0 or duration_ms == -99999:
            return 0  # Invalid durations, including the sentinel -99999, are set to zero.
        return duration_ms / 1000

    # --------------------------
    # 4. Apply onset and duration extraction.
    # --------------------------
    lss_df["onset"] = lss_df.apply(get_onset, axis=1)
    lss_df["duration"] = lss_df.apply(get_duration, axis=1)

    # --------------------------
    # 5. Shift time to zero when Fixation.OnsetTime is present.
    # --------------------------
    if "Fixation.OnsetTime" in lss_df.columns:
        del_time_sec = lss_df["Fixation.OnsetTime"].iloc[0] / 1000
        lss_df["onset"] = lss_df["onset"] - del_time_sec
        lss_df["onset"] = lss_df["onset"].clip(lower=0)  # Do not allow negative onsets.

    # Return the columns expected by the first-level model.
    return lss_df[["onset", "duration", "trial_type"]].dropna()
