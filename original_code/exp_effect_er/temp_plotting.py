#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 10:21:15 2025

    A simple script for re-plotting visualization figures

@author: dingrui
"""

#%% NECESSARY MODULES
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pygam import LinearGAM, s, f
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import gaussian_kde
from ridgeplot import ridgeplot

import mylib.plotting.beh_plot as bh_plot
import mylib.plotting.basic as bs

import warnings
warnings.filterwarnings('ignore', message='.*deprecated.*')

#%% IMPORT NECESSARY DATA

# subject data
dir_beh_data  = '/public/home/dingrui/fmri_analysis/data/beh'
data_subs_ER  = os.path.join(dir_beh_data, 'data_4_anova_ER.csv')
data_subs_ERs = os.path.join(dir_beh_data, 'er_profiles.csv')

df_data_ER  = pd.read_csv(data_subs_ER, sep=',', index_col=False)
df_data_ERs = pd.read_csv(data_subs_ERs, sep='\t', index_col=False)

df_data_ER['cognition'] = [cog.upper() for cog in list(df_data_ER['cognition'])]
df_data_ERs['ERQ_tendency'] = df_data_ERs['ERQ-CR']-df_data_ERs['ERQ-ES']

# subject basic info
sub_info = pd.read_csv(
    os.path.join(dir_beh_data, 'participants_in_tfmri_demographics.csv'), 
    sep=',', 
    usecols=['sub_id', 'gender', 'age', 'site_id', 'site_region']
)

col_site_id = []
col_site_region = []
for sub in list(df_data_ER['sub_id']):
    site_id = sub_info['site_id'][sub_info['sub_id']==sub].tolist()[0]
    site_region = sub_info['site_region'][sub_info['sub_id']==sub].tolist()[0]
    col_site_id.append(site_id)
    col_site_region.append(site_region)
    
df_data_ER['site_id'] = col_site_id
df_data_ER['site_region'] = col_site_region


#%% SET BASIC PROPERTY FOR PLOTTING

# fig properties
bs.set_pub_style(
    base_fontsize=16,
    font_family="DejaVu Sans",   # change to 'Arial' if your environment has it
    linewidth=1.8,
    tick_width=1.2,
    grid_alpha=0.25,
    dpi=300
)

#%% PLOTTING 1: boxplots for conditions of cognitive appraisal experiment

# boxplot for conditions comparison (experimental effects):
bh_plot.plot_condition_boxplots(
    df_data_ER,
    condition_col='cognition', score_col='emot_rating', subject_col='sub_id',
    condition_order=['RPSL', 'LKNG', 'LKNT'],
    group_col="gender", group_order=[0, 1],
    figsize=(5, 5),
    add_points=True, point_style='jitter', points_layer='behind',
    point_size=10, point_jitter=0.08, 
    point_palette=['gray', 'gray'], point_edgecolor='k',
    point_jitter_main_axis=0.1,
    box_width=0.22, box_colors={0:"#f68724", 1:"#3278a7"},
    show_counts=False,
)

#%% PLOTTING 2: GAM on age-reappraisal relations

# Define columns
age_col = "age"
sex_col = "sex"
subject_col = "subject"

# List of behavioral measures to be modeled
behavior_cols = [
    "success_ER",
    "efforts_ER",
    "ERQ-CR",
    "ERQ_tendency",
]

# ---------------
# preprocessing
# ---------------
# Encode categorical variables
sex_encoder = LabelEncoder()
df_data_ERs["sex_code"] = sex_encoder.fit_transform(df_data_ERs[sex_col])
sex_labels = sex_encoder.classes_

sub_encoder = LabelEncoder()
df_data_ERs["subject_code"] = sub_encoder.fit_transform(df_data_ERs[subject_col])

# standardization
df_dev = df_data_ERs.copy()
scaler = StandardScaler()
df_dev[behavior_cols] = scaler.fit_transform(df_dev[behavior_cols])

# Design matrix:
# column 0: age (smooth term)
# column 1: sex (factor)
# column 2: subject (factor, approximate random intercept)
X = df_dev[[age_col, "sex_code"]].values
X = np.column_stack([df_dev[[age_col, "sex_code"]].values, df_dev["sex_code"]])

# Age grid for trajectory prediction
age_grid = np.linspace(
    df_dev[age_col].min(),
    df_dev[age_col].max(),
    300
)

# ---------------
# models fitting
# ---------------
gam_models = {}
trajectories = {}

for beh in behavior_cols:
    y = df_dev[beh].values

    # GAM with sex-specific smooths
    gam = LinearGAM(
        s(0, n_splines=4) +
        f(1) +                     # sex main effect
        s(0, by=1, n_splines=4)    # age smooth modulated by sex
       
    )

    # Robust smoothing parameter selection
    gam.gridsearch(
        X,
        y,
        lam=np.logspace(-3, 3, 15)
    )

    gam_models[beh] = gam

    # Store trajectories for each sex
    trajectories[beh] = {}

    for sex_code, sex_label in enumerate(sex_labels):
        X_pred = np.zeros((len(age_grid), 3))
        X_pred[:, 0] = age_grid
        X_pred[:, 1] = sex_code
        X_pred[:, 2] = sex_code

        y_pred = gam.predict(X_pred)
        ci = gam.prediction_intervals(X_pred, width=0.95)

        trajectories[beh][sex_label] = {
            "age": age_grid,
            "mean": y_pred,
            "ci_low": ci[:, 0],
            "ci_high": ci[:, 1]
        }

# --------------------
# trajectory plotting
# --------------------
n_beh = len(behavior_cols)

fig, axes = plt.subplots(
    1,
    n_beh,
    figsize=(4 * n_beh, 6),
    sharey=True
)

if n_beh == 1:
    axes = [axes]

colors = ["#f68724", "#3278a7"]  # male / female

for ax, beh in zip(axes, behavior_cols):
    for (sex_label, color) in zip(sex_labels, colors):
        traj = trajectories[beh][sex_label]

        ax.plot(
            traj["age"],
            traj["mean"],
            label=sex_label,
            color=color,
            linewidth=4
        )

        # ax.fill_between(
        #     traj["age"],
        #     traj["ci_low"],
        #     traj["ci_high"],
        #     color=color,
        #     alpha=0.2
        # )
    
    x_ticks = np.arange(6, 19, 3) 
    ax.set_xticks(x_ticks)
    if beh != 'success_ER':
        ax.tick_params(axis='y', which='both', length=0)
    
    ax.set_title(beh)
    ax.set_xlabel("Age")
    for spine in ['top', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    if beh == 'success_ER':
        ax.spines['left'].set_visible(True)

axes[0].set_ylabel("Standardized RPSL measure")

plt.tight_layout()
plt.show()

#%% PLOTTING 3: age distribution

# ensure correct dtypes
df_ageDist = df_data_ER.dropna(subset=["age", "gender", "site_region"])
df_ageDist["age"] = pd.to_numeric(df_ageDist["age"])
df_ageDist["site_region"] = df_ageDist["site_region"].astype("category")

site_order = sorted(df_ageDist["site_region"].unique())

# recommended: explicit labels
df_ageDist["gender"] = df_ageDist["gender"].map({
    0: "Male",
    1: "Female",
    "M": "Male",
    "F": "Female"
})

# --------------------------
# age distribution by gender
# --------------------------

# histogram + KDE
fig, ax = plt.subplots(figsize=(7, 5))

palette = {
    "Male": "#f68724",    # muted blue
    "Female": "#3278a7"   # muted orange
}

sns.histplot(
    data=df_ageDist,
    x="age",
    hue="gender",
    bins=25,
    stat="density",
    common_norm=False,
    multiple='stack',
    kde=True, line_kws={"linewidth":3},
    element="bars",
    fill=True,
    linewidth=1.8,
    palette=palette,
    ax=ax, legend=False,
)

ax.set_xlabel("Age (years)")
ax.set_ylabel("Density")
ax.set_title("Age distribution by gender")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# --------------------------
# age distribution by site
# --------------------------

n_sites = len(site_order)

samples = []
labels = []       
row_labels = []    

for site in site_order:
    row = []
    male_ages = df_ageDist[(df_ageDist["site_region"] == site) & (df_ageDist["gender"] == "Male")]["age"].values
    female_ages = df_ageDist[(df_ageDist["site_region"] == site) & (df_ageDist["gender"] == "Female")]["age"].values

    row.append(male_ages)
    row.append(female_ages)

    samples.append(row)
    
    labels.append([f"{site}_Male", f"{site}_Female"])
    row_labels.append(site)

fig = ridgeplot(
    samples=samples,
    labels=labels,
    row_labels=row_labels,             
    kernel="gau",
    bandwidth='normal_reference',
    kde_points=500,
    color_discrete_map={
        **{f"{site}_Male": "#f68724" for site in site_order},
        **{f"{site}_Female": "#3278a7" for site in site_order},
    },
    opacity=0.75,
    line_color="black", line_width=2.5,
    spacing=0.8,
)


fig.update_layout(
    height=600,
    width=800,
    font_size=16, font_color='black',
    plot_bgcolor="rgb(255, 255, 255)",
    xaxis_gridcolor="white",
    yaxis_gridcolor="black",
    xaxis_gridwidth=2,
    # yaxis_title="Site",
    xaxis_title="Age (years)",
    showlegend=False,
)

fig.show()

