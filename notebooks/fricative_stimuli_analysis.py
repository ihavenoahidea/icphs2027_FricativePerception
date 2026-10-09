# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fricative stimuli analysis
#
# This notebook analyzes the stimuli used in the experiment and produces a figure for the paper.
# The notebook `notebooks/multiarrange_analysis.py` uses the outputs while analyzing the perceptual data.
#
# ## A word on normalization
#
# The stimuli analyzed in this script have already been normalized.
# All of the `wav` files for the stimuli were concatenated together and the max amplitude of the concatenation was scaled to one.
# More details are relayed in the paper.
#
# ## Preliminaries
#
# The first two cells are preliminaries: imports and default labels for the fricatives.
# Change `SAVE_FIGURE_MODE` to `True` if the figure outputs are to be saved.

# %%
# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import phonlab
import re
from pathlib import Path
from collections import namedtuple
from scipy.fft import dct
from fileorganize import dir_to_df
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=['#002676', '#FDB515', '#00553A', '#770747', '#431170'])
import pypst

# %%
# Constants, including directories.
DEFAULT_FRICATIVE_LABELS: tuple[str, ...] = ("bilabial", 
                                        "labio-dental", 
                                        "dental", 
                                        "alveolar", 
                                        "palato-alveolar", 
                                        "retroflex", 
                                        "alveolo-palatal", 
                                        "palatal", 
                                        "velar", 
                                        "uvular", 
                                        "pharyngeal", 
                                        "glottal",)

labels_key = {label: number for number, label in enumerate(DEFAULT_FRICATIVE_LABELS)}

project_dir = Path.cwd().parent
STIMULI_DIR = 'stimuli/normalized/wikicommons_initial'
FIGURE_DIR = project_dir / 'figures/'
RESULTS_DIR = 'data/results/'
TEXTGRID_DIR = Path('../stimuli/normalized/wikicommons_whole_files/')


SAVE_FIGURE_MODE = False
SAVE_OUTPUT_MODE = False

# %% [markdown]
# ## Spectral characteristics
#
# For this analysis, we use the `stimuli/normalized/wikicommons_initial` directory.
# Each file contains 200ms of the normalized fricative, extracted from the center of the fricative.
# The labio-dental fricative was only 183ms long, so the entire fricative is used. 
#
# The script extracts spectral measurements using the `phonlab.fricative()` function at ten equidistant points in the audio sample. 
# It uses `scipy.fft.dct()` to calculate the first four discrete cosine transform coefficients as well.

# %%
measurement_percentiles = np.linspace(0.1, 0.9, 9)

files_df = []
measurements_df = []
for file in (project_dir / STIMULI_DIR).iterdir():
    sig, sr = phonlab.loadsig(file)
    assert sig.ndim == 1
    filename = file.name
    fricative_label = re.search(r"^(.+)_", filename).group(1)
    duration = sig.shape[0]/sr
    files_df.append({
        'signal': sig,
        'sampling_rate': sr,
        'filename': filename,
        'fricative_label': fricative_label
    })
    for percentile in measurement_percentiles:
        t = duration * percentile
        measurements = phonlab.fricative(sig, sr, t)
        dcts = dct(measurements.spec)[0:4]
        measurements_df.append({
            'filename': filename,
            'fricative_label': fricative_label,
            'time': t,            
            'percentile': percentile,
            **measurements._asdict(),
            'DCT0': dcts[0],
            'DCT1': dcts[1],
            'DCT2': dcts[2],
            'DCT3': dcts[3],
        })
files_df = pd.DataFrame(files_df)
measurements_df = pd.DataFrame(measurements_df)
for fric in measurements_df['fricative_label']:
    assert fric in DEFAULT_FRICATIVE_LABELS
measurements_df = (measurements_df
    .sort_values(['fricative_label', 'percentile'], 
                 key=lambda x: x if x.dtype == 'float64' else x.apply(labels_key.get))
    .reset_index(drop=True)
                  )
measurements_df.head()


# %% [markdown]
# Now we'll plot sample spectra for each stimulus.
# For each stimulus, we'll take the sample with the Fm measurement closest to the median and use that for plotting.

# %%
# Helper function to find the spectral measurements at the point closest to the median Fm value
def closest_to_median(group):
    med = group['Fm'].median()
    idx = (group['Fm'] - med).abs().idxmin()
    return group.loc[idx]

# Create a dataframe of just those spectral measurements
plotframe = (measurements_df.groupby('fricative_label')
    .apply(closest_to_median, include_groups=False)
    .reset_index()
    .sort_values(['fricative_label', 'percentile'], key=lambda x: x if x.dtype == 'float64' else x.apply(labels_key.get))
    .reset_index(drop=True)
            )

# %%
nrow = 4
ncol = 3
fig, axes = plt.subplots(nrow, ncol, sharex = True, sharey=True, figsize=(10, 15), constrained_layout=True,)

for row in plotframe.itertuples():
    ax = axes[row.Index // 3, row.Index % 3]
    ax.plot(row.freq, row.spec)
    ax.axvline(row.COG, color = '#00553A', linestyle='--')
    ax.axvline(row.Fm, color='#FDB515', linestyle='--')
    ax.axhline(row.Am, color='#FDB515', linestyle='--')
    if row.Fm < row.COG:
        ax.text(row.Fm - 50, 0.05, f"Fm={row.Fm:.0f}Hz", rotation=90, color = '#FDB515', ha='right', va='bottom', transform=ax.get_xaxis_transform())
        ax.text(row.COG + 100, 0.05, f"COG={row.COG:.0f}Hz", rotation=90, color = '#00553A', ha='left', va='bottom', transform=ax.get_xaxis_transform())
    else:
        ax.text(row.Fm + 100, 0.05, f"Fm={row.Fm:.0f}Hz", rotation=90, color = '#FDB515', ha='left', va='bottom', transform=ax.get_xaxis_transform())
        ax.text(row.COG - 50, 0.05, f"COG={row.COG:.0f}Hz", rotation=90, color = '#00553A', ha='right', va='bottom', transform=ax.get_xaxis_transform())
    ax.text(.98, row.Am+1, f"Am={row.Am:.2f}dB", color="#FDB515", ha="right", transform=ax.get_yaxis_transform())
    ax.set_title(row.fricative_label)
    ax.annotate(f"{row.Fm:.0f}", xy=(row.Fm, -100))
    
fig.suptitle("Stimuli spectra", fontweight="bold")
fig.supylabel("Amplitude (dB)", fontweight="bold")
fig.supxlabel("Frequency (Hz)", fontweight="bold")
if SAVE_FIGURE_MODE:
    fig.savefig(FIGURE_DIR / 'stimuli_spectra.png')
plt.show()


# %% [markdown]
# ## Duration data
#
# Now we get data about the whole stimulus---basically, the duration of the segment and syllables.
# Each TextGrid labels the /Fa aFa/ stimulus with a word tier, for the syllables, and a segment tier for the segments.

# %%
textgrids = dir_to_df(TEXTGRID_DIR).loc[lambda df: df['fname'].str.endswith('TextGrid')]

duration_df = []


for textgrid in textgrids['fname']:
    # Separate 'word' and 'segment' tiers into dataframes.
    word_df, segment_df = phonlab.tg_to_df(TEXTGRID_DIR / textgrid)

    # Find first non-empty cell in the 'word' dataframe.
    # This is our stimulus syllable. Save the data.
    for idx, t1, t2, word in word_df.itertuples():
        if word != '':
            syllable_t1 = t1
            syllable_t2 = t2
            syllable = word
            break

    # Find first two non-empty cells in the 'segment' dataframe.
    # These are our fricative segment and its following vowel.
    # Save the data
    counter = 0
    for idx, t1, t2, segment in segment_df.itertuples():
        if segment != '' and counter == 0:
            fricative_t1= t1
            fricative_t2 = t2
            fricative = segment
            counter += 1
        elif segment != '' and counter == 1:
            vowel_t1 = t1
            vowel_t2 = t2
            vowel = segment
            counter += 1

    # Store the data in a dictionary and append it to a list.
    duration_df.append({
        'textgrid_name': textgrid,
        'syllable_t1': syllable_t1,
        'syllable_t2': syllable_t2,
        'syllable': syllable,
        'fricative_t1': fricative_t1,
        'fricative_t2': fricative_t2,
        'fricative': fricative,
        'vowel_t1': vowel_t1,
        'vowel_t2': vowel_t2,
        'vowel': vowel,
    })

# Convert list into a dataframe and calculate duration data.
duration_df = pd.DataFrame(duration_df).assign(
    fricative_label = lambda x: x['textgrid_name'].str.extract(r"_(.+)_"),
    syllable_dur = lambda x: x['syllable_t2'] - x['syllable_t1'],
    fricative_dur = lambda x: x['fricative_t2'] - x['fricative_t1'],
    vowel_dur = lambda x: x['vowel_t2'] - x['vowel_t1'],
                                      )

# Retain only the relevant columns
duration_df = duration_df[['textgrid_name', 'fricative_label', 'fricative', 'fricative_dur', 'syllable', 'syllable_dur', 'vowel', 'vowel_dur']]

# Check that every row of the dataframe is actually one of our fricatives
for fric in duration_df['fricative_label']:
    assert fric in DEFAULT_FRICATIVE_LABELS

duration_df

# %% [markdown]
# Now we take the medians for each fricative's measurements in `measurements_df` and merge them with the `duration_df`.
# If `SAVE_OUTPUT_MODE = True`, the resulting full dataframe is pickled, as `data/processed/fricative_df.pkl` to be used by `notebooks/multiarrange_analysis.py`.

# %%
# Find the medians of numeric columns in `measurements_df`
medians_df = measurements_df[['fricative_label', 'Fm', 'Am', 'AmpD',
       'Fsec', 'Asec', 'COG', 'SD', 'Skew', 'Kurtosis', 
       'DCT0', 'DCT1', 'DCT2', 'DCT3']].groupby('fricative_label').agg('median').reset_index()

# Merge medians_df with full_df
full_df = pd.merge(medians_df, duration_df).sort_values('fricative_label', key=lambda x: x.apply(labels_key.get)).reset_index(drop=True)

if SAVE_OUTPUT_MODE:
    full_df.to_pickle('../data/processed/fricative_df.pkl')
    
full_df 

