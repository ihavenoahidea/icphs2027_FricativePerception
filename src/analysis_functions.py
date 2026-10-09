import numpy as np
import pandas as pd
import json
from itertools import combinations, product
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.linalg import orthogonal_procrustes

def rectangle_meadows_json_distances(meadows_json: str) -> pd.DataFrame:
    """Extract dissimilarity data from a Meadows multiarrange experiment to a DataFrame. 

    The function receives a string naming a JSON from a Meadows experiment with
    one or more multiple arrangement tasks. It iterates through the JSON and 
    finds the multiple arrangement tasks, extracting them for the DataFrame.
    See the 'Returns' section below for a description of the DataFrame.

    By default, the function will NOT INCLUDE participants who didn't finish both
    multiarrange tasks, NOR will it include records from previews.

    Args:
        meadows_json: a string naming the filepath to the Meadows data.

    Returns:
        A pd.DataFrame containing four columns:
            Col 0: `fricative1`, the name of the first fricative in the pair.
            Col 1: `fricative2`, the name of the second fricative in the pair.
            Col 2: `distance`, the Meadows-generated dissimilarity between fricatives.
            Col 3: `subject`, the Meadows-generated alias for the subject.
            Col 4: `trial`, a number identifying the task: 0 if it was the first
                multiple arrange task of the experiment, 1 if it was the second, etc.
        Note that the order of the fricatives is purely alphabetical based on the stimulus
        name because there isn't a presentation order in a multiple arrangement experiment.
    """
    with open(meadows_json, mode = 'r') as fhandle:
        data = json.load(fhandle)
    distance_dfs = []
    for subject in data:
        if 'preview' in subject: # Ignore previews
            continue
        tasks = data.get(subject).get('tasks')
        trial = 0
        subject_dfs = []
        for task in tasks:
            if task['task']['task_type'] == 'multiarrange' and task['status'] == 'finished':
                stimuli_names: list = [item['name'] for item in task['stimuli']]
                df = pd.DataFrame(combinations(stimuli_names, 2), columns=['fricative1', 'fricative2'])
                df['distance'] = task['rdm']
                df['subject'] = subject
                df['trial'] = trial
                subject_dfs.append(df)
                trial += 1
        if trial == 2:
            distance_dfs.extend(subject_dfs) 
    return pd.concat(distance_dfs).reset_index(drop=True)

def rectangle_meadows_json_info(meadows_json: str) -> pd.DataFrame:
    """Extract questionnaire data from a Meadows multiarrange JSON to a DataFrame.

    The function receives a string naming a JSON from a Meadows experiment with a form task.
    It iterates through the JSON and finds the tasks with type 'dynform', saving the unhashables
    to a dictionary. Then it createes a dataframe from the dictionaries.

    Args:
        meadows_json: a string naming the filepath to the Meadows data.

    Returns:
        A pd.DataFrame containing one column for each item in the dynform task, plus the alias.
        Note that this means the number of columns is variable, depending on the number of items
        in the form, and that the DF will include the Meadows-internal labels (e.g., 'status')
        in addition to the form fields.
    """

    with open(meadows_json) as fhandle:
        data = json.load(fhandle)
    demographic_dicts = []
    for subject in data:
        tasks = data.get(subject).get('tasks')
        for task in tasks:
            if task['task']['task_type'] == 'dynform':
                usable_data = {key: task[key] for key in task if not isinstance(task[key], dict)}
                usable_data['alias'] = subject
                demographic_dicts.append(usable_data)
    return pd.DataFrame(demographic_dicts)

DEFAULT_FRICATIVE_LABELS: tuple[str, ...] = ("bilabial", 
                                        "labio-dental", 
                                        "dental", 
                                        "alveolar", 
                                        "alveolo-palatal", 
                                        "palato-alveolar", 
                                        "retroflex", 
                                        "palatal", 
                                        "velar", 
                                        "uvular", 
                                        "pharyngeal", 
                                        "glottal",)

def distance_matrix_from_df(df: pd.DataFrame, labels: tuple[str] = DEFAULT_FRICATIVE_LABELS) -> tuple[tuple[str, ...], np.ndarray]:
    """Form a distance matrix from a formatted DataFrame.

    The first column of the dataframe and the second column of the dataframe
    must be IDs for the two points. The third column must be the distances
    between them.

    Args:
        df: a Pandas dataframe formatted as above.
        labels: a tuple with the desired order of stimuli in the distance matrix.
            Each entry in the tuple MUST BE A SUBSTRING of the stimulus that is
            surrounded by underscores ('_') in the stimulus name.

    Returns:
        As a tuple: a list of the labels for the rows/columns of the distance matrix,
        and the np.ndarray distance matrix.
    """
    assert df.shape[0] == len(labels)*(len(labels)-1)*0.5, "Dataframe is not the right length."
    stimuli = pd.unique(df.iloc[:,0:2].values.ravel())
    label_idx = {l: i for i, l in enumerate(labels)}

    stimulus_to_label = {}
    for label, stimulus in product(labels, stimuli):
        if "_"+label+"_" in stimulus:
            stimulus_to_label[stimulus] = label
            
    stimuli_idx = {stimulus:label_idx[stimulus_to_label[stimulus]] for stimulus in stimuli}
    n = len(stimuli_idx)

    mat = np.zeros((n, n))
    for _, row in df.iterrows():
        i, j = stimuli_idx[row.iloc[0]], stimuli_idx[row.iloc[1]]
        mat[i, j] = row.iloc[2]
        mat[j, i] = row.iloc[2]

    return (labels, mat)

def mds_scree(rdm: np.ndarray, dims = range(1, 6), n_init=1, random_state=None):
    out = []
    for i in dims:
        mds = MDS(n_components=i, n_init=n_init, dissimilarity="precomputed", random_state=random_state, normalized_stress=True).fit(rdm)
        out.append(mds.stress_)
    return np.array(out)

def shuffle_rdm(rdm, rng):
    """Permute off-diagonal entries"""
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = rng.permutation(rdm[iu])
    out = np.zeros_like(rdm)
    out[iu] = vals
    out = out + out.T
    return out

def null_mds_scree(rdm: np.ndarray, dims = range(1, 6), n_init=1, n_perm=100, seed=0):
    """Null distribution of the scree curve from shuffled dissimilarities"""
    rng = np.random.default_rng(seed)
    dims = list(dims)
    null = np.empty((n_perm, len(dims)))
    for p in range(n_perm):
        null[p] = mds_scree(shuffle_rdm(rdm, rng), dims=dims, n_init=1, random_state=int(rng.integers(1e9)))
    return null

def plot_null_scree(distance_df, dims=range(1, 6), n_perm=100, seed=0):
    subjects = distance_df.drop('version', axis='columns').groupby('subject')
    n = subjects.ngroups
    fig, axes = plt.subplots(n, 1, figsize=(6, 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    dims = list(dims)

    for ax, (subject, df) in zip(axes, subjects):
        _, rdm = distance_matrix_from_df(df)
        obs = mds_scree(rdm, dims=dims, n_init=4, random_state=seed)
        null = null_mds_scree(rdm, dims=dims, n_perm=n_perm, seed=seed)
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0)

        ax.fill_between(dims, lo, hi, alpha=0.25, label='null 95%')
        ax.plot(dims, null.mean(0), '--', lw=1, label='null mean')
        ax.plot(dims, obs, 'o-', label='observed')
        ax.set_title(subject)
        ax.set_ylabel('stress-1')

    axes[-1].set_xlabel('Number of dimensions')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def gpa(shapes, tol=1e-7, max_iter=200):
    # center and scale each shape
    normed = []
    for A in shapes:
        A = A - A.mean(axis=0)
        A = A / np.linalg.norm(A)
        normed.append(A)

    ref = normed[0]
    prev = np.inf

    for i in range(max_iter):
        aligned = []
        for A in normed:
            R, _ = orthogonal_procrustes(A, ref)
            aligned.append(A @ R)
        mean = np.mean(aligned, axis=0)
        mean = mean - mean.mean(axis=0) # re-center
        mean = mean / np.linalg.norm(mean) # re-scale
        # Convergence: how much did the mean move?
        dist = np.linalg.norm(mean - ref)
        if abs(prev - dist) < tol:
            break
        prev, ref = dist, mean
    print(f"Concluded after {i+1} iterations")
    distances = [np.sum((a - mean) ** 2) for a in aligned] 
    return aligned, mean, distances

"""
Small-multiples visualization of aligned MDS configurations.

One panel per stimulus. Each panel shows, in the shared aligned coordinate
frame:
  - the full consensus configuration, faint grey, for spatial reference
  - each subject's two trials as points, connected by a hairline segment
  - the consensus location of THIS panel's stimulus, bold

Assumes coordinates are ALREADY Procrustes-aligned to the consensus frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_mds_small_multiples(
    df,
    consensus,
    dim_cols=(0, 1),
    ncols=3,
    point_size=9,
    hairline_width=0.5,
    hairline_alpha=0.55,
    point_alpha=1,
    point_color='#FDB515',
    ref_color="0.8",          # faint grey for the background consensus
    ref_size=14,
    consensus_color="#002676",
    panel_size=3,
    label_offset=0.015,
    symdict=None,
):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Columns [dim0, dim1, 'subject', 'label', 'trial'], aligned coords.
        `trial` is 0 or 1.
    consensus : pd.DataFrame
        Columns [dim0, dim1, 'label'] — the mean configuration.
    dim_cols : tuple
        The two coordinate column names (default (0, 1)).
    """
    dx, dy = dim_cols
    labels = list(consensus["label"])
    n = len(labels)
    nrows = int(np.ceil(n / ncols))

    # One consistent color per stimulus, from a perceptually even map.
    cmap = plt.colormaps["tab20"].resampled(n) if n > 10 else plt.colormaps["tab10"].resampled(n)
    color_for = {lab: cmap(i) for i, lab in enumerate(labels)}

    # Shared axis limits across all panels, with a small margin, so every
    # panel is in the same frame and directly comparable.
    all_x = np.concatenate([df[dx].values, consensus[dx].values])
    all_y = np.concatenate([df[dy].values, consensus[dy].values])
    xpad = 0.08 * (all_x.max() - all_x.min())
    ypad = 0.08 * (all_y.max() - all_y.min())
    xlim = (all_x.min() - xpad, all_x.max() + xpad)
    ylim = (all_y.min() - ypad, all_y.max() + ypad)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * panel_size, nrows * panel_size),
        squeeze=False,
    )

    ref_xy = consensus[[dx, dy]].values

    for idx, lab in enumerate(labels):
        ax = axes[idx // ncols][idx % ncols]
        color = point_color

        # 1. Background consensus: whole configuration, faint, for context.
        ax.scatter(ref_xy[:, 0], ref_xy[:, 1],
                   s=ref_size, c=ref_color, zorder=1, linewidths=0, alpha=.8)

        # 2. This stimulus's per-subject data.
        sub = df[df["label"] == lab]

        # Build one segment per subject connecting trial 0 -> trial 1.
        segs = []
        pts_x, pts_y = [], []
        for _, g in sub.groupby("subject"):
            g = g.sort_values("trial")
            if len(g) == 2:
                (x0, y0), (x1, y1) = g[[dx, dy]].values
                segs.append([[x0, y0], [x1, y1]])
            pts_x.extend(g[dx].values)
            pts_y.extend(g[dy].values)

        # Hairline segments (below the points).
        if segs:
            lc = LineCollection(segs, colors=[color], linewidths=hairline_width,
                                alpha=hairline_alpha, zorder=2,
                                capstyle="round")
            ax.add_collection(lc)

        # Trial endpoints.
        ax.scatter(pts_x, pts_y, s=point_size, color=color,
                   alpha=1, zorder=3, linewidths=0)

        # 3. Bold consensus point for this stimulus.
        cx, cy = consensus.loc[consensus["label"] == lab, [dx, dy]].values[0]
        ax.scatter([cx], [cy], s=ref_size * 10, color=consensus_color,
                   edgecolor="white", linewidth=0.8, zorder=4, marker='*')
        ax.annotate(symdict.get(lab), (cx, cy), fontsize=14, zorder=5, 
                  xytext=(cx + label_offset, cy + label_offset), color=consensus_color)

        ax.set_title(lab)
        # Cosmetics: equal aspect (MDS units are shared), no arbitrary ticks.
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        # ax.set_xticks([])
        # ax.set_yticks([])
        # for spine in ax.spines.values():
        #     spine.set_color(".85")

    # Blank any unused panels.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    return fig, axes


