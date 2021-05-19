# %% Dependencies.
import pandas as pd
import numpy as np
import os
import pickle

# %% Folders.
DATA_FOLDER = '../data'
PICKLE_FOLDER = '../pickles'

# %% Load FC matrices.
with open(f'{PICKLE_FOLDER}/fc-pearson.pickle', 'rb') as f:
    fc_pearson = pickle.load(f)

with open(f'{PICKLE_FOLDER}/fc-spearman.pickle', 'rb') as f:
    fc_spearman = pickle.load(f)

with open(f'{PICKLE_FOLDER}/fc-partial-pearson.pickle', 'rb') as f:
    fc_partial_pearson = pickle.load(f)

# %% Load targets.
df = pd.read_csv(f'{DATA_FOLDER}/patients-cleaned.csv', index_col=0)
df.head()

# %% Define constants.
total_samples, total_brain_regions, _ = fc_pearson.shape
target_index_change = 90

# %% Load test indices.
with open(f'{PICKLE_FOLDER}/test-indices.pickle', 'rb') as f:
    test_indices = pickle.load(f)

train_indices = ~np.isin(np.arange(total_samples), test_indices)

# %% [markdown]
## Generate datasets
#### Run from this point to change FC type.

# %% Set FC type.
corr_type = "spearman"
if corr_type == "pearson":
    fc = fc_pearson.copy()
elif corr_type == "spearman":
    fc = fc_spearman.copy()
elif corr_type == "partial-pearson":
    fc = fc_partial_pearson.copy()

# %% Direction of thresholding.
min_thresholding = True

# %% [markdown]
### 1. Min / max thresholded correlations by average control / patient diffs.

# %% Output folder.
folder = f'/fc-{corr_type}-abs-group-avg-diff'
if not os.path.exists(f'{PICKLE_FOLDER}{folder}'):
    os.makedirs(f'{PICKLE_FOLDER}{folder}')

# %% Average computation (! using train data only !).
control_train_indices = np.hstack(
    [train_indices[:target_index_change],
    np.repeat(False, total_samples-target_index_change)
])
patient_train_indices = np.hstack([
    np.repeat(False, target_index_change),
    train_indices[target_index_change:]
])

avg_fc_control = np.mean(fc[control_train_indices], axis=0)
avg_fc_patient = np.mean(fc[patient_train_indices], axis=0)
avg_fc_diff = np.abs(avg_fc_control - avg_fc_patient)

# %% Generate FC edges at different thresholds.
for min_diff_threshold in [0.01, 0.05, 0.1, 0.15]:

    if min_thresholding:
        fc_diff_thresholded = np.where(np.abs(avg_fc_diff) >= min_diff_threshold, True, False)
    else:
        fc_diff_thresholded = np.where(np.abs(avg_fc_diff) <= min_diff_threshold, True, False)

    fc_diff_tiled = np.tile(fc_diff_thresholded, (df.shape[0], 1, 1))

    fc_binary_thresholded = np.where(fc_diff_tiled, 1, 0)
    fc_real_thresholded = np.where(fc_diff_tiled, fc, 0)

    with open(f"{PICKLE_FOLDER}{folder}/{'min' if min_thresholding else 'max'}-th-{min_diff_threshold}-binary.pickle", 'wb') as f:
        pickle.dump(fc_binary_thresholded, f)

    with open(f"{PICKLE_FOLDER}{folder}/{'min' if min_thresholding else 'max'}-th-{min_diff_threshold}-real.pickle", 'wb') as f:
        pickle.dump(fc_real_thresholded, f)

# %% [markdown]
### 2. Min / max thresholded correlations by absolute sample values.

# %% Output folder.
folder = f'/fc-{corr_type}-abs-sample-diff'
if not os.path.exists(f'{PICKLE_FOLDER}{folder}'):
    os.makedirs(f'{PICKLE_FOLDER}{folder}')

# %% Generate FC edges at different thresholds.
for min_diff_threshold in [0.01, 0.05, 0.1, 0.15]:

    if min_thresholding:
        fc_diff_thresholded = np.where(np.abs(fc) >= min_diff_threshold, True, False)
    else:
        fc_diff_thresholded = np.where(np.abs(fc) <= min_diff_threshold, True, False)

    fc_binary_thresholded = np.where(fc_diff_thresholded, 1, 0)
    fc_real_thresholded = np.where(fc_diff_thresholded, fc, 0)

    with open(f"{PICKLE_FOLDER}{folder}/{'min' if min_thresholding else 'max'}-th-{min_diff_threshold}-binary.pickle", 'wb') as f:
        pickle.dump(fc_binary_thresholded, f)

    with open(f"{PICKLE_FOLDER}{folder}/{'min' if min_thresholding else 'max'}-th-{min_diff_threshold}-real.pickle", 'wb') as f:
        pickle.dump(fc_real_thresholded, f)
