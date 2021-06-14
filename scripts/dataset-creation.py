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
corr_type = "partial-pearson"
if corr_type == "pearson":
    fc = fc_pearson.copy()
elif corr_type == "spearman":
    fc = fc_spearman.copy()
elif corr_type == "partial-pearson":
    fc = fc_partial_pearson.copy()

# %% Direction of thresholding.
min_thresholding = False

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

# %% [markdown]
### 3. Edges based on important features for random forest model - Gini importance.
# RF hyperparameters are based on its exploration in `standard-ml.ipynb`.
# In effect it is used as supervised feature selection method that picks
# in a way relevant correlations.

# %% Output folder.
folder = f'/fc-{corr_type}-gini'
if not os.path.exists(f'{PICKLE_FOLDER}{folder}'):
    os.makedirs(f'{PICKLE_FOLDER}{folder}')

# %% Load the importance matrix.
with open(f"{PICKLE_FOLDER}/gini-importance-matrix.pickle", 'rb') as f:
    gini_matrix = pickle.load(f)

gini_matrix = np.repeat(np.expand_dims(gini_matrix, 0), total_samples, 0)

# %% Create gini indexed dataset.
fc_binary = np.where(gini_matrix, 1, 0)
fc_real = np.where(gini_matrix, fc, 0)

with open(f"{PICKLE_FOLDER}{folder}/binary.pickle", 'wb') as f:
    pickle.dump(fc_binary, f)

with open(f"{PICKLE_FOLDER}{folder}/real.pickle", 'wb') as f:
    pickle.dump(fc_real, f)

# %% [markdown]
### 4. Edges based on coefficients from SGD model that performed the best from all standard tried ML methods.
# SGD hyperparameters are based on its exploration in `standard-ml.ipynb`.

# %% Output folder.
folder = f'/fc-{corr_type}-sgd'
if not os.path.exists(f'{PICKLE_FOLDER}{folder}'):
    os.makedirs(f'{PICKLE_FOLDER}{folder}')

# %% Load the importance matrix.
with open(f"{PICKLE_FOLDER}/sgd-coefficients-matrix.pickle", 'rb') as f:
    sgd_matrix = pickle.load(f)

sgd_matrix = np.repeat(np.expand_dims(sgd_matrix, 0), total_samples, 0)

# %% Create gini indexed dataset.
fc_binary = np.where(sgd_matrix, 1, 0)
fc_real = np.where(sgd_matrix, fc, 0)

with open(f"{PICKLE_FOLDER}{folder}/binary.pickle", 'wb') as f:
    pickle.dump(fc_binary, f)

with open(f"{PICKLE_FOLDER}{folder}/real.pickle", 'wb') as f:
    pickle.dump(fc_real, f)

# %% [markdown]
## Directional datasets.
# Position (i, j) represents edge from i to j.

# %% Set FC type.
corr_type = "spearman"
if corr_type == "pearson":
    fc = fc_pearson.copy()
elif corr_type == "spearman":
    fc = fc_spearman.copy()
elif corr_type == "partial-pearson":
    fc = fc_partial_pearson.copy()

# %% Direction of thresholding.
min_thresholding = False

# %% [markdown]
### 5. kNN min / max correlations by average control / patient diffs.

# %% Output folder.
folder = f'/fc-{corr_type}-knn-abs-group-avg-diff'
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
for knn in [3, 5, 7, 10, 15, 20, 40]:

    fc_knn = np.zeros((total_brain_regions, total_brain_regions), dtype=bool)

    if min_thresholding:
        knn_index = np.argsort(avg_fc_diff)[:,-knn:]
    else:
        knn_index = np.argsort(avg_fc_diff)[:,:knn]

    fc_knn[np.repeat(np.arange(total_brain_regions), knn), np.reshape(knn_index, -1)] = True
    fc_diff_tiled = np.tile(fc_knn, (df.shape[0], 1, 1))

    fc_binary_thresholded = np.where(fc_diff_tiled, 1, 0)
    fc_real_thresholded = np.where(fc_diff_tiled, fc, 0)

    with open(f"{PICKLE_FOLDER}{folder}/{'large' if min_thresholding else 'small'}-knn-{knn}-binary.pickle", 'wb') as f:
        pickle.dump(fc_binary_thresholded, f)

    with open(f"{PICKLE_FOLDER}{folder}/{'large' if min_thresholding else 'small'}-knn-{knn}-real.pickle", 'wb') as f:
        pickle.dump(fc_real_thresholded, f)

# %% [markdown]
### 6. kNN min / max correlations by absolute sample values.

# %% Output folder.
folder = f'/fc-{corr_type}-knn-abs-sample-diff'
if not os.path.exists(f'{PICKLE_FOLDER}{folder}'):
    os.makedirs(f'{PICKLE_FOLDER}{folder}')

# %% Generate FC edges at different thresholds.
for knn in [3, 5, 7, 10, 15, 20, 40]:

    fc_knn = np.zeros((total_samples, total_brain_regions, total_brain_regions), dtype=bool)

    if min_thresholding:
        knn_index = np.argsort(np.abs(fc))[:,:,-knn:]
    else:
        knn_index = np.argsort(np.abs(fc))[:,:,:knn]

    fc_knn[
        np.repeat(np.arange(total_samples), knn*total_brain_regions),
        np.repeat(np.arange(total_brain_regions), knn*total_samples),
        np.reshape(knn_index, -1)
    ] = True

    fc_binary_thresholded = np.where(fc_knn, 1, 0)
    fc_real_thresholded = np.where(fc_knn, fc, 0)

    with open(f"{PICKLE_FOLDER}{folder}/{'large' if min_thresholding else 'small'}-knn-{knn}-binary.pickle", 'wb') as f:
        pickle.dump(fc_binary_thresholded, f)

    with open(f"{PICKLE_FOLDER}{folder}/{'large' if min_thresholding else 'small'}-knn-{knn}-real.pickle", 'wb') as f:
        pickle.dump(fc_real_thresholded, f)
