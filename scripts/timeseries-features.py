# %% General dependencies.
import numpy as np
import pandas as pd
import pickle

# %% Data.
with open('../pickles/timeseries.pickle', 'rb') as f:
    ts = pickle.load(f)

with open('../pickles/test-indices.pickle', 'rb') as f:
    test_indices = pickle.load(f)

# %%
print(f'All timeseries data: {ts.shape}')
print(f'Test size: {len(test_indices)} - {len(test_indices) / ts.shape[0] * 100:.2f} %')

# %% Filter out test data.
ts_train = ts[~np.isin(np.arange(ts.shape[0]), test_indices)]
print(f'Train size: {len(ts_train)} - {len(ts_train) / ts.shape[0] * 100:.2f} %')

# %% Get sample numbers.
train_samples, brain_regions, ts_length = ts_train.shape
print(ts_train.shape)

# %% Convert to dataframe for feature extraction.
df = pd.DataFrame(
    ts_train.transpose([0, 2, 1]).reshape(-1, brain_regions),
    index=np.repeat(np.arange(train_samples), ts_length)
)
df['id'] = df.index

print(df.shape)

# %% Load classification targets.
df_metadata = pd.read_csv(f'../data/patients-cleaned.csv', index_col=0)
df_metadata.head(2)

# %% Filter out test data.
y_train = df_metadata['target'][~np.isin(np.arange(ts.shape[0]), test_indices)].values
print(f'Train size: {len(y_train)} - {sum(y_train) / len(y_train) * 100:.2f} % patients')

# %% Import tsfresh package.
from tsfresh import select_features, extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

# %% Calculate features.
# Note: Parallelization is broken on Windows, hence `n_jobs=0`.
features = extract_features(df, column_id='id', n_jobs=0)

# %% Make copy before imputing.
features_all = features.copy()

# %% Save as it takes long time to run.
features_all.to_csv('../data/timeseries-tsfresh.csv')

# %% Remove all features with NAN's.
features_clean = features.dropna(axis=1)

print(f'Shape before cleaning: {features.shape}')
print(f'Shape after cleaning: {features_clean.shape}')

# %% Remove non-informative features.
features_filtered = select_features(features_clean, y_train, n_jobs=0)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics

# %%
skf = StratifiedKFold(n_splits=7, random_state=42, shuffle=True)
lr = LogisticRegression(random_state=42)

# %%
for train_index, val_index in skf.split(np.zeros(len(y_train)), y_train):
    lr.fit(features_filtered.iloc[train_index, :], y_train[train_index])
    y_pred = lr.predict(features_filtered.iloc[val_index, :])
    y_probs = lr.predict_proba(features_filtered.iloc[val_index, :])

    acc = metrics.accuracy_score(y_train[val_index], y_pred)
    print(acc)
    print(y_probs)
    print('============================\n\n')

# %%
