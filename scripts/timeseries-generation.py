# %%
from nolitsa import surrogates

# %%
import numpy as np

# %%
import pickle

with open('../pickles/timeseries.pickle', 'rb') as f:
    ts = pickle.load(f)

print(ts.shape)

# %%
subjects, brain_regions, ts_length = ts.shape

# %%
surr_count = 5  # 5 surrogates from single timeseries.

# %%
ts_surrogates = np.empty((surr_count * subjects, brain_regions, ts_length))
print(ts_surrogates.shape)

# %%
for s in range(subjects):
    for i in range(surr_count):
        for b in range(brain_regions):
            ts_surrogates[s*surr_count + i][b] = surrogates.aaft(ts[s][b])

# %%
with open('../pickles/timeseries-aaft-5.pickle', 'wb') as f:
    pickle.dump(ts_surrogates, f)

# %%
ts_surrogates = np.empty((surr_count * subjects, brain_regions, ts_length))
print(ts_surrogates.shape)

# %%
for s in range(subjects):
    for i in range(surr_count):
        for b in range(brain_regions):
            ts_surrogates[s*surr_count + i][b] = surrogates.iaaft(
                ts[s][b],
                maxiter=1000, atol=1e-8, rtol=1e-10
            )[0] # Default params.

# %%
with open('../pickles/timeseries-iaaft-5.pickle', 'wb') as f:
    pickle.dump(ts_surrogates, f)

# %%
