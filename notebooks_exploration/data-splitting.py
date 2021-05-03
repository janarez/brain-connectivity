# %% Data splitting script.
import pandas as pd
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import pickle

# %%
DATA_FOLDER = '../data'
df_metadata = pd.read_csv(f'{DATA_FOLDER}/patients-cleaned.csv', index_col=0)

# %%
df_metadata.head(3)

# %%
patient_indexes = df_metadata[df_metadata['target'] == 1].index
control_indexes = df_metadata[df_metadata['target'] == 0].index

# %%
print(f'# patients: {len(patient_indexes)}')
print(f'# controls: {len(control_indexes)}')

# %% Let's assess model uncertainty using binomial distribution on set of test
# sizes and possible classification accuracies.

fig, ax = plt.subplots(1, 1)
test_sizes, test_accuracies = [30, 40, 50, 60], [0.7, 0.8, 0.9]

for n in test_sizes:
    for p in test_accuracies:
        mean, var = binom.stats(n, p, moments='mv')

        x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))

        ax.plot(x / n, binom.pmf(x, n, p), '-o', ms=2, lw=1, label=f'{n}-{p}')
        print(f'n={n}, p={p}: left={(mean-var)/n:.2f}, right={(mean+var)/n:.2f}, width={2*var/n:.2f}')

ax.legend(loc='best')
plt.show()

# %% We can see that increasing test size has positive effect on the confidence
# of our results. We select 50 subjects into test size, 25 controls and 25 patients.

test_patients = np.random.choice(patient_indexes, size=25, replace=False)
test_controls = np.random.choice(control_indexes, size=25, replace=False)

test_indices = np.hstack([test_patients,test_controls])

# %%
print(f'# test patients: {len(np.unique(test_patients))}')
print(f'# test controls: {len(np.unique(test_controls))}')
print(f'# test total: {len(np.unique(test_indices))}')

# %% We save these for use across all experiments.
PICKLE_FOLDER = '../pickles'

# Do not call again! Already saved.
# with open(f'{PICKLE_FOLDER}/test-indices.pickle', 'wb') as f:
#     pickle.dump(test_indices, f)

# %%
