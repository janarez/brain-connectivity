# Notable results from experiments

train / test split - 140 / 50 = total 190 subjects

these are validation results

all are 7 fold stratified CV (train / val split - 120 / 20)

fixed random state 42

## Standard ML

- KNN
  - 71 += 6 %
  - 3 neighbors

- Naive Bayes
  - 79 += 9 %

- Random Forest
  - 80 += 10 %

- SVC
  - 85 += 10 %
  - low polynomial kernels
  - 4 folds get 95 % accuracy, 3 get 70 - 75 % accuracy.

- Elastic Net alá logistic regression
  - 87 += 8 %
  - folds are not strictly weak or strong
  - usually, a weak one, one or two middle and then strong

## Fully connected network

- id: 002
  - mean: 85 %
  - std: 5 %
  - ~80k parameters
  - 8 hidden nodes
- id: 003
  - mean: 84 %
  - std: 7 %
  - ~150k parameters
  - 16 hidden nodes

- the concat layer is not necessary, input to the last linear layer can be simply the output of preceding layer without decrease in accuracy

## Graph isomorphism network
