# Datasets

To run the experiments, the following two files must be put inside this directory.

## eso190_aal_strin_090_avg.mat

Contains mean timeseries for 190 subjects from 90 regions of interest.
This is an unbalanced sample with 100 patients and 90 controls.

(From these 180 subjects were selected to obtain sample (the unused correlation data) balanced in age and sex, however JH cannot now quickly find the selection key.)

## patients-metadata.csv

(converted eso190_covs_20180716_1545.xlsx to `.csv` format)

Contains metadata:
 - folder name - includes the unique identifier: ESO_C are healthy controls, ESO_P are patients
 - sex
 - age
