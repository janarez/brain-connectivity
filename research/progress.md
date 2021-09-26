# Brain Connectivity Diploma Thesis

## Goal

Binary classification of schizophrenia on data from brain functional magnetic
resonance data. We have 190 fMRI from 90 controls and 100 patients. Each fMRI
consists of 400 point time series, one per 90 brain regions.

## Approach

These data have been studied before using standard statistical methods and
machine learning. Accuracy of these methods is around 75 %. Our aim is to
improve upon these results using geometric deep learning. We presume that there
are connections between the 90 brain regions. We represent these as a graph on
90 vertices and look for best edge assignment such that the classification task
can be accurately solved by graph convolutional networks.

## Data

### Loading

We use `scipy` to load the matlab data right into 3D numpy array (`[patient, region, series]`). Additionally, we have each subjects target (patient / control), age and sex.

### Preprocessing
