# Proximal Causal Inference With Text Data

Supporting code for the paper Proximal Causal Inference With Text Data by Jacob M. Chen, Rohit Bhattacharya, and Katherine A. Keith. The link to the arXiv pre-print may be found here: https://arxiv.org/abs/2401.06687. Below, we describe the contents of the repository by folder.

There are multiple references in the code to csv files and pickle files that we have deliberately not included in this repository because MIMIC-III is not available to the general public (although it is accessible to anyone as long as they are credentialed). The intent of this repository is to allow readers who already have access to the MIMIC-III dataset to reproduce our work. Credentialed users may access MIMIC-III on https://physionet.org.

### MIMIC-III_processing

This folder contains code that transforms the original data tables in MIMIC-III into a central csv file that we use in subsequent analyses.

### fully_synthetic

This folder contains code for the fully synthetic experiments detailed in Section 5.1. Because the code in this folder does not utilize any data from MIMIC-III, anyone should be able to run it as it is.

### semi_synthetic

This folder contains code for the semi-synthetic experiments detailed in Section 5.2 along with implementations of the two-stage linear regression estimator for proximal causal inference. We also include here the code and prompt we used to make zero-shot predictions from Google's Flan-T5 large language model.

### plots

This folder contains code for creating the plot shown in Figure 3.