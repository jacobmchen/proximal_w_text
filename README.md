# Proximal Causal Inference With Text Data

Supporting code for the paper Proximal Causal Inference With Text Data by Jacob M. Chen, Rohit Bhattacharya, and Katherine A. Keith. The link to the arXiv pre-print may be found here: https://arxiv.org/abs/2401.06687. Below, we describe the contents of the repository by folder.

There are multiple references in the code to csv files and pickle files that we have deliberately not included in this repository because MIMIC-III is not available to the general public (although it is accessible to anyone as long as they are credentialed). The intent of this repository is to allow readers who already have access to the MIMIC-III dataset to reproduce our work. Credentialed users may access MIMIC-III at https://physionet.org.

*The current repository contains updated experiments. For experiments corresponding to those in the current arXiv pre-print, please consult this previous version of the repository: https://github.com/jacobmchen/proximal_w_text/tree/84ca5aad570f773f5074f65e4b04be6671ce176a. 

### fully_synthetic

This folder contains code for the fully synthetic experiments detailed in Section 5.1. Because the code in this folder does not utilize any data from MIMIC-III, anyone should be able to run it as it is.

### semi_synthetic

This folder contains all the files related to the semi-synthetic experiments detailed in Section 5.2. This includes preprocessing of the MIMIC-III dataset, diagnosing signal in the text data, creating predictions from Flan-T5 and OLMo, and estimation. There is a separate README.md file in this folder with specific details of all the steps we took to run the semi-synthetic experiments.