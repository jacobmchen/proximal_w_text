# Proximal Causal Inference With Text Data

Supporting code for the paper Proximal Causal Inference With Text Data by Jacob M. Chen, Rohit Bhattacharya, and Katherine A. Keith. The link to the arXiv pre-print may be found here: https://arxiv.org/abs/2401.06687. Below, we describe the contents of the repository by folder.

There are multiple references in the code to csv files and pickle files that we have deliberately not included in this repository because MIMIC-III is not available to the general public (although it is accessible to anyone as long as they are credentialed). The intent of this repository is to allow readers who already have access to the MIMIC-III dataset to reproduce our work. Credentialed users may access MIMIC-III at https://physionet.org.

### fully_synthetic

This folder contains code for the fully synthetic experiments detailed in Section 5. Because the code in this folder does not utilize any data from MIMIC-III, anyone should be able to run it as it is.

### semi_synthetic

<<<<<<< HEAD
This folder contains all the files related to the semi-synthetic experiments detailed in Section 5.2. This includes preprocessing of the MIMIC-III dataset, diagnosing signal in the text data, creating predictions from Flan-T5 and OLMo, and estimation. There is a separate README.md file in this folder with specific details of all the steps we took to run the semi-synthetic experiments.

### text_independence

This folder contains code for creating the tables in the Appendix section titled Empirical Evidence for Text Independence. We use a tfidf vectorizer to evaluate the most common vocabularies of different pairings of note categories and find preliminary qualitative evidence that support text independence based off of the most pertinent features of the text data.
=======
This folder contains all the files related to the semi-synthetic experiments detailed in Section 5. This includes preprocessing of the MIMIC-III dataset, diagnosing signal in the text data, creating predictions from Flan-T5 and OLMo, and estimation. There is a separate README.md file in this folder with specific details of all the steps we took to run the semi-synthetic experiments.
>>>>>>> 0a3fbf389de2a049c441a583f6a2a10565d34c5e
