# Which LLMs are Difficult to Detect? 

This provides the source code for our paper. 

## Datasets
The `datasets/` folder contains the data for the two datasets used in our paper - Deepfake and RIP.
The `test_with_statistics.csv` contains test data with several statistics computed from the `textdescriptives` library.

The RIP Dataset can also be found on [HuggingFace Datasets](https://huggingface.co/datasets/ShantanuT01/RIP-Dataset). 
## Helper Code
This can be found in the `libauc_training` directory. This includes the `TextDataset`  `LibAUCTrainer`, `Experiment` classes. For example usage, see an example Jupyter notebook that uses the `Experiment` class. 

## Experiments
Experiment data - such as models, predictions, and metrics - can be found in the respective folders. Jupyter (Kaggle) notebooks are also present. 

## Interpreting Results
The `auc_scores.csv` and `ap_scores.csv` file reports the respective metric based on the blocking factor (LLM model or LLM family). The `factor` column maps to the <i>test set</i> while individual columns correspond to the classifier with a training set containing all texts with that specific factor. 

