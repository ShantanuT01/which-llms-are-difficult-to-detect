# Which LLMs are Difficult to Detect? 

This provides the source code for our [paper](https://arxiv.org/abs/2410.14875). 

## Datasets
The `datasets/` folder contains the data for the two datasets used in our paper - Deepfake and RIP.
The `test_with_statistics.csv` contains test data with several statistics computed from the `textdescriptives` library.

The RIP Dataset can also be found on [HuggingFace Datasets](https://huggingface.co/datasets/ShantanuT01/RIP-Dataset). 
## Helper Code
This can be found in the `libauc_training` directory. This includes the `TextDataset`  `LibAUCTrainer`, `Experiment` classes. For example usage, see an example Jupyter notebook that uses the `Experiment` class. 

## Experiments
Experiment data - such as predictions and metrics - can be found in the respective folders. Jupyter (Kaggle) notebooks are also present. 
Models can be found in the latest GitHub release as assets. 

## Interpreting Results
The `auc_scores.csv` and `ap_scores.csv` file reports the respective metric based on the blocking factor (LLM model or LLM family). The `factor` column maps to the <i>test set</i> while individual columns correspond to the classifier with a training set containing all texts with that specific factor. 

## Citing
```bibtex
@misc{
  thorat2024llmsdifficultdetectdetailed,
  title={Which LLMs are Difficult to Detect? A Detailed Analysis of Potential Factors Contributing to Difficulties in LLM Text Detection}, 
  author={Shantanu Thorat and Tianbao Yang},
  year={2024},
  eprint={2410.14875},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.14875},
}
```

