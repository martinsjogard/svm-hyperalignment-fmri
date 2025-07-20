# fMRI MVPA Analysis and Hyperalignment

This repository contains a set of Python scripts for performing multivariate pattern analysis (MVPA) on fMRI data using PyMVPA. The analyses include subject-level classification using SVMs, group-level regularized hyperalignment, and visualization of classification accuracy.

## Repository Contents

### 1. `mvpa_pipeline.py`
Performs subject-level classification using a linear SVM:
- Loads a single subject’s fMRI data and associated condition labels
- Applies preprocessing (detrending, z-scoring)
- Trains and evaluates a classifier using n-fold cross-validation
- Displays classification accuracy and confusion matrix

### 2. `SVM.py`
Extends single-subject analysis by:
- Accepting subject ID, ROI name, and attribute type via command-line
- Running cross-validation on masked brain data
- Logging classification accuracy and saving detailed results to a file
- (Optional) Performs a searchlight analysis and significance testing via permutation tests

### 3. `hyperalignment_regularized_all.py`
Performs group-level **regularized hyperalignment**:
- Aligns neural data from multiple subjects into a shared space
- Sweeps a regularization parameter `alpha` from 0 (CCA) to 1 (standard hyperalignment)
- Evaluates between-subject classification performance across alpha values
- Produces a plot showing accuracy versus alpha

## Requirements

- Python 2.7 or 3.x
- [PyMVPA2](http://www.pymvpa.org/)
- NumPy
- Matplotlib
- nibabel

## Usage

Typical command-line usage examples:

```bash
python mvpa_pipeline.py 101
python SVM.py 103 PHC2 granularity_learn
python hyperalignment_regularized_all.py sub_list.txt singleEnv_learn
```

## Data Structure

Your project directory should be organized as follows:

```
project_root/
├── data/
│   └── <subject_id>/
│       ├── 1_session_mc_brain.nii.gz
│       ├── attributes/
│       │   └── <subject_id>_<attribute>.txt
│       └── masks_in_f_space/
│           └── <subject_id>_<roi>.nii.gz
├── attributes/
│   └── singleEnv_learn_attriblist.txt
├── sub_list.txt
├── mvpa_pipeline.py
├── SVM.py
└── hyperalignment_regularized_all.py
```

---

*These tools are designed for advanced neuroimaging analysis workflows where both within-subject and across-subject comparisons are essential.*