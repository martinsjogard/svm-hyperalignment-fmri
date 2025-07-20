# Regularized Hyperalignment Pipeline

This repository contains a Python script to perform **Regularized Hyperalignment** on multi-subject fMRI data using [PyMVPA](http://www.pymvpa.org/). The pipeline aligns neural activation patterns into a common representational space and evaluates classification performance as a function of a regularization parameter \( \alpha \), which interpolates between **Canonical Correlation Analysis (CCA)** and standard **Hyperalignment**.

## Features

- Loads subject-specific fMRI and attribute data
- Applies a GLM model using sample attributes
- Performs feature selection with one-way ANOVA
- Runs cross-validated regularized hyperalignment
- Evaluates classification performance using a linear SVM
- Plots accuracy across a range of \( \alpha \) values

## Requirements

- Python 2.7 or 3.x
- [PyMVPA 2](http://www.pymvpa.org/)
- NumPy
- Matplotlib
- A configured environment with fMRI data in the expected directory structure

## Usage

Run the script from the command line:
```bash
python hyperalignment_regularized_all.py sub_list.txt singleEnv_learn
```

- `sub_list.txt`: A text file with one subject ID per line.
- `singleEnv_learn`: The name of the attribute file suffix (e.g., `subjID_attributes/subjID_singleEnv_learn.txt`)

## Directory Structure

```
project_root/
├── data/
│   └── subjID/
│       ├── 1_session_mc_brain.nii.gz
│       ├── masks_in_f_space/wholebrain_mask.nii.gz
│       └── attributes/subjID_singleEnv_learn.txt
├── attributes/
│   └── singleEnv_learn_attriblist.txt
├── sub_list.txt
└── hyperalignment_regularized_all.py
```

## Output

- Accuracy vs. \( \alpha \) plot showing classification performance under different regularization strengths.
- Optional: aligned datasets saved in `/tmp/out.h5`.

## Notes

- Edit the script to remove `sys.exit()` and allow full subject list processing.
- Hardcoded subject IDs (`'103'`, `'104'`, `'105'`) can be made dynamic for flexibility.
- Results are visualized but not automatically saved—modify as needed for persistent output.

---

*Author: Adapted for GitHub from research workflow using PyMVPA*