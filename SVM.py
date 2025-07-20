# ! /usr/bin/python

# ####importing libraries
import sys
import numpy as np
import nibabel
import mvpa2.suite
from mvpa2.suite import *

# Subject
subject = sys.argv[1]
subject = sys.argv[1]
roi = sys.argv[2]
attribute_type = sys.argv[3]
attribute_sub_list=attribute_type.split("_")
fmri_condition = attribute_sub_list[1]
measure = attribute_sub_list[0]

# Definitions
root_dir = "/media/fmri/Ubuntu_data/landmarks/pyMVPA/data/"
mask_dir = "masks_in_f_space/"
mask_file = subject + "_1_PHC2_pos_pos.nii.gz"
attr_dir = "attributes/"
attr_suffix = "_main.txt"
nifti= "1_session_mc_brain.nii.gz"
# Nb, remember to specify z-contrast and what subset of the data to be analyzed
# Nb. condition failed is excluded for the garnularity analysis (see line 80), should we remove medium-grained as well?

# Define directories and suffixes
work_dir = os.getcwd()
data_dir = "data"
mask_dir = "masks_in_f_space/"
mask_file = subject + "_1_" + roi + ".nii.gz"
attr_dir = "attributes/"
# Attr_suffix = "_main.txt"
attr_suffix = "_" + attribute_type + ".txt"
nifti = "1_session_mc_brain.nii.gz"
output_dir = work_dir + "/" + "SVM" + "/" + attribute_type
output_file = "accuracy.txt" # For all subjects
# Nb, remember to specify z-contrast and what subset of the data to be analyzed

# Create directories if they do not exist
if not os.path.exists( work_dir + "/" + "SVM" ):
    os.makedirs( work_dir + "/" + "SVM" )

if not os.path.exists( output_dir ):
    os.makedirs( output_dir )

# Load your text file with the attributes listed.
attributesfile = work_dir + "/" + data_dir + "/" + subject + "/" + attr_dir + subject + attr_suffix
myattributes = SampleAttributes(attributesfile)
# Print np.unique(myattributes.targets)
# Print "number of runs: %r" % np.unique(myattributes.chunks)

# Load the dataset and assign targets, chunks, and a mask
datafile = work_dir + "/" + data_dir + "/" + subject + "/" + nifti
maskfile = work_dir + "/" + data_dir + "/" + subject + "/" + mask_dir + mask_file
ds = fmri_dataset(datafile, targets=myattributes.targets, chunks=myattributes.chunks, mask=maskfile)

# Check the shape of the dataset
# Print dataset.shape

# Detrend the data 
""" (Fitting a straight line to the time series of each voxel via linear regression and taking the residuals as the new feature values) """
poly_detrend(ds, polyord=2,chunks_attr="chunks")
 
# Z-score the data 
""" (Scale all features into approximately the same range, and also remove their mean --> get a per time-point voxel intensity difference from the rest average. """
zscore(ds,param_est=('targets', ['odd_even']), dtype='float32')

# Remove odd_even (rest condition) from the dataset 
ds = ds[ds.sa.targets != 'odd_even']
ds = ds[ds.sa.targets != 'dummy']
if fmri_condition == "learn":
   	ds = ds[ds.sa.targets != 'fix']
	if measure == "granularity":
		ds = ds[ds.sa.targets != 'learn_failed']	
elif fmri_condition == "fix":
	ds = ds[ds.sa.targets != 'learn']
	if measure == "granularity":
		ds = ds[ds.sa.targets != 'fix_failed']	

# Analyzing only a subset of the data
# Stimlist = ['learning', 'fix']
# Ds = ds[np.array([l in stimlist for l in ds.sa.targets])]

# Choose a partitioner
splitter=NFoldPartitioner()

# Choose a classifier algorithm
clf = LinearCSVMC() # Support vector machine (svm)
 
# Set up the cross-validation (split the data to test classifier on new unseen data)
cvte = CrossValidation(clf,splitter, errorfx=lambda p,t: np.mean(p==t), enable_ca=['stats'])

# Go!
results = cvte(ds)

# See the mean accuracy (the classification performance assessed by comparing predictions to the target labels.)
accuracy = np.mean(results)
accuracy = round(accuracy, 2)
# Print "accuracy is %f" % accuracy
with open(output_dir + "/" + output_file, 'a') as text_file:
    text_file.write(subject + " " + roi + " " + str(accuracy) + '\n')
# Print results.samples #print the average accuracy per run

# Output a comprehensive summary of the performed analysis.
analysis = cvte.ca.stats.as_string(description=True)
with open(output_dir + "/" + subject + "_" + roi + ".txt", "w") as text_file:
    text_file.write(analysis)
# See the confusion matrix
# Print cvte.ca.stats.matrix

sys.exit()

# Searchlight analysis
sl = sphere_searchlight(cvte, radius=3, postproc=mean_sample()) # Each sphere has a 3 voxel radius
res = sl(ds)
print res

# -------------------------------------significance testing--------------------------------------------
# Monte carlo shuffling (see doc/examples/permutation_test.py)
permutator = AttributePermutator('targets', count=5000) # Permute the target attribute of the dataset for each iteration.
distr_est = MCNullDist(permutator, tail='left', enable_ca=['dist_samples']) # Shuffle targets-> p-value from left tail of  *null* distribution
cv = CrossValidation(clf, partitioner,
                     errorfx=mean_mismatch_error,
                     postproc=mean_sample(),
                     null_dist=distr_est,
                     enable_ca=['stats'])
err = cv(ds)

print 'CV-error:', 1 - cv.ca.stats.stats['ACC']
p = cv.ca.null_prob
# Should be exactly one p-value
assert(p.shape == (1,1))
print 'Corresponding p-value:',  np.asscalar(p)

# Make new figure
pl.figure()
# Histogram of all computed errors from permuted data
pl.hist(np.ravel(cv.null_dist.ca.dist_samples), bins=20)
# Empirical error
pl.axvline(np.asscalar(err), color='red')
# Chance-level for a binary classification with balanced samples
pl.axvline(0.5, color='black', ls='--')
# Scale x-axis to full range of possible error values
pl.xlim(0,1)
pl.xlabel('Average cross-validated classification error')
# ---------------------------------------------------------------------------------------------------------