import sys
if (sys.argv.__len__() < 2):
 print "\n\tusage: mvpa.py <subject>\n\n"
 sys.exit()
else:
 subject = sys.argv[1]

quit()

# Start pymvpa
import numpy as np
import mvpa.suite
from mvpa.suite import *

attributesfile = "/mnt/host/pyMVPA/" + subject + "/mvpaconditions_" + subject + "unix_liking"
myattributes=SampleAttributes(attributesfile)

datafile = "/mnt/host/pyMVPA/" + subject + "/" + subject + "_AllAct_mcf.nii.gz"
maskfile = "/mnt/host/pyMVPA/" + subject + "/IFG_parsO_25_subjspace.nii.gz"
dataset=fmri_dataset(datafile, targets=myattributes.targets, chunks=myattributes.chunks, mask=maskfile)

# Check the shape of the dataset
print dataset.shape

# Detrend the data (remove polynomial trends from the data)
poly_detrend(dataset, polyord=2,chunks_attr="chunks")
 
# Z-score the data (scale all features into approximately the same range, and also remove their mean)
zscore(dataset,param_est=('targets',['Rest','Fixation']))

stimlist = ['Like','Dislike']
dataset = dataset[np.array([l in stimlist for l in dataset.sa.targets])]


# Choose a partitioner
splitter=NFoldPartitioner()

# Choose a classifier algorithm
clf=LinearCSVMC()

# Set up the cross-validation
cvte=CrossValidation(clf,splitter, errorfx=lambda p,t: np.mean(p==t), enable_ca=['stats'])

# Go!
results=cvte(dataset)

# See the mean
accuracy = np.mean(results)

print "Accuracy is %f" % accuracy
# See the confusion matrix
# Cvte.ca.stats.matrix
cvte.ca.stats.plot(numbers=True)
pl.show()