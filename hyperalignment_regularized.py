# !/usr/bin/env python
# Emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# Vi: set ft=python sts=4 ts=4 sw=4 et:
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


from mvpa2.suite import *
verbose.level = 2


verbose(1, "Loading data...")

# ####################################################################
# #### creating the data files ##################################

import numpy as np
subjects = sys.argv[1] # Give subject list (txt file "sub_list") with list of subject ids as argument
attribute = sys.argv[2] # Give a list (attributes_list_hyperalignment.txt), where each line indicates what subject specific attributes file (f.ex. {$subid}/attributes/{$subid}_singleenv_learn.txt) to be used for each analysis. 
print attribute


work_dir = os.getcwd()
data_dir = "data"
attr_dir = "attributes"
nifti = "1_session_mc_brain.nii.gz"
# Output_all = work_dir + "/" + "hyperalignment" + "/" "hypalout.txt"
# Output_one = work_dir + "/" + "hyperalignment" + "/" + "hyp_one.txt"
mask_dir = "masks_in_f_space"
mask_file = "wholebrain_mask.nii.gz"
atts = work_dir + "/" + "singleEnv_learn_attriblist.txt"
# Now a loop over subjects in subject list to store them in dictionary so they can be picked out later

# Write_singles = open(output_one,"r+") #this stores the subjects' dataset info in subsequent lines in a text file (to check it's going through)
with open(subjects, 'r') as f:
    subject_nr=0
    datadictionary = {}
    for subject in f.readlines():
        subject = subject.strip()
        print subject
        subject_nr = subject_nr + 1
        datafile = work_dir + "/" + data_dir + "/" + subject + "/" + nifti
        attributesfile = work_dir + "/" + data_dir + "/" + subject + "/" + attr_dir + "/" + subject + "_" + attribute + ".txt"
        myattributes = SampleAttributes(attributesfile) # Are the attributes now sorted...?
        roi = work_dir + "/" + data_dir + "/" + subject + "/" + mask_dir + "/" + mask_file
        ds = fmri_dataset(datafile, targets=myattributes.targets, chunks=myattributes.chunks, mask=roi)
        print ds
        regs = [line.strip() for line in open(atts, 'r')]
        print regs
        pe = mvpa2.mappers.glm.GLMMapper(ds, regs) # Outputs columns with the attributes from the regs file and a corrresponding parameter estimate for each attribute 
        print pe

        datadictionary['%s' % (subject)] = ds
        # Print datadictionary
        sys.exit()
        # With open(atts,'r') as a:          
            # Regs = a         
            # #print ds
            # Pe=mvpa2.mappers.glm.glmmapper(ds, regs) #outputs columns with the attributes from the regs file and a corrresponding parameter estimate for each attribute 
            # Print pe       
            # Datadictionary['%s' % (subject)] = ds   
            # Write_singles.write(str(pe) + '\n')

# Write_singles.close() #close the file

# Print datadictionary

# Write_all = open(output_all,"r+") #open txt file to store datadict in, not really necessary but just to check that all subs are in it
# Write_all.write(str(datadictionary))
# Write_all.close()

# Now define simple names for each entry (each subject) in dictionary
# Ds1 = datadictionary['101']
ds2 = datadictionary['103']
ds3 = datadictionary['104']
ds4 = datadictionary['105']

# Now store those subject datasets in h5 file
h5save('/tmp/out.h5',[ds2,ds3,ds4])

print 'h5saved'

# ###########################################################
# ######regularized hyperalignment############################

# Below i print a bunch of stuff to tell me where in the analysis we are (for troubleshooting)

ds_all = h5load('/tmp/out.h5')
print 'h5loaded'
# Zscore all datasets individually
_ = [zscore(ds) for ds in ds_all]
print 'zcored'
# Inject the subject id into all datasets
for i,sd in enumerate(ds_all):
    sd.sa['subject'] = np.repeat(i, len(sd))
print 'injected IDs'
# Number of subjects
nsubjs = len(ds_all)
print 'subj number'
# Number of categories
ncats = len(ds_all[0].UT)
print 'number categories'
# Number of run
nruns = len(ds_all[0].UC)
print 'number runs'
verbose(2, "%d subjects" % len(ds_all))
verbose(2, "Per-subject dataset: %i samples with %i features" % ds_all[0].shape)
verbose(2, "Stimulus categories: %s" % ', '.join(ds_all[0].UT))
""" --------------------------------------------------------------------------------- """

""" ---------Creating building blocks for the Regularized Hyperalignment analysis---------------------- """
""" Now we'll create a couple of building blocks for the intended analyses. We'll use a linear SVM classifier, and perform feature selection with a simple one-way ANOVA selecting the ``nf`` highest scoring features. """
# Use same classifier
clf = LinearCSVMC()
# Feature selection helpers
nf = 100
fselector = FixedNElementTailSelector(nf, tail='upper',
                                      mode='select',sort=False)
sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,
                                        enable_ca=['sensitivities'])
# Create classifier with automatic feature selection
fsclf = FeatureSelectionClassifier(clf, sbfs)
""" --------------------------------------------------------------------------------- """

""" ------------------------------Regularized Hyperalignment-------------------------- """
""" According to :ref:`Xu et al. 2012 <XLR2012>`, Hyperalignment can be reformulated to a regularized algorithm that can span the whole continuum between `canonical correlation analysis (CCA)`_ and regular hyperalignment by varying a regularization parameter (alpha).  Here, we repeat the above between-subject hyperalignment and classification analyses with varying values of alpha from 0 (CCA) to 1.0 (regular hyperalignment).  .. _`canonical correlation analysis (CCA)`: http://en.wikipedia.org/wiki/Canonical_correlation  The following code is essentially identical to the implementation of between-subject classification shown above. The only difference is an addition ``for`` loop doing the alpha value sweep for each cross-validation fold. """

alpha_levels = np.concatenate(
                    (np.linspace(0.0, 0.7, 8),
                     np.linspace(0.8, 1.0, 9)))
# To collect the results for later visualization
bsc_hyper_results = np.zeros((nsubjs, len(alpha_levels), nruns))
# Same cross-validation over subjects as before
cv = CrossValidation(clf, NFoldPartitioner(attr='subject'), 
                     errorfx=mean_match_accuracy)
print "141"
# Leave-one-run-out for hyperalignment training
for test_run in range(nruns):
    # Split in training and testing set
    ds_train = [sd[sd.sa.chunks != test_run,:] for sd in ds_all]
    ds_test = [sd[sd.sa.chunks == test_run,:] for sd in ds_all]

    print "aova next"
# Nb!! we've tended to get problems in next step (anova), with analysis essentially freezing up
    # Manual feature selection for every individual dataset in the list
    anova = OneWayAnova()
    print "fscores"
    fscores = [anova(sd) for sd in ds_train]
    featsels = [StaticFeatureSelection(fselector(fscore)) for fscore in fscores]
    ds_train_fs = [featsels[i].forward(sd) for i, sd in enumerate(ds_train)]
    print "alpha next"
    for alpha_level, alpha in enumerate(alpha_levels):
        hyper = Hyperalignment(alignment=ProcrusteanMapper(svd='dgesvd',
                                                           space='commonspace'),
                               alpha=alpha)
        hypmaps = hyper(ds_train_fs)
        ds_test_fs = [featsels[i].forward(sd) for i, sd in enumerate(ds_test)]
        ds_hyper = [ hypmaps[i].forward(sd) for i, sd in enumerate(ds_test_fs)]
        ds_hyper = vstack(ds_hyper)
        print "zscoreinanova"
        zscore(ds_hyper, chunks_attr='subject')
        res_cv = cv(ds_hyper)
        bsc_hyper_results[:, alpha_level, test_run] = res_cv.samples.T

""" Now we can plot the classification accuracy as a function of regularization intensity. """

print "173"
bsc_hyper_results = np.mean(bsc_hyper_results, axis=2)
pl.figure()
plot_err_line(bsc_hyper_results, alpha_levels)
pl.xlabel('Regularization parameter: alpha')
pl.ylabel('Average BSC using hyperalignment +/- SEM')
pl.title('Using regularized hyperalignment with varying alpha values')

if cfg.getboolean('examples', 'interactive', True):
    # Show all the cool figures
    pl.show()

# Line below is just something i made to define a plotting place, but never used.

# Plotplace = work_dir + "/" + "hyperalignment" + "/" + attribute + "/" + "hyperplot.jpg"

sys.exit()