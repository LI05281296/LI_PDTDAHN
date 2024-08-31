################################################
### Important notes on external dependencies ###
################################################

1. Our function prediction routine uses the LIBSVM package to train SVM
classifiers. This package is available at:

  https://www.csie.ntu.edu.tw/~cjlin/libsvm/

Please follow the instructions on this website to setup the library and its 
MATLAB interface. Please make sure "svmtrain" and "svmpredict" MEX functions 
are properly compiled and accessible in the MATLAB environment before running
our function prediction code. Our cross_validation.m depends on these two
functions.

2. One of the modes of Mashup (with svd_approx = false) uses an L-BFGS
library written in Fortran, which needs to be complied for the machine you are
using.

We use the MATLAB interface for this library written by Stephen Becker:

  https://www.mathworks.com/matlabcentral/fileexchange/35104-lbfgsb--l-bfgs-b--mex-wrapper

Please follow the instructions provided at this link to setup the library
and ensure that lbfgsb.m in the package above functions properly and is in the
search path when you run Mashup. Our code will make a call to this function
(see Line 15 in vector_embedding.m, which is called by mashup.m).

Note that the SVD version of Mashup does not depend on L-BFGS.

######################################
### Speeding up with multiple CPUs ###
######################################

Our function prediction code lends itself to easy parallelization. While the
provided code assumes access to a single CPU, if your resources allow, you can
replace the for-loops with parallel for-loops (parfor in MATLAB) to gain 
substantial speed-ups. Please search for "parfor" in cross_validation.m and swap
in the commented line of code to achieve this. Note there are two instances of
such edit. Please consult the documentation for parfor for setting up the
environment with appropriate number of cores.

##################
### Need help? ###
##################

Feel free to contact Hoon Cho (hhcho@mit.edu) for questions or comments!
