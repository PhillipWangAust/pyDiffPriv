from pyDiffPriv import dpacct
import numpy as np

# This script provides an example how dpacct can be used.

# first we create a DPAcct class object. This is used for doing meta-level DP accountant
acct = dpacct.DPAcct()

# this is used to account for privacy loss associated with a sequence of algorithms run on a data set

# Let's say we want to access the data set by 100 times, each with and (\eps,\delta)-DP algorithm
k=100
eps = 0.01
delta = 1e-8
for i in range(k):
    acct.update_DPlosses(eps,delta)

# we can then query the DP losses through the following two functions
print("privacy loss through naive composition is",acct.get_eps_delta_naive())
print("privacy loss through strong composition (using KOV) is",acct.get_eps(delta*k*1.1),delta*k)


# What if I am adding Gaussian noise and I do not know how to calculate the corresponding eps and delta?
# This is find, if you specify the standard deviation of the noise added w.r.t. the l2 sensitivity
# and your favorite delta, then you get the corresponding DP
sigma = 5
print("The corresponding eps with respect of the gaussian noise is ", dpacct.get_eps_gaussian(sigma,delta))

# the underlying implementation uses the Cumulant Generating Functions (CGF) to calculate
# the optimal order of moment to use and construct a tail bound with the give delta.

# Now we turn to something fancier that allows more efficient composition of heterogeneous mechanisms.

# there is one parameter m to set, and the CGFAcct will keep track of moments of privacy random variables up to order m.

m=500 # for small \eps, m needs to be large for big \eps, m needs to be
delta = 1e-8
delta2 = 1e-6
delta3 = 1e-4
cgfacct = dpacct.CGFAcct(m)
k=1000
sigma = 5 # each
prob=0.01 # sampling probability


eps_seq = []
eps_seq2= []
eps_seq3= []

for i in range(k):
    #cgfacct.update_cgf_gaussian(sigma)
    #cgfacct.update_cgf_laplace(100)
    # cgfacct.update_cgf_randresponse(0.49)
    cgfacct.update_cgf_subsamplegaussian(prob, sigma)

    eps_seq.append(cgfacct.get_eps(delta))
    eps_seq2.append(cgfacct.get_eps(delta2))
    eps_seq3.append(cgfacct.get_eps(delta3))

print("CGF composition of 1000 subsampled Gaussian mechanisms gives ", (cgfacct.get_eps(delta), delta))

#%matplotlib inline


import matplotlib
import matplotlib.pyplot as plt


plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(range(k), eps_seq)
plt.plot(range(k), eps_seq2)
plt.plot(range(k), eps_seq3)

plt.legend(['\delta = 1e-8', '\delta = 1e-6', '\delta = 1e-4'], loc='best')
plt.title('Overall (eps,delta)-DP over composition.')
plt.show()

#print(np.exp(dpacct.get_binom_coeffs(5)))


