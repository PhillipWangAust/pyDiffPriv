# This is the main file of the python package dp-acct
# that keeps track of the privacy loss of a sequence of randomized algorithms.


import numpy as np
# from scipy.optimize import fmin_bfgs


# utiltiy function for calculating the combinatorials
# save log(x+1).

def stable_logsumexp(x):
    a = np.max(x)
    return a+np.log(np.sum(np.exp(x-a)))
def stable_logsumexp_two(x,y):
    a = np.maximum(x,y)
    return a + np.log(np.exp(x-a) + np.exp(y-a))


def get_binom_coeffs(sz):
    C = np.zeros(shape = (sz + 1, sz + 1));
    #for k in range(1,sz + 1,1):
    #    C[0, k] = -np.inf
    for n in range(sz + 1):
        C[n, 0] = 0  # 1
    for n in range(1,sz + 1,1):
        C[n, n] = 0
    for n in range(1,sz + 1,1):
        for k in range(1,n,1):
            # numerical stable way of implementing the recursion rule
            C[n, k] = stable_logsumexp_two(C[n - 1, k - 1],C[n - 1, k])
    # only the lower triangular part of the matrix matters
    return C





# Get the eps and delta for a single Gaussian mechanism
def get_eps_gaussian(sigma, delta):
    """ This function calculates the eps for Gaussian Mech given a delta"""

    lamb = (np.log(1 / delta)*2)**0.5 * sigma

    # The following block solves for lamb numerically, in case it is not analytically tractable
    #def F(lamb):
    #    return (np.log(1 / delta) + 0.5 / sigma ** 2 * lamb * (lamb + 1)) / lamb
    #def G(x):
    #    return -np.log(1 / delta)/ x **2 +  0.5 / sigma ** 2
    #lamb = fmin_bfgs(F, 5, G, disp= True)
    return (np.log(1 / delta) + 0.5 / sigma ** 2 * lamb*(lamb+1)) / lamb


class DPAcct:
    """A class that keeps track of (eps,delta) of all mechanisms that got run so far"""
    #
    #DPlosses = []

    def __init__(self):
        self.DPlosses = []
        self.eps_state1 = 0
        self.eps_state2 = 0
        self.eps_state3 = 0
        self.delta_state = 0
        self.delta_state2 = 0

    def update_DPlosses(self,eps,delta):
        self.DPlosses.append([eps,delta])
        self.eps_state1 += eps
        self.eps_state2 += (np.exp(eps) - 1) * eps / (np.exp(eps) + 1)
        self.eps_state3 += eps ** 2
        self.delta_state += np.log(1-delta)
        self.delta_state2 += delta
        #update the optimal DPlosses here?

    def get_eps_delta_naive(self):
        return self.eps_state1, self.delta_state2

    def get_minimum_possible_delta(self):
        return 1-np.exp(self.delta_state)

    def get_eps(self,delta):
        """ make use of KOV'15 to calculate the composition for hetereogeneous mechanisms"""
        # TODO: further improve upon this with Salil Vadhan's approximation algorithm
        if delta>1 or delta<0:
            print("The chosen delta is clearly not feasible.")
            return -1
        if delta is 0:
            if self.delta_state2 is 0:
                return self.delta_state2
            else:
                print("Pure DP is not feabile. Choose non-zero delta")
                return -1

        # 1-delta =  (1-deltatilde) exp(state)
        deltatilde = 1 - np.exp(np.log(1-delta) -self.delta_state)
        if deltatilde <= 0:
            print("The chosen delta is not feasible. delta needs to be at least ", 1-np.exp(self.delta_state))
            return -1

        eps1 = self.eps_state1
        eps2 = self.eps_state2 + (self.eps_state3 *2 *  np.log(np.exp(1) + self.eps_state3 ** 0.5 / deltatilde ))**0.5
        eps3 = self.eps_state2 + (2*self.eps_state3*np.log( 1/ deltatilde))**0.5

        return np.minimum(np.minimum(eps1,eps2),eps3)


class CGFAcct:
    """A class that keep tracks of the cumulative generating functions of the privacy loss R.V."""
    """This is most general """
    # input to this algorithm is a sequence of lambdas we want to keep track of.
    # 1-100 sounds like a reasonable range that we want to keep track of


    def __init__(self,m):
        self.m=m
        self.lambs = np.linspace(1, self.m, self.m).astype(int)
        self.CGFs = np.zeros_like(self.lambs,float)
        self.CGF_inf = .0
        self.logBinomC = get_binom_coeffs(self.m + 1)
        self.cache = dict()
        # Interface for getting



    def get_eps(self, delta): # find the smallest eps that corresponds to a delta through a tail bound
        #print(self.CGFs)
        if delta<0 or delta > 1:
            print("Error! delta is a probability and must be between 0 and 1")
        if delta is 0:
            return self.CGF_inf
        else:
            #print(np.argmin((np.log(1 / delta) + self.CGFs) / self.lambs))
            return np.min((np.log(1 / delta) + self.CGFs) / self.lambs)




    # How about precomputing it?
    def update_cgf_subsamplegaussian(self, prob, sigma):
        """ update CGF for a subsampleGaussian mechanism """
        # This is for probability of sampling and then adding gaussian noise
        # with a normalized noise level of sigma
        # prob needs to be a floating point number in (0,1)

        # The first term is 1, the second term is prob^2*lamb*(lamb+1)
        # The third term onwards up to lamb+1 term is
        # {lamb+1\choose j} * prob^j[1-e^{-sigma^2/2}]^je^{(j^2+1)sigma^2/2}

        # term2 = prob ** 2 * (np.exp(sigma ** 2) - np.exp(-sigma ** 2));

        if (prob,sigma) in self.cache:
            self.CGFs += self.cache[(prob, sigma)]
            CGF_inf = np.inf
            return

        m = np.max(self.lambs)
        j = np.arange(3, m+2)
        logterm3plus = j * (np.log(prob) + np.log(1 - np.exp(-1/sigma ** 2 / 2))) + (j ** 2 + 1) / (sigma ** 2) / 2
        # make sure that lambs is n dimensional.
        results = np.log(1 + prob ** 2 * (np.exp(1/sigma ** 2) - np.exp(-1/sigma ** 2)) * self.lambs*(self.lambs+1))

        for lamb in range(3, m+1):
            tmp = stable_logsumexp(logterm3plus[0:lamb - 2] + self.logBinomC[lamb + 1, 3:lamb + 1])
            results[lamb-1] = stable_logsumexp_two(results[lamb-1], tmp)

        # cache the calculations so there's no need to do it again..
        self.cache[(prob,sigma)] = results
        self.CGFs += results
        self.CGF_inf = np.inf

    def update_cgf_gaussian(self,sigma):
        """ update CGF for a Gaussian mechanism """
        results = 0.5/sigma**2 * (self.lambs+1) * self.lambs

        self.CGFs += results
        self.CGF_inf = np.inf
        # in this cae everything can be calculated exactly

    def update_cgf_laplace(self,sigma):
        """ update CGF for a Laplace mechanism """
        # sigma is the lambda of Laplace distribution
        results = np.zeros_like(self.CGFs)
        results[0] = 1/sigma + np.exp(-1/sigma)-1
        results[1:] = 1/self.lambs[1:] * np.log((self.lambs[1:]+1)/(2*self.lambs[1:]+1)*np.exp(self.lambs[1:]/sigma)
                                                + self.lambs[1:]/(2*self.lambs[1:]+1)*np.exp(-(self.lambs[1:]+1)/sigma))
        results[:] = results*self.lambs
        self.CGFs += results
        self.CGF_inf += 1/sigma

#    def update_cgf_exponential(self,sigma):
#        """ update CGF for a Exponential mechanism with an insensitive cost function"""
#        print("Not implemented yet\n")

    def update_cgf_randresponse(self,p):
        """ update CGF for a Randomized response of {0,1} variable """
        # Randomized response toggle the response with probability p
        results = np.zeros_like(self.CGFs)
        results[0] = (2*p-1)*np.log(p/(1-p))
        results[1:] = 1/self.lambs[1:] * stable_logsumexp_two((self.lambs[1:]+1)*np.log(p)-self.lambs[1:]*np.log(1-p),
                                                              (self.lambs[1:]+1)*np.log(1-p) - self.lambs[1:]*np.log(p))
        results = results * self.lambs
        self.CGFs += results
        self.CGF_inf += np.abs(np.log((p/(1-p))))


# TODO: Getting rid of the dependence on parameter m
# TODO: Do an accounting of analytical functions of these moments by solve a 1D nonlinear optimization each time  we query for \eps with a fixed \delta.
# TODO:
