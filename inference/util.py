import numpy as np
from scipy.optimize import brentq

def ress(w):
    # relative ESS (effective sample size)
    return 1./(np.sum(w**2)*w.size)

def rcess(w,log_likelihood,phi,phi_r_minus_1):
    # TODO: test
    l = np.exp(log_likelihood - np.max(log_likelihood))
    l = l/np.sum(l)
    numerator = np.sum(w*(l**(phi-phi_r_minus_1)))**2
    denominator = np.sum(w*(l**(2*(phi-phi_r_minus_1))))
    # print('numerator', numerator)
    # print('denominator', denominator)
    #print("numerator/denominator",numerator/denominator)
    return numerator/denominator

def nextAnnealingParameter(w,log_likelihood,phi_r_minus_1,alpha):
    # w must be normalised
    # print('rcess(w,log_likelihood,1,phi_r_minus_1)', rcess(w,log_likelihood,1,phi_r_minus_1))
    if rcess(w,log_likelihood,1,phi_r_minus_1) >= alpha:
        return 1
    else:
        func = lambda phi : rcess(w,log_likelihood,phi,phi_r_minus_1) - alpha
        print("func(phi_r_minus_1), func(1)",func(phi_r_minus_1), func(1))
        phi, details = brentq(func,phi_r_minus_1,1,full_output=True)
        # print(details)
        print('phi brentq:',phi,'func(phi)', func(phi))
        # print('func(0)', func(0))
        #
        # print('func(10**(-10))', func(10**(-12)))

        return phi






