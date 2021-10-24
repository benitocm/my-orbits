from libc.math cimport pow, sqrt, cos, cosh, abs, sin, sinh


cdef double stump_C(double z) :
    # This function evaluates the Stumpff function C(z) according to Equation 3.52.        
    if z > 0 :
        return (1 - cos(sqrt(z)))/z        
    elif z < 0 :
        return (cosh(sqrt(-z)) - 1)/(-z)
    else :
        return 0.5
        
        
cdef double stump_S(double z) :
    #This function evaluates the Stumpff function C(z) according to Equation 3.53.
    if z > 0:
        sz = sqrt(z) 
        return (sz - sin(sz))/pow(sz,3)
    elif z < 0 :
        s_z = sqrt(-z) 
        # According to the equation the denominatori is pow(sqrt(z),3)
        return  (sinh(s_z) - s_z)/pow(s_z,3)
    else :
        return 0.1666666666666666
        

cdef double prv_kepler_U(double mu, double x , double dt, double ro, double vro, double inv_a, int nMax):
    """
    Compute the general anomaly by solving the universal Kepler
    function  using the Newton's method      

    Args:
        mu : Gravitational parameter (AU^3/days^2)
        dt : time since x = 0 (days)
        ro : radial position (AU) when x=0
        vro: rdial velocity (AU/days) when x=0
        inv_a : reciprocal of the semimajor axis (1/AU)
        nMax : maximum allowable number of iterations

    Returns :
        The universal anomaly (x) AU^.5
    """

    cdef double error = 1.0e-8
    cdef int n = 0
    cdef double ratio = 1.0
    cdef double C 
    cdef double S
    cdef double F 
    cdef double dFdx

    while (abs(ratio) > error) and  (n <= nMax) :
        n = n + 1
        C = stump_C(inv_a*x*x)
        S = stump_S(inv_a*x*x)
        F = ro*vro/sqrt(mu)*x*x*C + (1 - inv_a*ro)*pow(x,3)*S + ro*x - sqrt(mu)*dt
        dFdx = ro*vro/sqrt(mu)*x*(1 - inv_a*x*x*S) + (1 - inv_a*ro)*x*x*C + ro
        ratio = F/dFdx
        x = x - ratio
    return x
        
        
cpdef double kepler_U (double mu, double x , double dt, double ro, double vro, double inv_a, int nMax):
    return prv_kepler_U(mu, x , dt, ro, vro, inv_a, nMax)