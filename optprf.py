#  
#  Aim of following code is to optimize paramters of recent change of selective constraints in a population for a given site frequency spectum,RN/RS, and theta
#  Citation :
#  
# Implementation of Poisson Random Field is based on following references:
# Simultaneous inference of selection and population growth from patterns of variation in the human genome. Williamson et al. 2005 PNAS
# Inferring the Joint Demographic History of Multiple Populations from Multidimensional SNP Frequency Data. Gutenkunst et al. 2007. PLoS Genetics 
#   
#   

import os, sys, copy
import numpy as np
import scipy
import random
from scipy import stats
import pickle
from operator import mul    # 
from fractions import Fraction
from scipy.optimize import basinhopping

import hyperopt
from hyperopt import fmin, tpe, hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.linalg as la

import copy
#import numba

import pandas as pd

# on gen-comp1
if 1 :
    path2dat = "."
    path2out = "."
    sys.path.append('.')

import pyximport; pyximport.install()
import cython
import TDMA


if 1:
    argvs = sys.argv
    inx = int( argvs[ 1 ] )
    #niter = int( argvs[ 2 ] )
    #argvs = sys.argv
    #tt = int( argvs[ 1 ] )

# load SFS array
# f = open( path2dat + "sim_720m_n1.pkl" )
# obsfnfs = pickle.load( f )



print inx
#inx = 2000


best_setup = {'grate': 615.2792203108594,
 'gridsize': 36605.49334857545,
 'gshape': 23.311326464854073}


#xx = sfs_grid( best_setup[ "gridsize"], 0, best_setup["gshape"] )
#xx = sfs_grid( best_setup[ "gridsize"], 0, best_setup["gshape"] )


# sample size
nsize = 56885*2
best_setup = {'grate': 578.0 , 'gridsize':  3129.0, 'gshape': 10.0}
best_grate = 578.0



# average of rs, ri from ExAC
obs_rs = 2.2071483147409081
obs_rs_ri = 2.3031505393045419



# ancestral population
def phi_anc(xx, nu=1.0, theta0=1.0, gamma=0, h = 0.5, theta=None, beta=1):
    """
        One-dimensional phi for a constant-sized population with genic selection.
        the stationary distribution of allele fre- quency for the ancestral population size
        Williamson et al. PNAS. 2005
        """

    if theta is not None:
        raise ValueError('The parameter theta has been deprecated in favor of '
                         'parameters nu and theta0, for consistency with the '
                         'Integration functions.')
    if gamma == 0:
        # Standard neutral for ancestral population
        # constant nu for gamma == 0
        if xx[0] == 0:
            phi = 0*xx
            phi[1:] = nu*theta0/xx[1:]
            phi[0] = phi[1]
        else:
            phi = nu*theta0/xx
        return phi * 4.*beta/(beta+1.)**2

    # Beta effectively re-scales gamma.
    gamma = gamma * 4.*beta/(beta+1.)**2
    v = 1/np.float(nu)
    # Protect from warnings on division by zero

    phi = np.zeros( len( xx))
    if h == 0.5:
        try:
            phi[1:-1] = 1./(xx[1:-1]*(1-xx[1:-1]))\
                * (1-np.exp(-2*gamma*v*(1-xx[1:-1]),dtype=np.float128 ))/(1-np.exp(-2*gamma*v, dtype=np.float128))
        except ValueError:
            # Avoid overflow issues for very negative gammas
            phi[1:-1] = 1./(xx[1:-1]*(1-xx[1:-1])) * np.exp(2*gamma*xx[1:-1] , dtype=np.float128)

        phi[0] = phi[1]
        if xx[-1] == 1:
            limit = 2*gamma * np.exp(2*gamma, dtype=np.float128)/(np.exp(2*gamma, dtype=np.float128)-1)
            phi[-1] = limit

    if h != 0.5:
        gamma = gamma * 4.*beta/(beta+1.)**2
        # First we evaluate the relevant integrals.
        ints = np.empty(len(xx))
        integrand = lambda xi: np.exp(-4*gamma*h*xi - 2*gamma*(1-2*h)*xi**2 , dtype=np.float128 )
        val, eps = scipy.integrate.quad(integrand, 0, 1)
        #
        int0 = val
        for ii,q in enumerate(xx):
            val, eps = scipy.integrate.quad(integrand, q, 1)
            ints[ii] = val


        phi = np.exp( 4*gamma*h*xx + 2*gamma*(1-2*h)*xx**2 , dtype=np.float128)*ints/int0
        #
        # Protect from division by zero errors
        if xx[0] == 0 and xx[-1] == 1:
            phi[1:-1] *= 1./(xx[1:-1]*(1-xx[1:-1]))
        else:
            phi *= 1./(xx*(1-xx))

        if xx[0] == 0:
            # Technically, phi diverges at 0. This fixes lets us do numerics
            # sensibly.
            phi[0] = phi[1]
        if xx[-1] == 1:
            # I used Mathematica to check that this was the proper limit.
            phi[-1] = 1./int0
    # the probability that a particular site is at frequency i out out n
    return phi * nu*theta0 * 4.*beta/(beta+1.)**2



# dt : 1/(anc. effective population size * 2 )
def dt_gen( Ne_anc ):
    return 1/( Ne_anc*2.)


# grid setup
def sfs_grid(pts, minq , crwd=8.):
    # minq = 1/(Ne) or 1/(Nc)
    unif = np.linspace(-1,0, int( pts) )
    grid = 1./(1. + np.exp(-crwd*unif ))

    # Normalize
    grid = (grid-grid[0])/(grid[-1]-grid[0])
    return grid[ grid >= minq ]


# adding new mutations to lowest frequency class
def draft_mut_per_gen( phi, dt, xx, theta0):
    phi[ 1 ] += dt/xx[1 ] * theta0/2 * 2/( xx[ 2 ] - xx[ 0] )
    return phi


# infinitesimal mean and variance
# default parameters:  h = 0.5, beta = 1.
# mean
iMean = lambda x, gamma, h : gamma * 2 * (h + (1-2.*h )*x) * x * (1 - x )
# variance
iVar = lambda x, Nc, beta : 1./Nc * x * ( 1 - x) * ( beta + 1.)**2/(4.*beta )



dt = dt_gen(10000*2.)
delj = 0.5


def _compute_dfactor(dx):
    r"""
        \Delta_j from the paper.
        """
    # Controls how we take the derivative of the flux. The values here depend
    #  on the fact that we're defining our probability integral using the
    #  trapezoid rule.
    dfactor = np.zeros(len(dx)+1)
    dfactor[1:-1] = 2/(dx[:-1] + dx[1:])
    dfactor[0] = 2/dx[0]
    dfactor[-1] = 2/dx[-1]
    return dfactor


def _compute_delj(dx, MInt, VInt, axis=0):
    use_delj_trick = False
    r"""
        Chang an Cooper's \delta_j term. Typically we set this to 0.5.
        """
    # Chang and Cooper's fancy delta j trick...
    if use_delj_trick:
        # upslice will raise the dimensionality of dx and VInt to be appropriate
        # for functioning with MInt.
        upslice = [nuax for ii in xrange(MInt.ndim)]
        upslice [axis] = slice(None)

        wj = 2 *MInt*dx[upslice]
        epsj = np.exp(wj/VInt[upslice])
        delj = (-epsj*wj + epsj * VInt[upslice] - VInt[upslice])/(wj - epsj*wj)
        # These where statements filter out edge case for delj
        delj = np.where(np.isnan(delj), 0.5, delj)
        delj = np.where(np.isinf(delj), 0.5, delj)
    else:
        delj = 0.5
    return delj


def phi_per_gen(phi, xx, nu=1.0, gamma=0, h=0.5, beta=1., dt=0.001 ):
    #
    # change of allele frequency for each frequency class per generation
    #
    M = iMean(xx, gamma, h)
    MInt = iMean((xx[:-1] + xx[1:])/2, gamma, h)

    V = iVar(xx, nu, beta=beta)
    VInt = iVar((xx[:-1] + xx[1:])/2, nu, beta=beta)

    dx = np.diff(xx)

    dfactor = _compute_dfactor(dx)
    delj = _compute_delj(dx, MInt, VInt)

    a_tri = np.zeros(phi.shape)
    a_tri[1:] += dfactor[1:]*(-MInt * delj - V[:-1]/(2*dx))

    c_tri = np.zeros(phi.shape)
    c_tri[:-1] += -dfactor[:-1]*(-MInt * (1-delj) + V[1:]/(2*dx))

    b_tri = np.zeros(phi.shape)
    b_tri[:-1] += -dfactor[:-1]*(-MInt * delj - V[:-1]/(2*dx))
    b_tri[1:] += dfactor[1:]*(-MInt * (1-delj) + V[1:]/(2*dx))

    r_tri = phi/dt
    # Bondary conditions
    if(M[0] <= 0):
        b_tri[0] += (0.5/nu - M[0])*2/dx[0]
    if(M[-1] >= 0):
        b_tri[-1] += -(-0.5/nu - M[-1])*2/dx[-1]

    phi = np.asarray( TDMA.TDMA(a_tri, b_tri+1/dt, c_tri, r_tri, len(a_tri) ) )
    #phi = np.asarray( dadi.tridiag.tridiag(a, b + 1/dt, c, r) )
    return phi




def phi_for_contant_T(phi, xx, T, nu=1, gamma=0, h=0.5, theta0=1, initial_t=0, beta=1):
    # T = 0 : change of allele frequency per generation
    # T > 0 : cummulative changes of allele frequency over T
    'Integrate one population with constant parameters.'
    if np.any(np.less([T,nu,theta0], 0)):
        raise ValueError('A time, population size, migration rate, or theta0 '
                         'is < 0. Has the model been mis-specified?')
    if np.any(np.equal([nu], 0)):
        raise ValueError('A population size is 0. Has the model been '
                         'mis-specified?')

    M = iMean(xx, gamma, h)
    MInt = iMean((xx[:-1] + xx[1:])/2, gamma, h)
    V = iVar(xx, nu, beta=beta)
    VInt = iVar((xx[:-1] + xx[1:])/2, nu, beta=beta)

    dx = np.diff(xx)
    dfactor = _compute_dfactor(dx)
    delj = _compute_delj(dx, MInt, VInt)

    a_tri = np.zeros(phi.shape)
    a_tri[1:] += dfactor[1:]*(-MInt * delj - V[:-1]/(2*dx))

    c_tri = np.zeros(phi.shape)
    c_tri[:-1] += -dfactor[:-1]*(-MInt * (1-delj) + V[1:]/(2*dx))

    b_tri = np.zeros(phi.shape)
    b_tri[:-1] += -dfactor[:-1]*(-MInt * delj - V[:-1]/(2*dx))
    b_tri[1:] += dfactor[1:]*(-MInt * (1-delj) + V[1:]/(2*dx))

    # Boundary conditions
    if(M[0] <= 0):
        b_tri[0] += (0.5/nu - M[0])*2/dx[0]
    if(M[-1] >= 0):
        b_tri[-1] += -(-0.5/nu - M[-1])*2/dx[-1]

    dt = 1/(10000*2.)
    #dt = _compute_dt(dx,nu,[0],gamma,h)
    current_t = initial_t

    if T == 0:
        r = phi/dt
        # Bondary conditions
        if(M[0] <= 0):
            b_tri[0] += (0.5/nu - M[0])*2/dx[0]
        if(M[-1] >= 0):
            b_tri[-1] += -(-0.5/nu - M[-1])*2/dx[-1]
        phi = np.asarray( TDMA.TDMA(a_tri, b_tri+1/dt, c_tri, r, len(a_tri) ) )
        return phi
        #
    if T > 0:
        while current_t < T:
            this_dt = min(dt, T - current_t)

            draft_mut_per_gen(phi, this_dt, xx, theta0)
            #_inject_mutations_1D(phi, this_dt, xx, theta0)

            r0_tri =   phi/this_dt
            a0_tri = a_tri.copy() # copy(a)

            #
            phi = np.asarray( TDMA.TDMA(a0_tri, b_tri+1/this_dt, c_tri, r0_tri, len(a_tri)) )
            current_t += this_dt
        return phi




def phi_for_growth(phi, xx, T, nu, gamma, h= 0.5, theta0=1.0, current_t=0, beta=1 ):
    # recursive implicit function
    # nu, gamma are function of t
    # T : duration of growth
    dt = 1/(10000*2.)
    this_dt = min(dt, T - current_t)
    next_t = current_t + this_dt

    nu_t, gamma_t  = nu(current_t), gamma(current_t)
    #print nu_t, gamma_t
    dx = np.diff(xx)

    if dt > this_dt:
        return phi

    phi0 = draft_mut_per_gen(phi.copy() , this_dt, xx, theta0)

    M = iMean(xx, gamma_t, h)
    MInt = iMean((xx[:-1] + xx[1:])/2, gamma_t, h)

    V = iVar(xx, nu_t, beta=beta)
    VInt = iVar((xx[:-1] + xx[1:])/2, nu_t, beta=beta)

    dx = np.diff(xx)
    dfactor = _compute_dfactor(dx)
    delj = _compute_delj(dx, MInt, VInt)

    a_tri = np.zeros(phi.shape)
    a_tri[1:] += dfactor[1:]*(-MInt * delj - V[:-1]/(2*dx))

    c_tri = np.zeros(phi.shape)
    c_tri[:-1] += -dfactor[:-1]*(-MInt * (1-delj) + V[1:]/(2*dx))

    b_tri = np.zeros(phi.shape)
    b_tri[:-1] += -dfactor[:-1]*(-MInt * delj - V[:-1]/(2*dx))
    b_tri[1:] += dfactor[1:]*(-MInt * (1-delj) + V[1:]/(2*dx))

    # Boundary conditions
    if(M[0] <= 0):
        b_tri[0] += (0.5/nu_t - M[0])*2/dx[0]
    if(M[-1] >= 0):
        b_tri[-1] += -(-0.5/nu_t - M[-1])*2/dx[-1]

    r_tri = phi0/this_dt

    phi_t = np.asarray( TDMA.TDMA(a_tri, b_tri+1/dt, c_tri, r_tri, len(a_tri) ) )
    #
    #print next_t, nu_t, gamma_t, M[1:3] , sum( phi_t<0), phi_t[1]/sum(phi_t[2:]),  theta0

    if T > next_t:
        phi_t = phi_for_growth( phi_t.copy(), xx, T, nu, gamma, h , theta0=1.0, current_t = next_t, beta=1 )
        return phi_t



def sampleDerAlleles( phi,xx, ssize ):
    sf = np.zeros(ssize + 1 )
    for ii in xrange(0, 668 ): # int( ssize*0.01) ):  # ssize+1
        # binomial distribution
        binomDist = lambda  i , q  : scipy.stats.binom.pmf(i, ssize, q )
        biomd = binomDist( ii , xx )
        sf[ii] = scipy.trapz(biomd * phi, xx)
    return np.nan_to_num( sf )



# obtain_sfs

# geration in 2Ne, where Ne=10000
#
ts = np.array([ 213., 407., 100., 1280.] )
generations = np.array([ 0.01065,  0.02035,  0.005  ,  0.05  ])



def eu_pop( nu_current , scaledG ,  g_t , g_r , xx = []  ):
    generations = np.array([ 0.01065,  0.02035,  0.005  ,  0.05  ])
    # Nanc = 1.0, - G = 2*s*Nanc = 2*s
    # start with Nu = 1.0
    phi = phi_anc( xx , gamma = scaledG )
    # 1st bottleneck :  0.1520 - 0.04999
    #
    #T = 0.0766  # 0.1520 - 0.04999
    # Euorpean start : 43kya
    T_bot1 = 0.05
    phi = phi_for_contant_T( phi.copy(), xx,  T= T_bot1, nu = 1.0,  gamma = scaledG*1.0 )

    # recnet bottleneck : 720 g
    #T=  0.048974 #  0.04999 - 0.0111
    T_bot2 = 0.005
    phi = phi_for_contant_T( phi.copy(), xx,   T=T_bot2,  nu = 0.0549, gamma = scaledG*0.0549 )

    # constant before growth:620-213g
    T_bot3 = 0.02035
    phi = phi_for_contant_T( phi.copy(), xx,  T=T_bot3, nu=1.24, gamma = scaledG*1.24 )

    # growth : 213g = 5,325y
    #
    T_growth = 0.01065  # generations[0]
    nu_func = lambda t: np.exp(np.log(nu_current/1.24) * t/T_growth)*1.24
    #print nu_func(0), nu_func(0.005), nu_func(0.0111)
    gamma_relax = lambda t: scaledG*nu_func(t)
    if g_t != 0:
        def gamma_relax ( t ):
            if t <  g_t:
                return scaledG*nu_func(t)
            if t >=  g_t:
                return scaledG*g_r*nu_func(t)
    phi = phi_for_growth( phi.copy(), xx, T=T_growth , nu = nu_func , gamma = gamma_relax )
    #print scaledG, nu_func(T_growth), g_t, g_r, phi[ 1]/sum( phi[2:]), len( phi)
    return phi



#
# tuning growth rate,
# parameters sets of grids
#


def gamma_addrandom( G ):
    return np.mean( scipy.stats.gamma.rvs( 0.206, scale = abs( G ) , size = 30 )  )


def neutr( nu_current, newS=0 , g_t = 0, g_r = 0 , ssize =  nsize, xx = [] ):
    phi_s = eu_pop( nu_current, scaledG = 0, g_t = 0, g_r = 0 ,  xx = xx )
    fs_s = sampleDerAlleles( phi_s, xx , ssize )
    return fs_s


space = [
         hyperopt.hp.uniform( 'grate', 100., 3000. ),
         hyperopt.hp.uniform( 'gridsize', 1000, 56885 ),
         hyperopt.hp.uniform( 'gshape', 5.0, 30.0 ),
]


def theta_pi( sfs_xx ):
    combi = scipy.special.comb( nsize, 2 )
    ssfs = sum( [ i*(nsize - i)*sfs_xx[i-1] for i in range( 1, len( sfs_xx) + 1) ] )
    return ssfs/combi


def grate_est( params ):
    #print params
    grate, gridsize, gshape = params # 
    xx = sfs_grid( gridsize,  0, gshape )
    ss = neutr(  grate , newS = 0, g_t = 0, g_r = 0, ssize= nsize, xx = xx )
    sumdiff = sum( ( sfs_syn[1:1137]/float( sum( sfs_syn[1:1137])) -  ss[1:1137]/sum(ss[1:1137]) )**2 )
    return sumdiff


sfs_syn = sfs["sy"][2][1:1137] + sfs["sy"][3][1:1137]
sfs_syn = sfs["sy"][1] + sfs["sy"][2] + sfs["sy"][3] + sfs["sy"][4] + sfs["sy"][5]
trials = Trials()
best = fmin( grate_est , space= space , algo=tpe.suggest, max_evals= 10000, trials=trials)


xx = sfs_grid( best_setup[ "gridsize"], 0, best_setup["gshape"] )



def gamma_def(shape, scale ):
    # scaled to Nanc = 1.0
    # negative selection
    shapeMA = 0.65537660213331606
    scaleMA = 0.80389348002790517
    locMA = 0.010264378352720366
    qtl = [ 0.001, 0.2, 0.4,  0.6, 0.8, 0.9999]
    gbins = scipy.stats.gamma.ppf( qtl, shape, scale=scale)
    s_dist = lambda x : scipy.stats.gamma.pdf( x, shapeMA, loc =locMA , scale = scaleMA )
    gprob = np.array([ scipy.integrate.quad( s_dist, gbins[i], gbins[i+1]  )[0] for i in range( len( gbins) - 1 ) ])
    return gbins, gprob



def gamma_dist( nu_current, newS , g_t = 0.0, g_r = 1.0, xx = [], propneg = 1.0   ):
    nu_gazave = 10000.
    a = -1.
    if propneg == 0:
        a = 1.0
    gamma = 0.028
    if gamma <= 0:
        gamma = 0.0001
    shape = 0.206
    scale = 27184.466
    mgamma = a*np.mean( scipy.stats.gamma.rvs( shape,  scale = scale, size = 47 )  )/10000.
    phigamma = eu_pop( nu_current, mgamma , g_t , g_r , xx= xx  )
    fs_s = sampleDerAlleles( phigamma/0.403, xx , nsize )  #
    fs_s_0 = fs_s.copy()
    return fs_s_0




space = [
         hp.uniform('rt', 0.000, 0.01065 - 10/20000.   ),
         hp.loguniform('rr', np.log(10**-4) , np.log( 10**4) ),
         ]



if 1:
    ss = neutr(  best_setup[ "grate" ] , newS = 0, g_t = 0, g_r = 0, ssize= nsize, xx = xx )
    ss1 = ss[1]
    ss2 = np.float( sum( ss[2:667 ] ) )
    rs  = ss[1]/np.float( sum( ss[2:667] ))




def relax_est( params ):
        newS = 1.0
        rt, rr  = params
        propneg = 1.0
        if fnfs_obs < 0.935:
            propneg = 0.0
        ns = gamma_dist( nu_current = best_setup[ "grate" ] ,newS= newS, g_t = rt, g_r = rr , xx = xx , propneg = propneg )
        ns667 = ns[ 2: 667 ]
        fnfs_exp = ( ( ns[1] )/( sum( ns667[ ns667 > 0 ] ) ) )/(  rs )
        scaledns1 = ns[1]/ss2
        squardiff_rnrs = abs( fnfs_exp -  fnfs_obs  )/0.0805185  #
        squardiff_ns1 = abs( scaledns1 - ns1_obs)/0.076227 # 
        print  fnfs_obs, newS, rt, rr , propneg, ns[1], sum( ns667[ ns667 > 0.0 ] ) , ns1_obs, scaledns1,   squardiff_rnrs, squardiff_ns1
        if np.isnan( squardiff_rnrs*squardiff_ns1):
            squardiff_rnrs =  100.0
            squardiff_ns1 = 100.0
        return np.sqrt(  squardiff_rnrs**2 + squardiff_ns1**2 )



if 1:
    res_genes = {}
    for fnfs_obs_inx in  fnfs_sim_const  :  
        fnfs_obs = fnfs_obs_inx[ 0 ]
        ns1_obs = fnfs_obs_inx[ 1 ]
        trials = Trials()
        best = fmin( relax_est , space= space , algo=tpe.suggest, max_evals= 100, trials=trials)
        print fnfs_obs, ns1_obs, best
        nres = len( res_genes)
        res_genes[ nres ] = {}
        res_genes[ nres ]["tri"] = trials
        res_genes[ nres ]["best" ] = best
        res_genes[ nres ] ["rnrs" ] = fnfs_obs
    out = open( path2dat + "/esim_n1/sim7mfreeG_in%d.pkl" % (inx), "wb")
    pickle.dump(  res_genes, out, 2 )
    out.close()
