#!/usr/bin/env python

import kmeans_radec
import numpy as np
import os
import sys
import esutil

import matplotlib.pyplot as plt


def GetData2(dir=os.path.join(os.environ['GLOBALDIR'],'v2+3_matched'), band='i', bands='griz', killnosim=False):
    truth = esutil.io.read(os.path.join(dir, 'truth_%s-%s.fits'%(band,bands)))
    matched = esutil.io.read(os.path.join(dir, 'matched_%s-%s.fits'%(band,bands)))
    des = esutil.io.read(os.path.join(dir, 'des_%s-%s.fits'%(band,bands)))
    nosim = esutil.io.read(os.path.join(dir, 'nosim_%s-%s.fits'%(band,bands)))

    a = len(matched)
    if killnosim:
        vstr = 'version_%s'%(band)
        bstr = 'balrog_index'
        versions = np.unique(nosim[vstr])
        for v in versions:
            mv_cut = (matched[vstr]==v)
            nv_cut = (nosim[vstr]==v)
            bad = nosim[nv_cut][bstr]
            mb_cut = np.in1d(matched[bstr],bad)
            both = (mv_cut & mb_cut)
            matched = matched[-both]

    return truth, matched, des, nosim

def EnforceArray2D(arr):
    arr = EnforceArray(arr, arr)
    arr = EnforceArray(arr, arr[0])
    return arr

def EnforceArray1D(arr):
    arr = EnforceArray(arr, arr)
    return arr

def EnforceArray(arr, check):
    try:
        len(check)
    except:
        arr = [arr]
    return arr



"""
For the target function you give to JacknifeOnSphere, the first argument MUST be an array of the JKed tables. 
Even if you're only JKing a single data table, you'll receive this as an array like [JKed table].
I've done it like this, so it's easy to JK multiple tables simultaneously if you want.
You can define extra arguments if you want too.

You MUST return a 2D array from your target function. 
The first return element is an array of the quantities you want to find the covariances of.
The second return element is whatever else you want, maybe coordinates, or something else.
If you don't want anything, just set it to None. You should return it in an array though, this makes it easier to keep track of things.
"""

def Dummy(arr):
    a = arr[0]['mag_i']
    bins = np.arange(18, 25, 0.2)
    hist, bins = np.histogram(a, bins=bins)
    return [ [hist], [len(hist)] ]

def Histogram(arr, thing, binsize=0.1):
    sim = arr[0][thing]
    des = arr[1][thing]
    bins = np.arange(18, 25, binsize)
    cent = (bins[1:] + bins[0:-1]) / 2

    shist, bins = np.histogram(sim, bins=bins)
    dhist, bins = np.histogram(des, bins=bins)

    ss = float(len(sim))
    dd = float(len(des))

    s = shist / ss
    d = dhist / dd

    return [ [s,d], [cent, ss, dd] ]


"""
The first argument (jarrs) is an array of the tables (recarrays) you want to JK.
The second argument (jras) is a list of the ra columns in these tables.
The third argument (jdecs) is a list of the dec columns in these tables.
The fourth agrument (jfunc) is the target function you want to JK.

Optional Args:
    jargs: extra arguments to the target function, other than the required first first agrument, see comments above.
    jkwargs: keyword extra arguments to the target function

    jtype: 'generate' will use kmeans to generate the JK regions. 'read' reads them from a file
    jfile: used with jtype='read', the file to read from
    njack: number of JK regions, only relevant when generating, otherwise the file sets the number
    generateonly: returns just the name of the JK region file after generating it, without actually doing any JKing
    gindex: which element in jarrs/jras/jdecs to use for generating the JK regions.

Returns a 4-element array:
    Element 0: Array of the answers over the full area of the things you JKed
    Element 1: Covariances of the things you JKed
    Element 2: Array of whatever extra stuff you returned from your target function during the call for the full area
    Element 3: Like Element 2, but with arrays in each component for the extra returns of the target function in each JK realization.


"""

def JackknifeOnSphere(jarrs, jras, jdecs, jfunc, jargs=[], jkwargs={}, jtype='generate', jfile=None, njack=24, generateonly=False, gindex=0):
    jarrs = EnforceArray2D(jarrs)
    jras = EnforceArray2D(jras)
    jdec = EnforceArray2D(jdecs)

    if jtype=='generate':
        rdi = np.zeros( (len(jarrs[gindex]),2) )
        rdi[:,0] = jarrs[gindex][jras[gindex]]
        rdi[:,1] = jarrs[gindex][jdecs[gindex]]

        if jfile is None:
            jfile = 'JK-{0}.txt'.format(njack)
        km = kmeans_radec.kmeans_sample(rdi, njack, maxiter=100, tol=1.0e-5)
        if not km.converged:
            raise RuntimeError("k means did not converge")
        np.savetxt(jfile, km.centers)
        if generateonly:
            return jfile

    elif jtype=='read':
        centers = np.loadtxt(jfile)
        km = kmeans_radec.KMeans(centers)
        njack = len(centers)
    
    ind = []
    for i in range(len(jarrs)):
        rdi = np.zeros( (len(jarrs[i]),2) )
        rdi[:,0] = jarrs[i][jras[i]]
        rdi[:,1] = jarrs[i][jdecs[i]]
        index = km.find_nearest(rdi)
        ind.append(index)
    
    full_j, full_other = jfunc(jarrs, *jargs, **jkwargs)
    full_j = EnforceArray2D(full_j)
    full_other = EnforceArray1D(full_other)

    it_j = []
    it_other = [ [] ] * len(full_other)
    for j in range(njack):
        ja = []
        for i in range(len(jarrs)):
            cut = (ind[i]==j)
            ja.append(jarrs[i][-cut])
            
            if j==0:
                it_j.append( [] )

        i_j, i_other = jfunc(ja, *jargs, **jkwargs)
        i_j = EnforceArray2D(i_j)
        i_other = EnforceArray1D(i_other)
        for i in range(len(i_j)):
            it_j[i].append( np.copy(i_j[i]) )
        for i in range(len(i_other)):
            it_other[i].append(i_other[i])

    for i in range(len(it_j)):
        it_j[i] = np.array(it_j[i])

    cov_j = []
    for k in range(len(full_j)):
        csize = len(full_j[k])
        cov = np.zeros( (csize,csize) )
        
        for i in range(csize):
            for j in range(i, csize):
                cov[i,j] =  np.sum( (it_j[k][:,i] - full_j[k][i]) * (it_j[k][:,j] - full_j[k][j]) ) * float(njack-1)/njack

                if i!=j:
                    cov[j,i] = cov[i,j]
        cov_j.append(cov)

    return [full_j, cov_j, full_other, it_other]


    

def Test(band='i'):
  
    datadir = os.path.join(os.environ['GLOBALDIR'],'with-version_matched')
    truth, matched, des, nosim = GetData2(band=band, dir=datadir, killnosim=False)

    matched = matched[ matched['modest_i']==0 ]
    des = des[ des['modest_i']==0 ]

    #jfile = JackknifeOnSphere( [nosim], ['ra_i'], ['dec_i'], Dummy, jtype='generate', njack=22, jfile='nosim-JK-22.txt', generateonly=True)
    #print jfile
    #hists, covs, oth, oths = JackknifeOnSphere( [nosim], ['ra_i'], ['dec_i'], Dummy, jtype='read', jfile='nosim-JK-22.txt')
    #print hists, covs, oth, oths

    #jfile = JackknifeOnSphere( [matched,des], ['alphawin_j2000_i','alphawin_j2000_i'], ['deltawin_j2000_i', 'deltawin_j2000_i'], Histogram, jargs=['mag_auto_i'], jkwargs={'binsize':0.1}, jtype='generate', jfile='fullJK-24.txt', generateonly=True)
    hists, covs, extra, jextra = JackknifeOnSphere( [matched,des], ['alphawin_j2000_i','alphawin_j2000_i'], ['deltawin_j2000_i', 'deltawin_j2000_i'], Histogram, jargs=['mag_auto_i'], jkwargs={'binsize':0.2}, jtype='read', jfile='fullJK-24.txt')
    sim_hist, des_hist = hists
    sim_cov, des_cov = covs
    centers = extra[0]

    serr = np.sqrt(np.diag(sim_cov))
    derr = np.sqrt(np.diag(des_cov))
    #print sim_hist, serr
    #print np.array(serr) / np.array(sim_hist)
   
    fig, ax = plt.subplots(1,1, figsize=(10,6))
    ax.errorbar(centers, sim_hist, yerr=serr, color='blue', label='Balrog')
    ax.errorbar(centers, des_hist, yerr=derr, color='red', label='DES')
    ax.set_yscale('log')
    plt.show()


if __name__=='__main__': 
   
    Test()
