from scipy.signal import fftconvolve
import numpy as np
from scipy import signal, fft as sp_fft


def _get_kernels(max_radius, linear_steps):

    if linear_steps > max_radius:
        linear_steps = max_radius

    vals = np.ones((max_radius)+1)
    vals[0] = 0
    vals[linear_steps+1:] += np.arange(max_radius-linear_steps)
    vals = np.cumsum(vals)
    vals = vals[:(vals < max_radius).sum()+1]
    vals[-1] = max_radius

    span = np.arange(-max_radius, max_radius + 1)
    X, Y = np.meshgrid(span, span)
    dists = (X**2+Y**2)**0.5

    kernel_1 = np.zeros_like(X)
    kernel_1[max_radius, max_radius] = 1
    kernels = [kernel_1]

    for i in range(len(vals)-1):

        r1 = vals[i]
        r2 = vals[i+1]

        kernel_1 = (dists-r1)
        kernel_1 = -(kernel_1-1)*(kernel_1 < 1)
        kernel_1[kernel_1 > 1] = 1
        kernel_1 = 1-kernel_1

        kernel_2 = (dists-r2)
        kernel_2 = -(kernel_2-1)*(kernel_2 < 1)
        kernel_2[kernel_2 > 1] = 1
        kernel_2[kernel_2 == 1] = kernel_1[kernel_2 == 1]

        kernels.append(kernel_2/kernel_2.sum())

    kernels = np.array(kernels)

    return (kernels,vals)

def get_histograms(sdata, category=None,resolution=5):

    mins = sdata.coordinates.min(0)
    maxs = sdata.coordinates.max(0)

    n_bins = np.ceil(np.divide(maxs-mins, resolution)).astype(int)

    histograms=[]
    
    if category is None:
        for gene in sdata.g:
            histograms.append(np.histogram2d(*sdata[sdata.g==gene].coordinates.T, bins=n_bins, range=([mins[0], maxs[0]], [mins[1], maxs[1]]))[0])
        
    else:
        for c in sdata[category].cat.categories:
            histograms.append(np.histogram2d(*sdata[sdata[category]==c].coordinates.T, bins=n_bins, range=([mins[0], maxs[0]], [mins[1], maxs[1]]))[0])

    return histograms

def co_occurrence(sdata, resolution=5, max_radius=400, linear_steps=5, category=None):
    
    hists = get_histograms(sdata, category=category)  

    if max_radius is None:
        max_radius=np.ceil(np.min(hists[0].shape)/4).astype(int)
    else:
        max_radius = np.ceil(max_radius/resolution).astype(int)
    if (linear_steps is None) or (linear_steps>=max_radius):
        linear_steps=min(max_radius,20)


    kernels,radii = _get_kernels(max_radius,linear_steps)
  
    co_occurrences = np.zeros((len(hists),)*2+(len(kernels),))

    shape = [hists[0].shape[a]+kernels[0].shape[a]-1 for a in [0,1]]
    fshape = [sp_fft.next_fast_len(shape[a], True) for a in [0,1]]
    
    kernels_fft = (sp_fft.rfftn(kernels, fshape,axes=[1,2]))
        
        
    for i in range(len(hists)):
        
        h1_fft = sp_fft.rfftn(hists[i], fshape,axes=[0,1])
        h1_fftprod =  (h1_fft*kernels_fft)
        h1_conv = sp_fft.irfftn(h1_fftprod,fshape,axes=[1,2])
        h1_conv = signal._signaltools._centered(h1_conv, (len(kernels),)+hists[0].shape).copy()
        
        h1_product=h1_conv*hists[i]
        co_occurrences[i,i]=h1_product.sum(axis=(1,2))
        
        for j in range(i+1,len(hists)):
            h2_product=h1_conv*hists[j]
            co_occurrences[i,j] = h2_product.sum(axis=(1,2))
            co_occurrences[j,i]= co_occurrences[i,j]

    return(co_occurrences,radii*resolution,kernels)


