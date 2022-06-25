from scipy.signal import fftconvolve
import numpy as np
# import


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

        kernels.append(kernel_2)

    kernels = np.array(kernels)

    return (kernels,vals)


def get_histograms(coords1, coords2=None,resolution=5):
    if coords2 is None:
        coords2 = coords1

    mins = np.min((coords1.min(0), coords2.min(0)), axis=0)
    maxs = np.max((coords1.max(0), coords2.max(0)), axis=0)

    n_bins = np.ceil(np.divide(maxs-mins, resolution)).astype(int)

    return(np.histogram2d(*coords1.T, bins=n_bins, range=([mins[0], maxs[0]], [mins[1], maxs[1]])))


def co_occurrence(coords1, coords2,resolution=5, max_radius=None, linear_steps=None):

    hist1 = get_histograms(coords1, coords2,resolution=resolution)[0]
    hist2 = get_histograms(coords2, coords1,resolution=resolution)[0]

    if max_radius is None:
        max_radius=np.ceil(np.min(hist1.shape)/4).astype(int)
    else:
        max_radius = np.ceil(max_radius/resolution).astype(int)
    if (linear_steps is None) or (linear_steps>=max_radius):
        linear_steps=min(max_radius,20)


    kernels,radii = _get_kernels(max_radius,linear_steps)

    co_occurrences = np.zeros((2,2,len(kernels)))

    kernel=np.zeros_like(hist1)

    for i,k in enumerate(kernels):
        conved = fftconvolve(hist1,k,mode='same')

        co_occurrences[0,0,i]=(conved*hist1).sum()/k.sum()
        co_occurrences[0,1,i]=(conved*hist2).sum()/k.sum()

        conved = fftconvolve(hist2,k,mode='same')

        co_occurrences[1,0,i]=(conved*hist1).sum()/k.sum()
        co_occurrences[1,1,i]=(conved*hist2).sum()/k.sum()



    return co_occurrences,radii*resolution

    # for k in kernels:

