from threading import local
import numpy as np
from requests import patch
from scipy.ndimage import gaussian_filter, maximum_filter



def get_histograms(sdata, category=None,resolution=5):

    mins = sdata.coordinates.min(0)
    maxs = sdata.coordinates.max(0)

    n_bins = np.ceil(np.divide(maxs-mins, resolution)).astype(int)

    histograms=[]
    
    if category is None:
        for gene in sdata.genes:
            histograms.append(np.histogram2d(*sdata[sdata.g==gene].coordinates.T, bins=n_bins, range=([mins[0], maxs[0]], [mins[1], maxs[1]]))[0])
        
    else:
        for c in sdata[category].cat.categories:
            histograms.append(np.histogram2d(*sdata[sdata[category]==c].coordinates.T, bins=n_bins, range=([mins[0], maxs[0]], [mins[1], maxs[1]]))[0])

    return histograms


def ssam(sdata,adata=None,kernel_bandwidth=2.5, patch_length=500):

    patch_delimiters = list(range(0,np.ceil(sdata.x.max()).astype(int),int(patch_length-kernel_bandwidth*3)))+[np.ceil(sdata.x.max()).astype(int)]

    localmax_vectors = []
    for i,p in enumerate(patch_delimiters[:-1]):
        hists = get_histograms(sdata.spatial[p:patch_delimiters[i+1]],resolution=1)
        hists = [gaussian_filter(h,kernel_bandwidth) for h in hists]

        norm=np.sum(hists,axis=0)
        
        if adata is None:
            localmaxs=maximum_filter(norm,size=5)
            localmaxs = (localmaxs==norm)&(localmaxs!=0)
            # localmaxs = np.where(localmaxs)
            localmaxs= [h[localmaxs] for h in hists]
            localmax_vectors.append(localmaxs)
        # break

    return localmax_vectors
    


