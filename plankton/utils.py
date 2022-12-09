from importlib.resources import path
from textwrap import fill
from multiprocessing import Pool

import numpy as np
import community
import networkx as nx

# from requests import patch
from scipy.ndimage import gaussian_filter, maximum_filter
from plankton.pixelmaps import PixelMap

from sklearn.neighbors import NearestNeighbors,kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN
from sklearn import preprocessing

from skimage import filters

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
import scipy
from scipy.spatial.distance import cdist

def determine_gains(stats1, stats2):

    norm_counts1 = stats1.counts/stats1.counts.sum()
    norm_counts2 = stats2.counts/stats2.counts.sum()

    return norm_counts1/norm_counts2


def hbar_compare(stat1, stat2, labels=None, text_display_threshold=0.02, c=None):

    genes_united = sorted(
        list(set(np.concatenate([stat1.index, stat2.index]))))[::-1]
    counts_1 = [
        0]+[stat1.loc[i].counts if i in stat1.index else 0 for i in genes_united]
    counts_2 = [
        0]+[stat2.loc[i].counts if i in stat2.index else 0 for i in genes_united]
    cum1 = np.cumsum(counts_1)/sum(counts_1)
    cum2 = np.cumsum(counts_2)/sum(counts_2)

    if c is None:
        c = [None]*len(cum1)

    for i in range(1, len(cum1)):

        bars = plt.bar([0, 1], [cum1[i]-cum1[i-1], cum2[i]-cum2[i-1]],
                       bottom=[cum1[i-1], cum2[i-1], ], width=0.4, color=c[i-1])
        clr = bars.get_children()[0].get_facecolor()
        plt.plot((0.2, 0.8), (cum1[i], cum2[i]),
                 c=plt.rcParams['axes.facecolor'], alpha=0.7)
        plt.fill_between(
            (0.2, 0.8), (cum1[i], cum2[i]), (cum1[i-1], cum2[i-1]), color=clr, alpha=0.2)

        if (counts_1[i]/sum(counts_1) > text_display_threshold) or \
                (counts_2[i]/sum(counts_2) > text_display_threshold):
            plt.text(0.5, (cum1[i]+cum1[i-1]+cum2[i]+cum2[i-1])/4,
                     genes_united[i-1], ha='center',)

    if labels is not None:
        plt.xticks((0, 1), labels)


def sorted_bar_compare(stat1, stat2, kwargs1={}, kwargs2={}):
    categories_1 = (stat1.index)
    counts_1 = (np.array(stat1.counts).flatten())
    counts_1_idcs = np.argsort(counts_1)
    # count_ratios = np.log(self.determine_gains())
    # count_ratios -= count_ratios.min()
    # count_ratios /= count_ratios.max()

    ax1 = plt.subplot(311)
    ax1.set_title('compared     lecule counts:')
    ax1.bar(np.arange(len(counts_1)),
            counts_1[counts_1_idcs], color='grey', **kwargs1)
    # ax1.set_ylabel('log(count) scRNAseq')
    ax1.set_xticks(np.arange(len(categories_1)),
                   categories_1[counts_1_idcs],
                   )
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_yscale('log')

    ax2 = plt.subplot(312)
    for i, gene in enumerate(categories_1[counts_1_idcs]):
        if gene in stat2.index:
            plt.plot(
                [i, stat2.count_ranks[gene]],
                [1, 0],
            )
    plt.axis('off')
    ax2.set_ylabel(' ')

    ax3 = plt.subplot(313)
    ax3.bar(np.arange(len(stat2)),
            stat2.counts[stat2.count_indices], color='grey', **kwargs2)

    ax3.set_xticks(np.arange(len(stat2.index)),
                   stat2.index[stat2.count_indices],
                   rotation=90)
    ax3.set_yscale('log')
    ax3.invert_yaxis()
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')
    # ax3.set_ylabel('log(count) spatial')
    return(ax1, ax2, ax3)


def fill_celltypemaps(ct_map, fill_blobs=True, min_blob_area=0, filter_params={}, output_mask=None):
    """
    Post-filter cell type maps created by `map_celltypes`.

    :param min_r: minimum threshold of the correlation.
    :type min_r: float
    :param min_norm: minimum threshold of the vector norm.
        If a string is given instead, then the threshold is automatically determined using
        sklearn's `threshold filter functions <https://scikit-image.org/docs/dev/api/skimage.filters.html>`_ (The functions start with `threshold_`).
    :type min_norm: str or float
    :param fill_blobs: If true, then the algorithm automatically fill holes in each blob.
    :type fill_blobs: bool
    :param min_blob_area: The blobs with its area less than this value will be removed.
    :type min_blob_area: int
    :param filter_params: Filter parameters used for the sklearn's threshold filter functions.
        Not used when `min_norm` is float.
    :type filter_params: dict
    :param output_mask: If given, the cell type maps will be filtered using the output mask.
    :type output_mask: np.ndarray(bool)
    """

    from skimage import measure

    filtered_ctmaps = np.zeros_like(ct_map) - 1

    for cidx in np.unique(ct_map):
        mask = ct_map == cidx
        if min_blob_area > 0 or fill_blobs:
            blob_labels = measure.label(mask, background=0)
            for bp in measure.regionprops(blob_labels):
                if min_blob_area > 0 and bp.filled_area < min_blob_area:
                    for c in bp.coords:
                        mask[c[0], c[1], ] = 0

                    continue
                if fill_blobs and bp.area != bp.filled_area:
                    minx, miny,  maxx, maxy, = bp.bbox
                    mask[minx:maxx, miny:maxy, ] |= bp.filled_image

        filtered_ctmaps[np.logical_and(mask == 1, np.logical_or(
            ct_map == -1, ct_map == cidx))] = cidx

    return filtered_ctmaps


def get_histograms(sdata, mins=None, maxs=None, category=None, resolution=5):

    if mins is None:
        mins = sdata.coordinates.min(0)

    if maxs is None:
        maxs = sdata.coordinates.max(0)+1

    # print(mins,maxs)

    n_bins = np.ceil(np.divide(maxs-mins, resolution)).astype(int)

    histograms = []

    if category is None:
        for gene in sdata.genes:
            histograms.append(np.histogram2d(
                *sdata[sdata.g == gene].coordinates.T, bins=n_bins, range=([mins[0], maxs[0]], [mins[1], maxs[1]]))[0])

    else:
        for c in sdata[category].cat.categories:
            histograms.append(np.histogram2d(
                *sdata[sdata[category] == c].coordinates.T, bins=n_bins, range=([mins[0], maxs[0]], [mins[1], maxs[1]]))[0])

    return histograms,


def crosscorr(x, y):
    x -= np.array(x.mean(1))[:, None]
    y -= np.array(y.mean(1))[:, None]
    c = (np.dot(x, y.T)/x.shape[1]).squeeze()

    return np.nan_to_num(np.nan_to_num(c/np.array(x.std(1))[:, None])/np.array(y.std(1))[None, :])


def ssam(sdata, signatures=None, adata_obs_label='celltype', kernel_bandwidth=2.5, output_um_p_px=5,
         patch_length=1000, threshold_exp=0.1, threshold_cor=0.1, background_value=-1,
         fill_blobs=True, min_blob_area=10):

    if signatures is None:
        signatures = sdata.uns['ssam']['centroids']

    # if (sdata.scanpy is not None):
        # signatures = sdata.scanpy.generate_signatures(adata_obs_label)

    kernel_bandwidth_px = kernel_bandwidth/output_um_p_px

    out_shape = (np.ceil(sdata.x.max()/output_um_p_px+kernel_bandwidth_px*3).astype(int),
                 np.ceil(sdata.y.max()/output_um_p_px+kernel_bandwidth_px*3).astype(int))

    ct_map = np.zeros((out_shape), dtype=int)+background_value
    vf_norm = np.zeros_like(ct_map)

    range_x = np.floor(sdata.x.min()).astype(
        int), np.ceil(sdata.x.max()).astype(int)
    patch_delimiters_x = list(
        range(range_x[0], range_x[1], patch_length))+[range_x[1]]

    range_y = np.floor(sdata.y.min()).astype(
        int), np.ceil(sdata.y.max()).astype(int)
    patch_delimiters_y = list(
        range(range_y[0], range_y[1], patch_length))+[range_y[1]]

    print(list(patch_delimiters_x), list(patch_delimiters_y))

    with tqdm_notebook(total=(len(patch_delimiters_x)-1)*(len(patch_delimiters_y)-1)) as pbar:
        for i, x in enumerate(patch_delimiters_x[:-1]):
            for j, y in enumerate(patch_delimiters_y[:-1]):
                sdata_patch = sdata.raw().spatial[x:patch_delimiters_x[i+1],
                                                y:patch_delimiters_y[j+1]]

                if not len(sdata_patch):
                    break

                mins = sdata_patch.coordinates.min(0).astype(int)
                maxs = sdata_patch.coordinates.max(
                    0).astype(int)+kernel_bandwidth*3
                hists = get_histograms(
                    sdata_patch, mins=mins, maxs=maxs, resolution=output_um_p_px)
                hists = np.concatenate(
                    [gaussian_filter(h, kernel_bandwidth_px) for h in hists])

                # print(hists.shape)

                norm = np.sum(hists, axis=0)

                vf_norm[(x+mins[0])//output_um_p_px:(x+mins[0])//output_um_p_px+norm.shape[0],
                        (y+mins[1])//output_um_p_px:(y+mins[1])//output_um_p_px+norm.shape[1], ] = norm

                mask = norm > threshold_exp

                exps = np.zeros((len(sdata.genes), mask.sum()))

                # print(exps.shape,signatures.shape)
                exps[sdata.stats.loc[sdata_patch.genes].gene_ids.values,
                    :] = hists[:, mask]

                local_ct_map = mask.astype(int)-1

                corrs = crosscorr(exps.T, signatures)
                # print(corrs.min())
                corrs_winners = corrs.argmax(1)
                corrs_winners[corrs.max(1) < threshold_cor] = background_value
                local_ct_map[mask] = corrs_winners

                # idcs=sdata_patch.index

                x_ = int(max(0, (x+mins[0])-kernel_bandwidth_px*3))//output_um_p_px
                y_ = int(max(0, (y+mins[1])-kernel_bandwidth_px*3))//output_um_p_px

                ct_map[x_:x_+local_ct_map.shape[0],
                    y_:y_+local_ct_map.shape[1]] = local_ct_map

                pbar.update(1)


    if not 'ssam' in sdata.uns.keys():
        sdata.uns['ssam']={}

    sdata.uns['ssam']['ct_map_raw']=ct_map
    sdata.uns['ssam']['vf_norm']=vf_norm

    ct_map = fill_celltypemaps(
        ct_map, min_blob_area=min_blob_area, fill_blobs=fill_blobs)


    sdata.uns['ssam']['ct_map_filtered']=ct_map

    return PixelMap(ct_map.T, px_p_um=1/output_um_p_px)


def localmax_sampling(sdata, min_distance=3, bandwidth=4):
    n_bins = np.array(sdata.coordinates.max(0)).astype(int)+2

    print(n_bins)
    vf = (gaussian_filter(np.histogram2d(
        *sdata.coordinates.T, bins=(range(n_bins[0]),range(n_bins[1])))[0], bandwidth))

    # return vf
    localmaxs = np.where((maximum_filter(vf, min_distance) == vf) & (vf > 0.2))
    knn = NearestNeighbors(n_neighbors=150)
    knn.fit(sdata.coordinates)
    dists, nbrs = knn.kneighbors(np.array(localmaxs).T)
    neighbor_types = np.array(sdata.gene_ids)[nbrs]

    localmax_samples = np.zeros((dists.shape[0], len(sdata.genes)))

    bandwidth = bandwidth
    def kernel(x): return np.exp(-x**2/(2*bandwidth**2))

    for i in range(0, dists.shape[1]):
        localmax_samples[np.arange(dists.shape[0]), neighbor_types[:, i]
               ] += kernel(dists[:, i])

    if not 'ssam' in sdata.uns.keys():
        sdata.uns['ssam']={}

    sdata.uns['ssam']['localmax_coords']=localmaxs
    sdata.uns['ssam']['localmax_samples']=localmax_samples

    return localmax_samples,localmaxs

def normalize_vectors(sdata, use_expanded_vectors=False, normalize_gene=False, normalize_vector=False, normalize_median=False, 
                        norm='l1',size_after_normalization=1e4, log_transform=False, scale=False):
    """
    Normalize and regularize vectors

    :param use_expanded_vectors: If True, use averaged vectors nearby local maxima of the vector field.
    :type use_expanded_vectors: bool
    :param normalize_gene: If True, normalize vectors by sum of each gene expression across all vectors.
    :type normalize_gene: bool
    :param normalize_vector: If True, normalize vectors by sum of all gene expression of each vector.
    :type normalize_vector: bool
    :param log_transform: If True, vectors are log transformed.
    :type log_transform: bool
    :param scale: If True, vectors are z-scaled (mean centered and scaled by stdev).
    :type scale: bool
    """
    if use_expanded_vectors:
        vec = np.array(sdata.uns['ssam']['expanded_vectors'], copy=True)
    else:
        vec = np.array(sdata.uns['ssam']['localmax_samples'], copy=True)
    if normalize_gene:
        vec = preprocessing.normalize(vec, norm=norm, axis=0) * size_after_normalization  # Normalize per gene
    if normalize_vector:
        vec = preprocessing.normalize(vec, norm="l1", axis=1) * size_after_normalization # Normalize per vector
    if normalize_median:
        def n(v):
            s, m = np.sum(v, axis=1), np.median(v, axis=1)
            s[m > 0] = s[m > 0] / m[m > 0]
            s[m == 0] = 0
            v[s > 0] = v[s > 0] / s[s > 0][:, np.newaxis]
            v[v == 0] = 0
            return v
        vec = n(vec)
    if log_transform:
        vec = np.log2(vec + 1)
    if scale:
        vec = preprocessing.scale(vec)
    sdata.uns['ssam']['normalized_vectors'] = vec
    return

def cluster_vectors(sdata, pca_dims=10, min_cluster_size=0, resolution=0.6, prune=1.0/15.0, snn_neighbors=30, max_correlation=1.0,
                    metric="correlation", subclustering=False, dbscan_eps=0.4, centroid_correction_threshold=0.8, random_state=0):
    """
    Cluster the given vectors using the specified clustering method.

    :param pca_dims: Number of principal componants used for clustering.
    :type pca_dims: int
    :param min_cluster_size: Set minimum cluster size.
    :type min_cluster_size: int
    :param resolution: Resolution for Louvain community detection.
    :type resolution: float
    :param prune: Threshold for Jaccard index (weight of SNN network). If it is smaller than prune, it is set to zero.
    :type prune: float
    :param snn_neighbors: Number of neighbors for SNN network.
    :type snn_neighbors: int
    :param max_correlation: Clusters with higher correlation to this value will be merged.
    :type max_correlation: bool
    :param metric: Metric for calculation of distance between vectors in gene expression space.
    :type metric: str
    :param subclustering: If True, each cluster will be clustered once again with DBSCAN algorithm to find more subclusters.
    :type subclustering: bool
    :param centroid_correction_threshold: Centroid will be recalculated with the vectors
        which have the correlation to the cluster medoid equal or higher than this value.
    :type centroid_correction_threshold: float
    :param random_state: Random seed or scikit-learn's random state object to replicate the same result
    :type random_state: int or random state object
    """
    
    vecs_normalized = sdata.uns['ssam']['normalized_vectors']
    vecs_normalized_dimreduced = PCA(n_components=pca_dims, random_state=random_state).fit_transform(vecs_normalized)

    print(vecs_normalized_dimreduced.shape)

    def cluster_vecs(vecs):
        k = min(snn_neighbors, vecs.shape[0])
        knn_graph = kneighbors_graph(vecs, k, mode='connectivity', include_self=True, metric=metric).todense()
        intersections = np.dot(knn_graph, knn_graph.T)
        snn_graph = intersections / (k + (k - intersections)) # borrowed from Seurat
        snn_graph[snn_graph < prune] = 0
        G = nx.from_numpy_matrix(snn_graph)
        partition = community.best_partition(G, resolution=resolution, random_state=random_state)
        lbls = np.array(list(partition.values()))
        return lbls

    def remove_small_clusters(lbls, lbls2=None):
        small_clusters = []
        cluster_indices = []
        lbls = np.array(lbls)
        for lbl in np.unique(lbls):
            if lbl == -1:
                continue
            cnt = np.sum(lbls == lbl)
            if cnt < min_cluster_size:
                small_clusters.append(lbl)
            else:
                cluster_indices.append(lbl)
        for lbl in small_clusters:
            lbls[lbls == lbl] = -1
        tmp = np.array(lbls, copy=True)
        for i, idx in enumerate(cluster_indices):
            lbls[tmp == idx] = i
        if lbls2 is not None:
            for lbl in small_clusters:
                lbls2[lbls2 == lbl] = -1
            tmp = np.array(lbls2, copy=True)
            for i, idx in enumerate(cluster_indices):
                lbls2[tmp == idx] = i
            return lbls, lbls2
        else:
            return lbls
    
    if subclustering:
        super_lbls = cluster_vecs(vecs_normalized_dimreduced)
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_cluster_size, metric=metric)
        all_lbls = np.zeros_like(super_lbls)
        global_lbl_idx = 0
        for super_lbl in set(list(super_lbls)):
            super_lbl_idx = np.where(super_lbls == super_lbl)[0]
            if super_lbl == -1:
                all_lbls[super_lbl_idx] = -1
                continue
            sub_lbls = dbscan.fit(vecs_normalized_dimreduced[super_lbl_idx]).labels_
            for sub_lbl in set(list(sub_lbls)):
                if sub_lbl == -1:
                    all_lbls[tuple([super_lbl_idx[sub_lbls == sub_lbl]])] = -1
                    continue
                all_lbls[tuple([super_lbl_idx[sub_lbls == sub_lbl]])] = global_lbl_idx
                global_lbl_idx += 1
    else:
        all_lbls = cluster_vecs(vecs_normalized_dimreduced)            
            
    new_labels = __correct_cluster_labels(vecs_normalized,all_lbls, centroid_correction_threshold)
    new_labels, all_lbls = remove_small_clusters(new_labels, all_lbls)
    centroids, centroids_stdev = __calc_centroid(vecs_normalized, new_labels)

    merge_candidates = []
    if max_correlation < 1.0:
        Z = scipy.cluster.hierarchy.linkage(centroids, metric='correlation')
        clbls = scipy.cluster.hierarchy.fcluster(Z, 1 - max_correlation, 'distance')
        for i in set(clbls):
            leaf_indices = np.where(clbls == i)[0]
            if len(leaf_indices) > 1:
                merge_candidates.append(leaf_indices)
        removed_indices = []
        for cand in merge_candidates:
            for i in cand[1:]:
                all_lbls[all_lbls == i] = cand[0]
                removed_indices.append(i)
        for i in sorted(removed_indices, reverse=True):
            all_lbls[all_lbls > i] -= 1

        new_labels = __correct_cluster_labels(vecs_normalized,all_lbls, centroid_correction_threshold)
        new_labels, all_lbls = remove_small_clusters(new_labels, all_lbls)
        centroids, centroids_stdev = __calc_centroid(vecs_normalized, new_labels)
            
    sdata.uns['ssam']['cluster_labels'] = all_lbls
    sdata.uns['ssam']['filtered_cluster_labels'] = new_labels
    sdata.uns['ssam']['centroids'] = np.array(centroids)
    sdata.uns['ssam']['centroids_stdev'] = np.array(centroids_stdev)
    #self.dataset.medoids = np.array(medoids)
    
    print("Found %d clusters"%len(centroids))
    return

def __correct_cluster_labels(normalized_vectors, cluster_labels, centroid_correction_threshold):
    new_labels = np.array(cluster_labels, copy=True)
    if centroid_correction_threshold < 1.0:
        for cidx in np.unique(cluster_labels):
            if cidx == -1:
                continue
            prev_midx = -1
            while True:
                vecs = normalized_vectors[new_labels == cidx]
                vindices = np.where(new_labels == cidx)[0]
                midx = vindices[np.argmin(np.sum(cdist(vecs, vecs), axis=0))]
                if midx == prev_midx:
                    break
                prev_midx = midx
                m = normalized_vectors[midx]
                for vidx, v in zip(vindices, vecs):
                    if crosscorr(v[None], m[None]) < centroid_correction_threshold:
                        new_labels[vidx] = -1
    return new_labels

def __calc_centroid(normalized_vectors, cluster_labels):
    centroids = []
    centroids_stdev = []
    #medoids = []
    for lbl in sorted(list(set(cluster_labels))):
        if lbl == -1:
            continue
        cl_vecs = normalized_vectors[cluster_labels == lbl, :]
        #cl_dists = scipy.spatial.distance.cdist(cl_vecs, cl_vecs, metric)
        #medoid = cl_vecs[np.argmin(np.sum(cl_dists, axis=0))]
        centroid = np.mean(cl_vecs, axis=0)
        centroid_stdev = np.std(cl_vecs, axis=0)
        #medoids.append(medoid)
        centroids.append(centroid)
        centroids_stdev.append(centroid_stdev)
    return centroids, centroids_stdev#, medoids

