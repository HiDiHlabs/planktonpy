from importlib.resources import path
from textwrap import fill
from threading import local

import numpy as np
# from numpy.ma.extras import _covhelper
# from numpy.ma.core import dot

from requests import patch
from scipy.ndimage import gaussian_filter, maximum_filter
from plankton.pixelmaps import PixelMap

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

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
    ax1.set_title('compared molecule counts:')
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
# def plot_gains(self):

#     sc_genes = self.adata.var.index

#     count_ratios = self.determine_gains()
#     idcs = np.argsort(count_ratios)

#     ax = plt.subplot(111)

#     span = np.linspace(0, 1, len(idcs))
#     clrs = np.stack([
#         span,
#         span * 0,
#         span[::-1],
#     ]).T

#     ax.barh(range(len(count_ratios)), np.log(
#         count_ratios[idcs]), color=clrs)

#     ax.text(0,
#             len(idcs) + 3,
#             'lost in spatial ->',
#             ha='center',
#             fontsize=12,
#             color='red')
#     ax.text(0, -3, '<- lost in SC', ha='center', fontsize=12, color='lime')

#     for i, gene in enumerate(sc_genes[idcs]):
#         if count_ratios[idcs[i]] < 1:
#             ha = 'left'
#             xpos = 0.05
#         else:
#             ha = 'right'
#             xpos = -0.05
#         ax.text(0, i, gene, ha=ha)

#     ax.set_yticks([], [])

# def compare_counts(stats1,stats2):

#     sc_genes = (self.adata.var.index)
#     sc_counts = (np.array(self.adata.X.sum(0)).flatten())
#     sc_count_idcs = np.argsort(sc_counts)
#     count_ratios = np.log(self.determine_gains())
#     count_ratios -= count_ratios.min()
#     count_ratios /= count_ratios.max()

#     ax1 = plt.subplot(311)
#     ax1.set_title('compared molecule counts:')
#     ax1.bar(np.arange(len(sc_counts)),
#             sc_counts[sc_count_idcs], color='grey')
#     ax1.set_ylabel('log(count) scRNAseq')
#     ax1.set_xticks(np.arange(len(sc_genes)),
#                     sc_genes[sc_count_idcs],
#                     rotation=90)
#     ax1.set_yscale('log')

#     ax2 = plt.subplot(312)
#     for i, gene in enumerate(sc_genes[sc_count_idcs]):
#         plt.plot(
#             [i, self.sdata.stats.get_count_rank(gene)],
#             [1, 0],
#         )
#     plt.axis('off')
#     ax2.set_ylabel(' ')

#     ax3 = plt.subplot(313)
#     self.sdata.plot_bars(ax3, color='grey')
#     ax3.invert_yaxis()
#     ax3.xaxis.tick_top()
#     ax3.xaxis.set_label_position('top')
#     ax3.set_ylabel('log(count) spatial')


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

    if (sdata.scanpy is not None) and (signatures is None):
        signatures = sdata.scanpy.generate_signatures(adata_obs_label)

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

    ct_map = fill_celltypemaps(
        ct_map, min_blob_area=min_blob_area, fill_blobs=fill_blobs)
    return PixelMap(ct_map.T, px_p_um=1/output_um_p_px)


def localmax_sampling(sdata, n_clusters=10, min_distance=3, bandwidth=4):
    n_bins = np.array(sdata.spatial.shape).astype(int)

    vf = (gaussian_filter(np.histogram2d(
        *sdata.coordinates.T, bins=n_bins)[0], 2))
    localmaxs = np.where((maximum_filter(vf, min_distance) == vf) & (vf > 0.2))
    knn = NearestNeighbors(n_neighbors=150)
    knn.fit(sdata.coordinates)
    dists, nbrs = knn.kneighbors(np.array(localmaxs).T)
    neighbor_types = np.array(sdata.gene_ids)[nbrs]

    counts = np.zeros((dists.shape[0], len(sdata.genes)))

    bandwidth = bandwidth
    def kernel(x): return np.exp(-x**2/(2*bandwidth**2))

    for i in range(0, dists.shape[1]):
        counts[np.arange(dists.shape[0]), neighbor_types[:, i]
               ] += kernel(dists[:, i])

    # assert (all(counts.sum(1)) > 0)

    counts=np.nan_to_num(counts/counts.sum(1)[:,None])

    ica = FastICA(n_components=30)
    facs = ica.fit_transform(counts)

    km = KMeans(
        n_clusters=n_clusters,
        max_iter=100,
        n_init=1,).fit(counts[:])

    return km.cluster_centers_

