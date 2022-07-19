from threading import local
import numpy as np
from requests import patch
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pyplot as plt

def get_histograms(sdata, mins=None, maxs=None, category=None, resolution=5):

    if mins is None:
        mins = sdata.coordinates.min(0)

    if maxs is None:
        maxs = sdata.coordinates.max(0)

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

    return histograms

def determine_gains(stats1,stats2):

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

def ssam(sdata, signatures, supervised=None, adata_obs_label='celltype', kernel_bandwidth=2.5, patch_length=500, threshold_exp=0.1, threshold_cor=0.2):

    # if sdata.scanpy is not None and ((supervised is None) or (supervised is False) ):
    #     signatures = sdata.scanpy.generate_signatures(adata_obs_label)
    # else:
    #     signatures=None
    out_shape = (np.ceil(sdata.x.max()).astype(int),
                 np.ceil(sdata.y.max()).astype(int))

    ct_map = np.zeros((out_shape), dtype=int)
    patch_delimiters = list(range(0, np.ceil(sdata.x.max()).astype(int), int(
        patch_length-kernel_bandwidth*3)))+[np.ceil(sdata.x.max()).astype(int)]

    localmax_vectors = []
    for i, p in enumerate(patch_delimiters[:-1]):
        sdata_patch = sdata.spatial[p:patch_delimiters[i+1]]
        hists = get_histograms(sdata_patch, resolution=1)
        hists = np.array([gaussian_filter(h, kernel_bandwidth) for h in hists])

        norm = np.sum(hists, axis=0)

        if signatures is None:
            localmaxs = maximum_filter(norm, size=5)
            localmaxs = (localmaxs == norm) & (localmaxs != 0)
            # localmaxs = np.where(localmaxs)
            localmaxs = [h[localmaxs] for h in hists]
            localmax_vectors.append(localmaxs)

        mask = norm > 0.02

        exps = np.zeros((len(sdata.genes),mask.sum()))
        exps[sdata.stats.loc[sdata_patch.genes].gene_ids.values,:]=hists[:, mask]

        signatures -= signatures.mean(0)
        signatures /= signatures.std(0)

        exps -= exps.mean(0)
        exps -= exps.std(0)

        local_ct_map = mask.astype(int)
        corrs = np.inner(exps.T, signatures.T)
        local_ct_map[mask] = corrs.argmax(1)

        idcs=sdata_patch.index.values
        x_=int(sdata[idcs].x.min())
        y_=int(sdata[idcs].y.min())

        ct_map[x_:x_+local_ct_map.shape[0],
               y_:y_+local_ct_map.shape[1]]=local_ct_map

    return ct_map.T,
