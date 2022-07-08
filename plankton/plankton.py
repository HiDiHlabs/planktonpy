
from __future__ import annotations
from graph import SpatialGraph
from pixelmaps import PixelMap, PixelMask
import numpy as np
import pandas as pd
import random
import scipy
import anndata
from stats import co_occurrence
import scanpy as sc
import collections
from scipy import sparse
import pickle

import matplotlib
from matplotlib.cm import get_cmap
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as PathEffects
from matplotlib import pyplot as plt

# from typing import Union
# from cgitb import text
# from enum import unique
# from hashlib import new
# from msilib import add_data
# from turtle import color

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


# from sklearn.manifold import TSNE

plt.style.use('dark_background')
# matplotlib.rcParams['figure.figsize'] = (15, 15)


class ScanpyDataFrame():

    def __init__(self, sdata, scanpy_ds):
        self.sdata = sdata
        self.adata = scanpy_ds
        self.stats = ScStatistics(self)
        self.celltype_labels = None
        self.signature_matrix = None

    @property
    def shape(self):
        return self.adata.shape

    def generate_signatures(self, celltype_obs_marker='celltype'):

        self.celltype_labels = np.unique(self.adata.obs[celltype_obs_marker])

        self.signature_matrix = np.zeros((
            len(self.celltype_labels),
            self.adata.shape[1],
        ))

        for i, label in enumerate(self.celltype_labels):
            self.signature_matrix[i] = np.array(
                self.adata[self.adata.obs[celltype_obs_marker] == label].X.sum(
                    0)).flatten()

        self.signature_matrix = self.signature_matrix - self.signature_matrix.mean(
            1)[:, None]
        self.signature_matrix = self.signature_matrix / self.signature_matrix.std(
            1)[:, None]

        self.signature_matrix = pd.DataFrame(
            self.signature_matrix, index=self.celltype_labels, columns=self.stats.index)

        return self.signature_matrix

    def synchronize(self):

        joined_genes = (self.stats.genes.intersection(
            self.sdata.genes)).sort_values()

        # print(len(joined_genes))

        self.sdata.reset_index()
        self.adata = self.adata[:, joined_genes]
        self.stats = ScStatistics(self)
        self.sdata.drop(index=list(
            self.sdata.index[~self.sdata.g.isin(joined_genes)]),
            inplace=True)

        self.sdata.stats = PointStatistics(self.sdata)

        self.sdata.graph = SpatialGraph(self.sdata)

    def determine_gains(self):

        sc_genes = self.adata.var.index
        counts_sc = np.array(self.adata.X.sum(
            0) / self.adata.X.sum()).flatten()

        counts_spatial = np.array(
            [self.sdata.stats.get_count(g) for g in sc_genes])

        counts_spatial = counts_spatial / counts_spatial.sum()
        count_ratios = counts_sc / counts_spatial
        return count_ratios

    def plot_gains(self):

        sc_genes = self.adata.var.index

        count_ratios = self.determine_gains()
        idcs = np.argsort(count_ratios)

        ax = plt.subplot(111)

        span = np.linspace(0, 1, len(idcs))
        clrs = np.stack([
            span,
            span * 0,
            span[::-1],
        ]).T

        ax.barh(range(len(count_ratios)), np.log(
            count_ratios[idcs]), color=clrs)

        ax.text(0,
                len(idcs) + 3,
                'lost in spatial ->',
                ha='center',
                fontsize=12,
                color='red')
        ax.text(0, -3, '<- lost in SC', ha='center', fontsize=12, color='lime')

        for i, gene in enumerate(sc_genes[idcs]):
            if count_ratios[idcs[i]] < 1:
                ha = 'left'
                xpos = 0.05
            else:
                ha = 'right'
                xpos = -0.05
            ax.text(0, i, gene, ha=ha)

        ax.set_yticks([], [])

    def compare_counts(self):

        sc_genes = (self.adata.var.index)
        sc_counts = (np.array(self.adata.X.sum(0)).flatten())
        sc_count_idcs = np.argsort(sc_counts)
        count_ratios = np.log(self.determine_gains())
        count_ratios -= count_ratios.min()
        count_ratios /= count_ratios.max()

        ax1 = plt.subplot(311)
        ax1.set_title('compared molecule counts:')
        ax1.bar(np.arange(len(sc_counts)),
                sc_counts[sc_count_idcs], color='grey')
        ax1.set_ylabel('log(count) scRNAseq')
        ax1.set_xticks(np.arange(len(sc_genes)),
                       sc_genes[sc_count_idcs],
                       rotation=90)
        ax1.set_yscale('log')

        ax2 = plt.subplot(312)
        for i, gene in enumerate(sc_genes[sc_count_idcs]):
            plt.plot(
                [i, self.sdata.stats.get_count_rank(gene)],
                [1, 0],
            )
        plt.axis('off')
        ax2.set_ylabel(' ')

        ax3 = plt.subplot(313)
        self.sdata.plot_bars(ax3, color='grey')
        ax3.invert_yaxis()
        ax3.xaxis.tick_top()
        ax3.xaxis.set_label_position('top')
        ax3.set_ylabel('log(count) spatial')

    def score_affinity(self, labels_1, labels_2=None, scanpy_obs_label='celltype'):

        if labels_2 is None:
            labels_2 = (self.adata.obs[~self.adata.obs[scanpy_obs_label].isin(labels_1)])[
                scanpy_obs_label]

        mask_1 = self.adata.obs[scanpy_obs_label].isin(labels_1)
        mask_2 = self.adata.obs[scanpy_obs_label].isin(labels_2)

        samples_1 = self.adata[mask_1, ]
        samples_2 = self.adata[mask_2, ]

        counts_1 = np.array(samples_1.X.mean(0)).flatten()
        counts_2 = np.array(samples_2.X.mean(0)).flatten()

        return np.log((counts_1+0.1)/(counts_2+0.1))


class GeneStatistics(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(GeneStatistics, self).__init__(*args, **kwargs)

    @property
    def counts_sorted(self):
        return self.data.counts[self.stats.count_indices]

    @property
    def genes(self):
        return self.index

    def get_count(self, gene):
        if gene in self.genes.values:
            return int(self.counts[self.genes == gene])

    def get_id(self, gene_name):
        return int(self.gene_ids[self.genes == gene_name])

    def get_count_rank(self, gene):
        if gene in self.genes.values:
            return int(self.count_ranks[self.genes == gene])


class PointStatistics(GeneStatistics):
    def __init__(self, sdata):
        genes, indicers, inverse, counts = np.unique(
            sdata['g'],
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        count_idcs = np.argsort(counts)
        count_ranks = np.argsort(count_idcs)

        super(PointStatistics, self).__init__(
            {
                'counts': counts,
                'count_ranks': count_ranks,
                'count_indices': count_idcs,
                'gene_ids': np.arange(len(genes))
            },
            index=genes)

        sdata['gene_id'] = inverse
        self.sdata = sdata

        sdata.graph = SpatialGraph(self)

    def co_occurrence(self, category=None, resolution=5, max_radius=None, linear_steps=20,):
        return co_occurrence(self.sdata, resolution=resolution, max_radius=max_radius, linear_steps=linear_steps, category=category)


class ScStatistics(GeneStatistics):

    def __init__(self, scanpy_df):

        counts = np.array(scanpy_df.adata.X.sum(0)).squeeze()
        genes = scanpy_df.adata.var.index

        count_idcs = np.argsort(counts)
        count_ranks = np.argsort(count_idcs)

        super(ScStatistics, self).__init__(
            {
                'counts': counts,
                'count_ranks': count_ranks,
                'count_indices': count_idcs,
                'gene_ids': np.arange(len(genes))
            },
            index=genes)


class SpatialIndexer():

    def __init__(self, df):
        self.sdata = df

    @property
    def shape(self):
        if self.sdata.background is None:
            return np.ceil(self.sdata.x.max() - self.sdata.x.min()).astype(
                int), np.ceil(self.sdata.y.max() - self.sdata.y.min()).astype(int)
        else:
            return self.sdata.background.shape

    def create_cropping_mask(self, start, stop, series):

        if start is None:
            start = 0

        if stop is None:
            stop = series.max()

        return ((series > start) & (series < stop))

    def join_cropping_mask(self, xlims, ylims):
        return self.create_cropping_mask(
            *xlims, self.sdata.x) & self.create_cropping_mask(*ylims, self.sdata.y)

    def crop(self, xlims, ylims):

        mask = self.join_cropping_mask(xlims, ylims)

        pixel_maps = []

        if xlims[0] is None:
            start_x = 0
        else:
            start_x = xlims[0]
        if ylims[0] is None:
            start_y = 0
        else:
            start_y = ylims[0]

        for pm in self.sdata.pixel_maps:
            pixel_maps.append(pm[xlims[0]:xlims[1], ylims[0]:ylims[1]])

        if self.sdata.scanpy is not None:
            adata = self.sdata.scanpy.adata
        else:
            adata = None

        return SpatialData(self.sdata.g[mask],
                           self.sdata.x[mask] - start_x,
                           self.sdata.y[mask] - start_y, pixel_maps,
                           adata, self.sdata.synchronize)

    def __getitem__(self, indices):

        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices, )
        if len(indices) == 1:
            ylims = (0, None)
        else:
            ylims = (indices[1].start, indices[1].stop)

        xlims = (indices[0].start, indices[0].stop)

        return self.crop(xlims, ylims)


class ObscDF(pd.DataFrame):
    """ObscDF _summary_

    :param pd: _description_
    :type pd: _type_
    """

    def __init__(self, sdata, assign_colors=True):
        super(ObscDF, self).__init__(index=sdata.stats.index)
        self['genes'] = self.index
        if assign_colors:
            self.assign_colors()
        self.sdata = sdata

    def assign_colors(self, label='genes', cmap=None, shuffle=False):

        # if label is None:
        #     uniques = self.index
        # else:
        uniques = self[label].unique()

        if cmap is None:
            cmap = get_cmap('nipy_spectral')
            clrs = [cmap(f) for f in np.linspace(0.07, 1, len(uniques))]

        else:
            cmap = get_cmap(cmap)
            clrs = [cmap(f) for f in np.linspace(0, 1, len(uniques))]

        if shuffle:
            random.shuffle(clrs)

        clrs = {u: clrs[i] for i, u in enumerate(uniques)}
        self['c_'+label] = self[label].apply(lambda x: clrs[x])
        # print(len(uniques),len(clrs),self.shape)
        # if label is None:
        #     self['c_genes']=clrs
        # else:
        # clr_list=[(0,0,0,0,)]*len(self)
        # for i,u in enumerate(uniques):
        # print([clrs[i]]*sum(self[label]==u))

        # self.loc[self[label]==u,'c_'+label]=[[clrs[i]]]*sum(self[label]==u)
        # self[]

    def project(self, label):
        return(self.loc[self.sdata.g][label])

    def copy(self):
        return ObscDF(self.sdata)


class SpatialData(pd.DataFrame):

    def __init__(self,
                 genes,
                 x_coordinates,
                 y_coordinates,
                 pixel_maps=[],
                 scanpy=None,
                 synchronize=True,
                 obsc=None):

        # Initiate 'own' spot data:
        super(SpatialData, self).__init__({
            'g': genes,
            'x': x_coordinates,
            'y': y_coordinates
        })

        self['g'] = self['g'].astype('category')

        # Initiate pixel maps:
        self.pixel_maps = []
        self.stats = PointStatistics(self)

        if obsc is None:
            self.obsc = ObscDF(self)
        else:
            self.obsc = obsc

        self.graph = SpatialGraph(self)

        for pm in pixel_maps:
            if not type(pm) == PixelMap:
                self.pixel_maps.append(PixelMap(pm))
            else:
                self.pixel_maps.append(pm)

        self.synchronize = synchronize

        # Append scanpy data set, synchronize both:
        if scanpy is not None:
            self.scanpy = ScanpyDataFrame(self, scanpy)
            if self.synchronize:
                self.sync_scanpy()
        else:
            self.scanpy = None

        # self.obsm = {"spatial":np.array(self.coordinates).T}
        # self.obs = pd.DataFrame({'gene':self.g})
        self.uns = {}

    @property
    def gene_ids(self):
        return self.gene_id

    @property
    def coordinates(self):
        return np.array([self.x, self.y]).T

    @property
    def counts(self):
        return self.stats['counts']

    @property
    def counts_sorted(self):
        return self.stats.counts[self.stats.count_indices]

    @property
    def genes_sorted(self):
        return self.genes[self.stats.count_indices]

    @property
    def genes(self):
        return self.stats.index

    @property
    def spatial(self):
        return SpatialIndexer(self)

    @property
    def background(self):
        if len(self.pixel_maps):
            return self.pixel_maps[0]

    @property
    def adata(self):
        if self.scanpy is not None:
            return self.scanpy.adata

    @property
    def X(self):
        return scipy.sparse.csc_matrix((np.ones(len(self.g),), (np.arange(len(self.g)), np.array(self.gene_ids).flatten())),
                                       shape=(len(self.g), self.genes.shape[0],))

    @property
    def var(self):
        return pd.DataFrame(index=self.stats.genes)

    @property
    def obs(self):
        return pd.DataFrame({'gene': self.g}).astype(str).astype('category')

    @property
    def obsm(self):
        return {"spatial": np.array(self.coordinates).T}

    def __getitem__(self, *arg):

        if (len(arg) == 1):

            if type(arg[0]) == str:

                return super().__getitem__(arg[0])

            if (type(arg[0]) == slice):
                new_data = super().iloc.__getitem__(arg)

            elif (type(arg[0]) == int):
                new_data = super().iloc.__getitem__(slice(arg[0], arg[0] + 1))

            elif isinstance(arg[0], pd.Series):
                # print(arg[0].values)
                new_data = super().iloc.__getitem__(arg[0].values)

            elif isinstance(arg[0], np.ndarray):
                new_data = super().iloc.__getitem__(arg[0])

            elif isinstance(arg[0], collections.Sequence):
                if all([a in self.keys() for a in arg[0]]):
                    return super().__getitem__(*arg)
                new_data = super().iloc.__getitem__(arg[0])

            if self.scanpy is not None:
                scanpy = self.scanpy.adata
                synchronize = self.scanpy.synchronize
            else:
                scanpy = None
                synchronize = None

            new_frame = SpatialData(new_data.g,
                                    new_data.x,
                                    new_data.y,
                                    self.pixel_maps,
                                    scanpy=scanpy,
                                    synchronize=synchronize,
                                    obsc=self.obsc.copy())

            new_prop_entries = self.obsc.loc[new_frame.genes]
            new_frame.obsc[new_prop_entries.columns] = new_prop_entries
            new_frame.obsc.sdata = new_frame
            new_frame.obsc.drop(self.genes.symmetric_difference(
                new_frame.genes), inplace=True)

            if self.graph._umap is not None:
                new_frame.graph._umap = self.graph._umap[self.index.isin(
                    new_frame.index)]

            return (new_frame)

        print('Reverting to generic Pandas.')
        return super().__getitem__(*arg)

    def sync_scanpy(self,
                    mRNA_threshold_sc=1,
                    mRNA_threshold_spatial=1,
                    verbose=False,
                    anndata=None):
        if anndata is None and self.scanpy is None:
            print('Please provide some scanpy data...')

        if anndata is not None:
            self.scanpy = ScanpyDataFrame(anndata)
        else:
            self.scanpy.synchronize()

    def get_id(self, gene_name):
        return int(self.stats.gene_ids[self.genes == gene_name])

    def scatter(self,
                c=None,
                color=None,
                color_prop='genes',
                marker='.',
                legend=None,
                axd=None,
                plot_bg=True,
                cmap='jet',
                scalebar=True,
                **kwargs):

        if axd is None:
            axd = plt.gca()

        handle_imshow = None
        handle_legend = None

        if self.background and plot_bg:
            handle_imshow = self.background.imshow(
                cmap='Greys', axd=axd)

        if c is None:
            if color is None:
                c = self.obsc.project('c_'+color_prop)

        if legend:
            labels = sorted(self.obsc[color_prop].unique())
            # print(labels)
            clrs = [self.obsc[(self.obsc[color_prop] == l)]
                    ['c_'+color_prop][0] for l in labels]

            handles = [plt.scatter([], [], color=c) for c in clrs]
            handle_legend = plt.legend(handles, labels, loc='right')

        # axd.set_title(gene)
        handle_scatter = axd.scatter(self.x,
                                     self.y,
                                     c=c,
                                     marker=marker,
                                     color=color,
                                     cmap=cmap,
                                     **kwargs)

        if scalebar:
            self.add_scalebar(axd=axd)

        return handle_scatter, handle_imshow, handle_legend

    def add_scalebar(self, length=None, unit=r'$\mu$m', axd=None, color='w'):

        if axd is None:
            axd = plt.gca()

        x_, _x = axd.get_xlim()
        y_, _y = axd.get_ylim()

        if length is None:
            length = (_x-x_)*0.2
            decimals = np.ceil(np.log10(length))-1
            inter = length/10**decimals
            # int(np.around(_x-x_,1-int(np.log10((_x-x_)/2))))
            length = np.ceil(inter) * 10**decimals

        _x_ = _x-x_
        _y_ = _y-y_

        new_x_ = x_ + _x_/20
        new__x = new_x_+length
        new_y = y_ + _y_/20

        scbar = plt.Line2D([0.1, 0.9], [0.5, 0.5], c='w', marker='|', linewidth=2,
                           markeredgewidth=3, markersize=10, color=color)
        scbar.set_data([new_x_, new__x], [new_y, new_y])
        scbar.set_path_effects(
            [PathEffects.withStroke(linewidth=4, foreground='k')])

        magnitudes = ['nm', r'$\mu$m', 'mm', 'm', 'km']
        bar_label = f'{int(length%999)}{magnitudes[np.floor(np.log10(length)/3+1).astype(int)]}'

        sctxt = axd.text((new_x_+new__x)/2, (y_+_y_/15), bar_label,
                         fontweight='bold', ha='center', color=color)

        axd.add_artist(scbar)
        sctxt.set_path_effects(
            [PathEffects.withStroke(linewidth=3, foreground='k')])

        return scbar, sctxt

    def plot_bars(self, axis=None, **kwargs):
        if axis is None:
            axis = plt.subplot(111)
        axis.bar(np.arange(len(self.stats.counts)), self.counts_sorted,
                 **kwargs)
        axis.set_yscale('log')

        axis.set_xticks(
            np.arange(len(self.genes_sorted)),
            list(self.genes_sorted),
        )
        axis.tick_params(axis='x', rotation=90)
        axis.set_ylabel('molecule count')

    def plot_overview(self):

        colors = ('royalblue', 'goldenrod', 'red', 'lime')

        scatter_idcs = np.round(np.linspace(0,
                                            len(self.stats.counts) - 1,
                                            4)).astype(int)

        fig, axd = plt.subplot_mosaic(
            [['scatter_1', 'scatter_2', 'scatter_3', 'scatter_4'],
             ['bar', 'bar', 'bar', 'bar']],
            figsize=(11, 7),
            constrained_layout=True)

        self.plot_bars(
            axd['bar'],
            color=[
                colors[np.where(
                    scatter_idcs == i)[0][0]] if i in scatter_idcs else 'grey'
                for i in range(len(self.stats.counts))
            ])

        for i in range(4):
            idx = self.stats.count_indices[scatter_idcs[i]]
            gene = self.genes[idx]
            plot_name = 'scatter_' + str(i + 1)
            axd[plot_name].set_title(gene)
            axd[plot_name].scatter(self.x, self.y, color=(0.5, 0.5, 0.5, 0.1))
            axd[plot_name].scatter(self.x[self['gene_id'] == idx],
                                   self.y[self['gene_id'] == idx],
                                   color=colors[i],
                                   marker='.')

            axd[plot_name].set_xticks([], [])
            # if i>0:
            axd[plot_name].set_yticks([], [])

            con = ConnectionPatch(xyA=(scatter_idcs[i],
                                       self.stats.counts[idx]),
                                  coordsA=axd['bar'].transData,
                                  xyB=(np.mean(axd[plot_name].get_xlim()),
                                       axd[plot_name].get_ylim()[0]),
                                  coordsB=axd[plot_name].transData,
                                  color='white',
                                  linewidth=1,
                                  linestyle='dotted')
            fig.add_artist(con)

        plt.suptitle('Selected Expression Densities:', fontsize=18)

    def squidpy(self):
        # obs={"cluster":self.gene_id.astype('category')}
        obsm = {"spatial": np.array(self.coordinates)}
        # var= self.genes
        # self.obs = self.index
        # X = self.X #scipy.sparse.csc_matrix((np.ones(len(self.g),),(np.arange(len(self.g)),np.array(self.gene_ids).flatten())),
        # shape=(len(self.g),self.genes.shape[0],))

        # sparse_representation = scipy.sparse.scr()
        # var = self.var #pd.DataFrame(index=self.genes)
        uns = self.uns.update({'Image': self.background})
        obs = pd.DataFrame({'gene': self.g})
        obs['gene'] = obs['gene'].astype('category')
        return anndata.AnnData(X=self.X, obs=obs, var=self.var, obsm=obsm)

    def save(self, path):
        pickle.dump({'sdata': self, 'graph': self.graph,
                    'obsc': self.obsc, 'pixel_maps': self.pixel_maps}, open(path, "wb"))

# here starts plotting.py


def load(path):
    data = pickle.load(open(path, "rb"))
    sdata = SpatialData(data['sdata']['g'], data['sdata']
                        ['x'], data['sdata']['y'], pixel_maps=data['pixel_maps'])
    print(data['sdata'].columns)

    for i, c in enumerate(data['sdata'].columns[4:]):
        sdata[c] = data['sdata'][c]
    sdata.graph = data['graph']
    sdata.graph.sdata = sdata
    sdata.obsc = data['obsc']
    sdata.obsc.sdata = sdata
    return sdata


def create_colorarray(sdata, values, cmap=None):
    if cmap is None:
        return values[sdata.gene_ids]


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


