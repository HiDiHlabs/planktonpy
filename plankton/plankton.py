
from __future__ import annotations
from tokenize import group

from matplotlib import cm
# from sympy import N
from plankton.graph import SpatialGraph
from plankton.pixelmaps import PixelMap, PixelMask
import numpy as np
import pandas as pd
import random
import scipy
import anndata
from plankton.stats import co_occurrence
import scanpy as sc
import collections
from scipy import sparse
import pickle

import matplotlib
from matplotlib.cm import get_cmap
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as PathEffects
from matplotlib import pyplot as plt


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)



plt.style.use('dark_background')
matplotlib.rcParams['figure.figsize'] = (15, 15)


class ScanpyDataFrame():
    """A plankton.py wrapper for scanpy-derived annotated data frames that provides an interface with many of the planktonpy functionalities.

    :param sdata: SpatialData the spatial data object that this ScanpyDataFrame 
    :type sdata: SpatialData
    :param scanpy_ds: the scanpy AnnData object containing the data set
    :type scanpy_ds: :class: anndata.AnnData
    """

    def __init__(self, sdata, scanpy_ds):
        """__init__ initialization method.

        """
        self.sdata = sdata
        self.adata = scanpy_ds
        self.stats = ScStatistics(self)
        self.celltype_labels = None
        self.signature_matrix = None

    @property
    def shape(self):
        return self.adata.shape

    @property
    def obs(self):
        return self.adata.obs
    
    @property
    def var(self):
        return self.adata.var

    @property
    def obsm(self):
        return self.adata.obsm

    @property
    def X(self):
        return self.adata.X

    def generate_signatures(self, celltype_obs_marker='celltype'):
        """generate_signatures Generates a signature matrix of gene->celltype correlations.

        :param celltype_obs_marker: column given in the 'obs' data field denoting the cell type, defaults to 'celltype'
        :type celltype_obs_marker: str, optional
        :return: signature matrix
        :rtype: pandas.DataFrame
        """
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
        """synchronizes the ScanpyDataFrame with its SpatialData reference objects. Unused genes are removed, statistics updated and sorted.
        """

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

    def _repr_html_(self):

        magma = get_cmap('Greys')

        html = """  <style> 
                    .column-header{font-weight:bold;writing-mode:vertical-rl;text-orientation:mixed;}
                    .row-header{font-weight:bold;}
                    .x-value{border:collapse}
                    </style> <table>"""


        html+="<tr><td></td>"
        for i,h in enumerate(self.obs.index.values[:10]):
            html+=f'<td><div class="column-header">{h}</div></td>'

        html+='<td>...</td>'        

        for i,h in enumerate(self.obs.index.values[-10:]):
            html+=f'<td><div class="column-header">{h}</div></td>'

        for i,k in enumerate(self.obs.keys()):  
            html+="<tr>"
            html+=f"<td>{k}</td>"
            for j,v in enumerate(self.obs[k].values[:10]):
                html+=f"<td><div style='width:40px;overflow-wrap: break-word; '>{v}</div></td>"
            html+="<td>...</td>"

            for j,v in enumerate(self.obs[k].values[-10:]):
                html+=f"<td><div style='width:40px;overflow-wrap: break-word; '>{v}</div></td>"
            html+="</tr>"          


        html+="</tr>"

        X = self.X
        X_norm=X-X.min()
        X_norm/=X_norm.max()
        X_norm = X_norm.astype(float)
        X_norm = (X_norm)

        def return_cell(x):
            color='rgba'+''.join(str(magma(float(x)**0.8,bytes=True)).split(' '))
            return f"""<td bgcolor="{color}"><div class='x-value' style='width:100%;height:100%;color:white;font-weight:bold;'>{x:.2f}</div></td>"""

        for i,v in enumerate(self.var.index.values[:10]):
            html+=f"<tr><td><div class='row-header'>{v}</div>"
            html+=''.join(list(map(return_cell,list(X_norm[:10,i]   ))))
            if X_norm.shape[0]>20:
                html+= """<td bgcolor="white"><div class='x-value' style='width:100%;height:100%;'> ...</div></td>"""
            # html+=f"<tr><td><div class='row-header'>{v}</div>"
            html+=''.join(list(map(return_cell,list(X_norm[-10:,i]   ))))
            html+="</tr>"
        html+= "<tr>"+ """<td bgcolor="white"><div class='x-value' style='width:100%;height:100%;'> ...</div></td>"""*21+"</tr>"

        for i in range(-10,-1):
            html+=f"<tr><td><div class='row-header'>{self.var.index.values[i]}</div>"
            html+=''.join(list(map(return_cell,list(X_norm[:10,i]   ))))
            if X_norm.shape[0]>20:
                html+= """<td bgcolor="white"><div class='x-value' style='width:100%;height:100%;'> ...</div></td>"""
            html+=''.join(list(map(return_cell,list(X_norm[-10:,i]   ))))
            html+="</tr>"
            
        return html+"</table>"


class GeneStatistics(pd.DataFrame):
    """GeneStatistics Base class for the different stats data frames containing gene count data.

    """
    def __init__(self, *args, **kwargs):
        super(GeneStatistics, self).__init__(*args, **kwargs)

    @property
    def counts_sorted(self):
        return self.data.counts[self.stats.count_indices]

    @property
    def genes(self):
        return self.index

    def get_count(self, gene):
        """get_count returns the count for a given gene.

        :param gene: gene name
        :type gene: str
        :return: count value
        :rtype: int
        """
        if gene in self.genes.values:
            return int(self.counts[self.genes == gene])

    def get_id(self, gene_name):
        """get_id gets the id value for a certain gene

        :param gene_name: gene name
        :type gene_name: str
        :return: id value
        :rtype: int
        """
        return int(self.gene_ids[self.genes == gene_name])

    def get_count_rank(self, gene):
        if gene in self.genes.values:
            return int(self.count_ranks[self.genes == gene])

    def syncronized(self, stats):
        """syncronized returns a synchronized sdata count list for further statistical processing.
        NEEDS UPDATING!

        :param stats: external stats object to compare against
        :type stats: :class: plankton.GeneStatistics
        :return: concatenated count data frame
        :rtype: :class: pandas.DataFrame
        """
        return pd.concat([self,stats], axis=1).fillna(0)


class PointStatistics(GeneStatistics):
    """PointStatistics _summary_

    :param sdata: :class: SpatialData
    :type GeneStatistics: _type_
    """
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

    def co_occurrence(self, resolution=5, max_radius=None, linear_steps=20, category=None,):
        """co_occurrence Generate a co-occurrence matrix for this data set's source data.

        :param resolution: Smallest resolution of the co-occurence model in um, defaults to 5.
        :type resolution: float, optional
        :param max_radius: Largest radius of the co-occurrence model in um, defaults to 400.
        :type max_radius: float, optional
        :param linear_steps: Number of linear steps to model local heterogeniety. Afterwards, distance bins get wider to save computational resources, defaults to 5
        :type linear_steps: int, optional
        :param category: Category to model the co-occurrence of. Must be a column in 'sdata' with dtype 'category'. When given 'None', the algorithm defaults to gene labels.
        :rtype: str
        :return: co-occurrence matrix 
        :rtype: pandas.core.frame.DataFrame
        """
        return co_occurrence(self.sdata, resolution=resolution, max_radius=max_radius, linear_steps=linear_steps, category=category)


class ScStatistics(GeneStatistics):
    """ScStatistics Statistics object for ScanpyDataFrame

    :param sdata: Source data frame 
    :type sdata: plankton.ScanpyDataFrame
    """

    def __init__(self, sdata):

        counts = np.array(sdata.adata.X.sum(0)).squeeze()
        genes = sdata.adata.var.index

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
    """SpatialIndexer implements spatial view of a source plankton.SpatialData frame. It can be used for simple cropping and to extract spatial size information.

    :param sdata: source data
    :type sdata: plankton.SpatialData
    """

    def __init__(self, sdata):
        self.sdata = sdata

    @property
    def shape(self):
        """shape returns the shape of the data (instead of the dataframe shape). When available, the function reports the size of a (cropped) dataset background map. Otherwise, the extreme values in the x- and y- directions are returned.

        :return: Spatial extent/shape of the data.
        :rtype: tuple(int,int)
        """
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
        """crop crops the source data accordng to its spatial extent:

        :param xlims: limits in the x-direction
        :type xlims: (float,float)
        :param ylims: limits in the y-direction
        :type ylims: (float,float)
        :return: New, cropped data set.
        :rtype: plankton.SpatialData
        """

        mask = self.join_cropping_mask(xlims, ylims)

        pixel_maps = {}

        if xlims[0] is None:
            start_x = 0
        else:
            start_x = xlims[0]
        if ylims[0] is None:
            start_y = 0
        else:
            start_y = ylims[0]

        for key,pm in self.sdata.pixel_maps.items():
            pixel_maps[key] = (pm[xlims[0]:xlims[1], ylims[0]:ylims[1]])

        if self.sdata.scanpy is not None:
            adata = self.sdata.scanpy.adata
        else:
            adata = None

        return SpatialData(self.sdata.g[mask],
                           self.sdata.x[mask] - start_x,
                           self.sdata.y[mask] - start_y, pixel_maps,
                           adata, self.sdata.synchronize)

    def __getitem__(self, indices):
        """__getitem__ cropping interface for the source data set.

        :param indices: cropping indices.
        :type indices:  slice, iterable
        :return: cropped source data
        :rtype: plankton.SpatialData
        """

        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices, )
        if len(indices) == 1:
            ylims = (0, None)
        else:
            ylims = (indices[1].start, indices[1].stop)

        xlims = (indices[0].start, indices[0].stop)

        return self.crop(xlims, ylims)


class ObscDF(pd.DataFrame):
    """ObscDF A dataframe encoding gene-centric observations.
    
    :param sdata: Source data
    :type sdata: plankton.SpatialData
    :param assign_colors: initialize a column with color-values for all genes, defaults to True
    :type assign_colors: bool, optional    
    """

    def __init__(self, sdata, assign_colors=True):

        super(ObscDF, self).__init__(index=sdata.stats.index)
        self['genes'] = self.index
        if assign_colors:
            self.assign_colors()
        self.sdata = sdata

    def assign_colors(self, label='genes', cmap=None, shuffle=False):
        """assign_colors generates colors for a given label, which are stored in a column 'c_'+label

        :param label: column label to generate colors for, defaults to 'genes'
        :type label: str, optional
        :param cmap: colormap to pick colors from, defaults to None/'nipy_spectral'
        :type cmap: _type_, optional
        :param shuffle: Shuffle the colors randomly before assignment, defaults to False
        :type shuffle: bool, optional
        """

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


    def project(self, label):
        """project Projects a column onto the gene composition of the source data set.

        :param label: Label of column to project.
        :type label: str
        """
        return(self.loc[self.sdata.g][label])

    def copy(self):
        return ObscDF(self.sdata)


class SpatialData(pd.DataFrame):
    """SpatialData A data handling interface for Spot-Based spatial data.

    :param genes: A list of gene indicators
    :type genes: list(str)
    :param x_coordinates: a list of x-coordinates in micrometers.
    :type x_coordinates: list(float)
    :param y_coordinates: a list of y-coordinates in micrometers
    :type y_coordinates: list(float)
    :param pixel_maps: A dictionary of pixelmaps, can be used to define background images during initialization. defaults to {}
    :type pixel_maps: dict(str,plankton.pixelmaps.PixelMap), optional
    :param scanpy: An optional input for single-cell derived expression data. defaults to None
    :type scanpy: plankton.ScanpyDataFrame, optional
    :param synchronize: Whether the different data modalities should be cropped and synchronized at all times. defaults to True
    :type synchronize: bool, optional
    :param obsc: A data frame for gene-centric information, defaults to None
    :type obsc: plankton.ObscDF, optional
    """

    def __init__(self,
                 genes,
                 x_coordinates,
                 y_coordinates,
                 pixel_maps={},
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

        self.stats = PointStatistics(self)

        if obsc is None:
            self.obsc = ObscDF(self)
        else:
            self.obsc = obsc

        self.graph = SpatialGraph(self)

        # Initiate pixel maps:
        self.pixel_maps = {}
        self.bg_key = None

        for key, pm in pixel_maps.items():
            if self.bg_key is None: self.bg_key=key

            if not type(pm) == PixelMap:
                self.pixel_maps[key]=(PixelMap(pm))
            else:
                self.pixel_maps[key]=(pm)


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
            return self.pixel_maps[self.bg_key]

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

            for c in self.columns[4:]:
                new_frame[c]=new_data[c]

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
        """sync_scanpy Synchronizes the SpatialData set with its provided single-cell expression frame. Unpaired genes are removed and statistics re-calculated.

        :param mRNA_threshold_sc: Count below which a gene in the single-cell data set is removed, defaults to 1
        :type mRNA_threshold_sc: int, optional
        :param mRNA_threshold_spatial: Count below which a gene in the spatial data set is removed, defaults to 1
        :type mRNA_threshold_spatial: int, optional
        :param verbose: Displat warnings (legacy), defaults to False
        :type verbose: bool, optional
        :param anndata: _description_, defaults to None
        :type anndata: Provide a ScanpyDataFrame if the field 'self.scanpy' is empty, optional
        """
        if anndata is None and self.scanpy is None:
            print('Please provide some scanpy data...')

        if anndata is not None:
            self.scanpy = ScanpyDataFrame(anndata)
        else:
            self.scanpy.synchronize()

    def get_id(self, gene_name):
        """get_id Gets the dataframe-id to a certain gene

        :param gene_name: Gene name
        :type gene_name: str
        :return: Gene's id number that's shared across data sets/data modalities
        :rtype: int
        """
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
        """scatter Leightweight implementation of a scatter plot, using maplotlib as a backend.

        :param c: Defines an individual color for each data point, either through a list of float scale values (color is sampled for a colormap) or a list of matplotlib-interpretable colors. defaults to None
        :type c: list(float),list(rgb), optional
        :param color: Marker color, defaults to None
        :type color: str, optional
        :param color_prop: column name in self.obsc, used as a color source, defaults to 'genes'
        :type color_prop: str, optional
        :param marker: Marker shape, defaults to '.'
        :type marker: str, optional
        :param legend: Whether to add a legend artist, defaults to None
        :type legend: bool, optional
        :param axd: Provide an axis object that the artists are added to. If none is provided, the current active axis is used. Defaults to None
        :type axd: matplotlib.pyplot.Asix, optional
        :param plot_bg: Whether to plot the background image, defaults to True
        :type plot_bg: bool, optional
        :param cmap: Define the colormap that colors are sampled from if a 'c' argument is provided, defaults to 'jet'
        :type cmap: str, optional
        :param scalebar: Whether to print a scalebar in micrometers, defaults to True
        :type scalebar: bool, optional
        :return: handles for the scatters,background and legend.
        :rtype: tuple
        """

        if axd is None:
            axd = plt.gca()

        axd.set_aspect('equal', adjustable='box')

        handle_imshow = None
        handle_legend = None

        if self.background and plot_bg:
            handle_imshow = self.background.imshow(
                cmap='Greys', axd=axd)
        else:
            axd.invert_yaxis()

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
        """add_scalebar: Adds a scalebar to a plot.

        :param length: Length of the scalebar, defaults to None
        :type length: float, optional
        :param unit: Unit that's displayed on the scalebar, defaults to r'$\mu'
        :type unit: str, optional
        :param axd: Axis to draw on, if none is proveded, the current active axis is used. Defaults to None
        :type axd:  matplotlib.pyplot.Axis, optional
        :param color:, Font/bar color. defaults to 'w'
        :type color: str, optional
        :return: handler for scalebar and text.
        :rtype: tuple
        """

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
        """plot_bars: Plots a sorted bar  graph representation of the gene counts in the sample.

        :param axis: Axis to draw on, if none is proveded, the current active axis is used. Defaults to None
        :type axis: matplotlib.pyplot.Axis, optional
        """
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
        """plot_overview: Plots a lightweight overview of the spatial data set. A sorted bar representation of molecule counts is shown, alongside a few representative scatter plots of individual genes in the sample.
        """

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

    def squidpy(self, groupby=None):
        """squidpy: Generates an anndata.AnnData based squidpy-compatible object from the spatial data, including pixel maps and obervation matrices.

        :param groupby: Name of a dataset column (of type 'category') that denotes membership to certain spatial entities (such as cells) by which to agglomerate the data. I none is provided, every data point is grouped individually. Defaults to None
        :type groupby: str, optional
        :return: squidpy-compatible spatial data set. 
        :rtype: anndata.AnnData
        """
        # obs={"cluster":self.gene_id.astype('category')}

        if groupby is None:
            obsm = {"spatial": np.array(self.coordinates)}

            uns = self.uns.update({'Image': self.background})
            obs = pd.DataFrame({'gene': self.g})
            obs['gene'] = obs['gene'].astype('category')
            return anndata.AnnData(X=self.X, obs=obs, var=self.var, obsm=obsm)

        else:


            uns = self.uns.update({'Image': self.background})
            obs = pd.DataFrame({groupby: self[groupby].unique()})
            obsm = {"spatial": np.zeros((len(obs),2))}
            var=self.var

            X=np.zeros((len(obs),len(var)))

            for i,o in enumerate(obs[groupby].values):
                subset = self[self[groupby]==o]
                obsm['spatial'][i]=subset.coordinates.mean(0)
                locs = [self.genes.get_loc(i) for i in subset.counts.index]
                X[i,locs] = subset.counts.values
                
            return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)


    def save(self, path):
        """save: Save a deep copy of the spatial data, including graphs and pixelmaps.

        :param path: Save path/filename
        :type path: str
        """
        pickle.dump({'sdata': self, 'graph': self.graph,
                    'obsc': self.obsc, 'pixel_maps': self.pixel_maps}, open(path, "wb"))

# here starts plotting.py


def load(path):
    """load: Loads a spatial data set from a disk file.

    :param path: File path/name
    :type path: str
    :return: spatial data set
    :rtype: plankton.SpatialData
    """
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
