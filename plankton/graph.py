
from turtle import color
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import Button, TextBox
import matplotlib.patheffects as PathEffects

import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets, interactive, HBox, VBox, Output, Layout

import pandas as pd
from scipy.stats import binom
from sklearn.decomposition import PCA, FastICA

from sklearn.neighbors import NearestNeighbors

from umap import UMAP


class SpatialGraph():
    """SpatialGraph: Container object for all KNN-graph based operations on spatial data sets.

    :param sdata: The spatial data source 
    :type sdata: plankton.SpatialData
    :param n_neighbors: number of nearset neighbors to infer, defaults to 10
    :type n_neighbors: int, optional
    """
    def __init__(self, sdata, n_neighbors=10) -> None:

        self.sdata = sdata
        self.n_neighbors = n_neighbors
        self._neighbors = None
        self._neighbor_types = None
        self._distances = None
        self._umap = None
        self._tsne = None

    @property
    def neighbors(self):
        """neighbors: Returns an array of neighbors, where each entry encoding the row position of a point in the data set.

        :return: 2d array of n-neighbors per sample data point
        :rtype: numpy.Array
        """
        if self._neighbors is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._neighbors[:, :self.n_neighbors]

    @property
    def distances(self):
        """distances: Returns an array of inter-spot distances, which each entry encoding the distance to a defined neighbor. 

        :return: 2d array of distances to the data points of self.neighbors.
        :rtype: numpy.Array
        """
        if self._distances is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._distances[:, :self.n_neighbors]

    @property
    def neighbor_types(self):
        """neighbor_types: Returns an array of gene_ids for the spots listed in self.neighbors

        :return: 2d array of gene_ids.
        :rtype: numpy.Array
        """
        if self._neighbor_types is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._neighbor_types[:, :self.n_neighbors]

    @property
    def umap(self):
        """umap: Returns the coordinates of the UMAP representation of the source data.

        :return: UMAP coordinates
        :rtype: numpy.Array
        """
        if self._umap is None:
            print("No UMAP calculated at this point. Please call 'run_umap' first.")
        return self._umap

    def __getitem__(self, *args):
        sg = SpatialGraph(self.sdata, self.n_neighbors)
        if self._distances is not None:
            sg._distances = self._distances.__getitem__(*args)
        if self._neighbors is not None:
            sg._neighbors = self._neighbors.__getitem__(*args)
        if self._neighbor_types is not None:
            sg._neighbor_types = self._neighbor_types.__getitem__(*args)

    def update_knn(self, n_neighbors, re_run=True):
        """update_knn: Updates the KNN representation that gives rise to self.neighbors, self.distances and self.neighbor_types. This does not happen automatically after slicing, so it should be performed regularity before any graph-based analysis. Can be memory-intensive.

        :param n_neighbors: Number of neighbors for each neighbor graph.
        :type n_neighbors: int
        :param re_run: re-run, or use current representation, defaults to True
        :type re_run: bool, optional
        """

        if self._neighbors is not None and (n_neighbors <
                                            self._neighbors.shape[1]):
            self.n_neighbors = n_neighbors
            # return (self._neighbors[:, :n_neighbors],
            #         self._distances[:, :n_neighbors],
            #         self._neighbor_types[:, :n_neighbors])
        else:

            coordinates = np.stack([self.sdata.x, self.sdata.y]).T
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            knn.fit(coordinates)
            self._distances, self._neighbors = knn.kneighbors(coordinates)
            self._neighbor_types = np.array(self.sdata.gene_ids)[
                self._neighbors]

            self.n_neighbors = n_neighbors

            # return self.distances, self.neighbors, self.neighbor_types

    def knn_entropy(self, n_neighbors=4):

        self.update_knn(n_neighbors=n_neighbors)
        indices = self.neighbors  # (n_neighbors=n_neighbors)

        knn_cells = np.zeros_like(indices)
        for i in range(indices.shape[1]):
            knn_cells[:, i] = self.sdata['gene_id'].iloc[indices[:, i]]

        H = np.zeros((len(self.sdata.genes), ))

        for i, gene in enumerate(self.sdata.genes):
            x = knn_cells[self.sdata['gene_id'] == i]
            _, n_x = np.unique(x[:, 1:], return_counts=True)
            p_x = n_x / n_x.sum()
            h_x = -(p_x * np.log2(p_x)).sum()
            H[i] = h_x

        return (H)

    def plot_entropy(self, n_neighbors=4):

        H = self.knn_entropy(n_neighbors)

        idcs = np.argsort(H)
        plt.figure(figsize=(25, 25))

        fig, axd = plt.subplot_mosaic([
            ['scatter_1', 'scatter_2', 'scatter_3', 'scatter_4'],
            ['bar', 'bar', 'bar', 'bar'],
            ['scatter_5', 'scatter_6', 'scatter_7', 'scatter_8'],
        ],
            figsize=(11, 7),
            constrained_layout=True)

        dem_plots = np.array([
            0,
            2,
            len(H) - 3,
            len(H) - 1,
            1,
            int(len(H) / 2),
            int(len(H) / 2) + 1,
            len(H) - 2,
        ])
        colors = ('royalblue', 'goldenrod', 'red', 'purple', 'lime',
                  'turquoise', 'green', 'yellow')

        axd['bar'].bar(
            range(len(H)),
            H[idcs],
            color=[
                colors[np.where(
                    dem_plots == i)[0][0]] if i in dem_plots else 'grey'
                for i in range(len(self.sdata.stats.counts))
            ])

        axd['bar'].set_xticks(range(len(H)),
                              [self.sdata.genes[h] for h in idcs],
                              rotation=90)
        axd['bar'].set_ylabel('knn entropy, k=' + str(n_neighbors))

        for i in range(8):
            idx = idcs[dem_plots[i]]
            gene = self.sdata.genes[idx]
            plot_name = 'scatter_' + str(i + 1)
            axd[plot_name].set_title(gene)
            axd[plot_name].scatter(
                self.sdata.x, self.sdata.y, color=(0.5, 0.5, 0.5, 0.1))
            axd[plot_name].scatter(self.sdata.x[self.sdata['gene_id'] == idx],
                                   self.sdata.y[self.sdata['gene_id'] == idx],
                                   color=colors[i],
                                   marker='.')
            axd[plot_name].set_xticks([], [])
            # if i>0:
            axd[plot_name].set_yticks([], [])

            if i < 4:
                y_ = (H[idcs])[i]
                _y = 0
            else:
                y_ = 0
                _y = 1

            con = ConnectionPatch(xyA=(dem_plots[i], y_),
                                  coordsA=axd['bar'].transData,
                                  xyB=(np.mean(axd[plot_name].get_xlim()),
                                       axd[plot_name].get_ylim()[_y]),
                                  coordsB=axd[plot_name].transData,
                                  color='white',
                                  linewidth=1,
                                  linestyle='dotted')
            fig.add_artist(con)

    def _determine_counts(self, bandwidth=1, kernel=None):
        """_determine_counts: Determines the count distributions of genes around each molecule, thus effectively generating models of the immediate environment. 

        :param bandwidth: Bandwidth of the default Gaussian kernel, defaults to 1
        :type bandwidth: int, optional
        :param kernel: callable kernel function. If none is provided, a Gaussian function is used. Defaults to None
        :type kernel: callable, optional
        :return: 2d array of count distributions
        :rtype: np.Array
        """

        counts = np.zeros((len(self.sdata,), len(self.sdata.genes)))
        if kernel is None:
            def kernel(x): return np.exp(-x**2/(2*bandwidth**2))

        for i in range(0, self.n_neighbors):
            counts[np.arange(len(self.sdata)), self.neighbor_types[:, i]
                   ] += kernel(self.distances[:, i])
        return counts

    def run_umap(self, bandwidth=1, kernel=None, metric='euclidean', zero_weight=1.0, cutoff = None, *args, **kwargs):
        """run_umap: Creates a UMAP representation of recurring local contexts in the source data.

        :param bandwidth: Bandwidth of the default Gaussian kernel used to build local environment models, defaults to 1
        :type bandwidth: int, optional
        :param kernel: Callable function to generate a model of the local spatial context for each spot's KNN graph. If none is provided, the default Gaussian is used. Defaults to None
        :type kernel: callable, optional
        :param metric:  Kernel description used as a UMAP parameter, defaults to 'euclidean'
        :type metric: str, callable, optional
        :param zero_weight: Regularization parameter that adds information of each spot's gene label to its local context model. High value encourages spots to form clusters with spots of their own gene label, defaults to 1
        :type zero_weight: float, optional
        """
        # print(kwargs)
        counts = self._determine_counts(bandwidth=bandwidth, kernel=kernel)
        assert (all(counts.sum(1)) > 0)
        counts[np.arange(len(self.sdata)),
               self.sdata.gene_ids] += zero_weight-1

        if cutoff is not None:

            # print(f'PCA redution to {cutoff} dimensions.')
            # pca=PCA()
            # facs = pca.fit_transform(counts)

            # counts = np.zeros((counts.shape[0],cutoff+10))

            # counts[:,:cutoff] = facs[:,:cutoff]

            print(f'Reducing dimensions with FastICA')
            ica = FastICA(n_components=cutoff)
            counts = ica.fit_transform(counts)

            # counts[:,cutoff:] = facs
            # del facs,ica
            print('Calculating UMAP embedding.')
            # counts[:,-1]=facs[:,cutoff:].sum(1)
            

        umap = UMAP(metric=metric, *args, **kwargs)
        self._umap = umap.fit_transform(counts)

    def plot_umap(self, color_prop='genes', text_prop=None,
                  text_color_prop=None, c=None, color=None, color_dict=None, text_distance=1,
                  thlds_text=(1.0, 0.0, 0.0), text_kwargs={}, legend=False, **kwargs):

        embedding = self.umap

        categories = self.sdata.obsc[color_prop].unique()
        colors = self.sdata.obsc.project('c_'+color_prop)

        if color is not None:
            colors = (color,)*len(self.sdata)
        if c is not None:
            colors = c

        if legend:
            handlers = [plt.scatter([], [], color=self.sdata.obsc[self.sdata.obsc[color_prop]
                                    == c]['c_'+color_prop][0]) for c in sorted(categories)]
            plt.legend(handlers, sorted(categories))

        plt.scatter(*embedding.T, c=colors, **kwargs)

        if text_prop is not None:

            text_xs = []
            text_ys = []
            text_cs = []
            text_is = []
            text_zs = []

            X, Y = np.mgrid[embedding[:, 0].min():embedding[:, 0].max():100j,
                            embedding[:, 1].min():embedding[:, 1].max():100j]
            positions = np.vstack([X.ravel(), Y.ravel()])

            for i, g in enumerate(self.sdata.obsc[text_prop].unique()):
                if text_color_prop is not None:
                    fontcolor = self.sdata.obsc[self.sdata.obsc[text_prop]
                                                == g]['c_'+text_color_prop][0]
                else:
                    fontcolor = 'w'

                # try:
                embedding_subset = embedding[self.sdata.g.isin(
                    self.sdata.obsc[self.sdata.obsc[text_prop] == g].index)].T
                if embedding_subset.shape[1] > 2:
                    kernel = scipy.stats.gaussian_kde(embedding_subset)

                    Z = np.reshape(kernel(positions).T, X.shape)
                    localmaxs = scipy.ndimage.maximum_filter(Z, size=(10, 10))
                    maxz = (localmaxs == Z) & (localmaxs >=
                                               (localmaxs.max()*thlds_text[0]))
                    maxs = np.where(maxz)
                    # maxz = Z.max()
                    maxx = X[:, 0][maxs[0]]
                    maxy = Y[0][maxs[1]]

                    for j in range(len(maxx)):
                        # plt.suptitle(maxx[j])
                        text_xs.append(maxx[j])
                        text_ys.append(maxy[j])
                        text_is.append(g)
                        text_cs.append(fontcolor)
                        text_zs.append(Z[maxs[0][j], maxs[1][j]])

            cogs = self._untangle_text(
                np.array([text_xs, text_ys, ]).T, min_distance=text_distance)
            for i, c in enumerate(cogs):
                # plt.suptitle(text_zs)
                if (text_zs[i] > (max(text_zs)*thlds_text[1])):
                    if (thlds_text[2] is not None) or ((text_is[i] in self.sdata.counts) and (self.sdata.counts[text_is[i]] > thlds_text[2])):
                        txt = plt.text(
                            c[0], c[1], text_is[i], color=text_cs[i], ha='center', **text_kwargs)
                        txt.set_path_effects(
                            [PathEffects.withStroke(linewidth=3, foreground='k')])

                # except:
                #     print(f'Failed to assign text label to {g}')
    def umap_js(self, color_prop='c_genes'):
        """umap_js: A javascript-based display and selection function for the spatial source data and generated UMAP embedding. Can be used to understand UMAP agglomerations by displaying their gene compositions and projecting them back onto the original source data coordinates.

        :param color_prop: Property by which to color the spots in the scatter plots of the source data coordinates as well as the UMAP representation. Needs to be a column in sdata.obsc, defaults to 'c_genes'
        :type color_prop: str, optional
        :return: plotly HTML/javascript widget that displays in the jupyter notebook.
        :rtype: plotly.Widget
        """

        n_bars = 20

        if False:
            f_scatter = go.FigureWidget(px.imshow(np.repeat(self.sdata.background.data[:, :, None], 3, axis=-1,),
                                                x=np.linspace(
                                                    self.sdata.background.extent[0], self.sdata.background.extent[1], self.sdata.background.data.shape[0]),
                                                y=np.linspace(
                                                    self.sdata.background.extent[2], self.sdata.background.extent[3], self.sdata.background.data.shape[1])
                                                ),
                                        layout=Layout(border='solid 4px', width='100%'))

            trace_scatter = go.Scattergl(x=self.sdata.x,
                                        y=self.sdata.y,
                                        mode='markers',
                                        marker=dict(
                                            color=self.sdata.obsc.project(color_prop)),
                                        hoverinfo='none', meta={'name': 'tissue-scatter'},
                                        unselected={'marker': {
                                            'color': 'black', 'opacity': 0.2}},
                                        )

            f_scatter.add_trace(trace_scatter)
        else:
            trace_scatter = go.Scattergl(x=self.sdata.x,
                            y=self.sdata.y,
                            mode='markers',
                            marker=dict(
                                color=self.sdata.obsc.project(color_prop)),
                            hoverinfo='none', meta={'name': 'tissue-scatter'},
                            unselected={'marker': {
                                'color': 'black', 'opacity': 0.2}},
                            )

            f_scatter = go.FigureWidget(trace_scatter,)


        f_umap = go.FigureWidget(go.Scattergl(x=self.sdata.graph.umap[:, 0],
                                              y=self.sdata.graph.umap[:, 1],
                                              mode='markers',
                                              marker=dict(
                                                  color=self.sdata.obsc.project(color_prop)),
                                              unselected={'marker': {
                                                  'color': 'black', 'opacity': 0.2}},
                                              hoverinfo='none', meta={'name': 'umap-scatter'}),
                                 )

        colors = self.sdata.obsc.loc[self.sdata.stats.sort_values(
            'counts')[-n_bars:].index, color_prop].values

        w_bars = go.Bar(x=self.sdata.stats.sort_values('counts')[-n_bars:].index,
                        y=self.sdata.stats.sort_values(
                            'counts')[-n_bars:]['counts'],
                        marker={
                            'color': ['rgb'+str(tuple((np.array(c)*256).astype(int))) for c in colors]},
                        )
        f_bars = go.FigureWidget(w_bars)

        f_bars.data[0]['showlegend'] = False

        w_bars_ratio_up = go.Bar(x=self.sdata.stats.sort_values('counts')[-n_bars:].index,
                                 y=[0]*n_bars,
                                 marker={
                                     'color': ['rgb'+str(tuple((np.array(c)*256).astype(int))) for c in colors]},
                                 )
        f_bars_ratio_up = go.FigureWidget(w_bars_ratio_up)

        f_bars_ratio_up.data[0]['showlegend'] = False

        w_bars_binom = go.Bar(x=self.sdata.stats.sort_values('counts')[-n_bars:].index,
                              y=[0]*n_bars,
                              marker={
                                  'color': ['rgb'+str(tuple((np.array(c)*256).astype(int))) for c in colors]},
                              )
        f_bars_binom = go.FigureWidget(w_bars_binom)

        f_bars_binom.data[0]['showlegend'] = False

        out = widgets.Output(layout={'border': '1px solid black'})

        def update_bars(plot, points, selector):

            if plot['meta']['name'] == 'tissue-scatter':
                f_umap.data[0].selectedpoints = plot.selectedpoints

            else:
                f_scatter.data[-1].selectedpoints = plot.selectedpoints

            subset = self.sdata[np.array(points.point_inds)]

            colors = subset.obsc.loc[subset.stats.sort_values(
                'counts')[-n_bars:].index, color_prop].values
            colors = ['rgb'+str(tuple((np.array(c)*256).astype(int)))
                      for c in colors]
            ys = subset.stats.sort_values('counts')[-n_bars:]['counts']
            xs = subset.stats.sort_values('counts')[-n_bars:].index

            f_bars.data[0].marker.color = colors
            f_bars.data[0].x = xs
            f_bars.data[0].y = ys

            vals = (subset.stats.counts) / \
                (self.sdata.stats.loc[subset.stats.index].counts)
            idcs = np.argsort(vals)

            colors = subset.obsc[color_prop][idcs]
            colors = ['rgb'+str(tuple((np.array(c)*256).astype(int)))
                      for c in colors]
            ys = vals[idcs]
            xs = subset.stats.index[idcs]

            f_bars_ratio_up.data[0].marker.color = colors[-n_bars:]
            f_bars_ratio_up.data[0].x = xs[-n_bars:]
            f_bars_ratio_up.data[0].y = (ys[-n_bars:])

            vals = binom.cdf(subset.stats.counts, self.sdata.stats.loc[subset.stats.index].counts, len(
                subset)/len(self.sdata))
            idcs = np.argsort(vals)

            colors = subset.obsc[color_prop][idcs]
            colors = ['rgb'+str(tuple((np.array(c)*256).astype(int)))
                      for c in colors]
            ys = vals[idcs]
            xs = subset.stats.index[idcs]

            f_bars_binom.data[0].marker.color = colors[-n_bars:]
            f_bars_binom.data[0].x = xs[-n_bars:]
            f_bars_binom.data[0].y = (ys[-n_bars:])

        f_scatter.data[-1].on_selection(update_bars)
        f_umap.data[0].on_selection(update_bars)

        text_field = widgets.Text(
            value='selection1',
            placeholder='Label for storing',
            description='Name:',
            disabled=False
        )

        store_button = widgets.Button(
            description='store selection',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        )

        def store_selection(event):
            self.sdata[text_field.value] = (pd.Series(
                np.arange(len(self.sdata))).isin(f_umap.data[0].selectedpoints)).values

        store_button.on_click(store_selection)

        reset_button = widgets.Button(
            description='reset selection',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        )

        return widgets.VBox([widgets.HBox([f_scatter, f_umap], layout=Layout(display='flex', width='100%', height='80%', border='red solid 1px', align_items='stretch', justify_content='space-around', flex_direction='row')),
                            widgets.HBox([widgets.HBox([f_bars, f_bars_ratio_up, f_bars_binom], layout=Layout(display='flex', width='80%')),
                                          widgets.VBox([widgets.HBox([text_field,
                                                                      ]), widgets.HBox([store_button,reset_button])], layout=Layout(display='flex', width='20%', height='100%', border='red solid 1px', justify_content='space-around', flex_direction='column')
                                                       )]
                                         )], layout=Layout(width='100%', height='80vh', background='red', border='solid 1px'))

    def map_and_umap(self, color_prop=None, scalebar=True, cmap='jet',
                         **kwargs):
        """map_and_umap: Plots a side-by-side representation of the available UMAP- and coordinate data, with styling arguments passed to both plotting functions. 

        :param color_prop: Property to color the individual markers by. Needs to be a column in self.sdata.obsc, defaults to None
        :type color_prop: _type_, optional
        :param scalebar: Whether to display a scalebar in the coordinate representation, defaults to True
        :type scalebar: bool, optional
        """

        if color_prop is None:
            color_prop = 'genes'
        else:
            kwargs['color_prop'] = color_prop
            # scatter_kwargs['color_prop'] = color_prop

        plt.style.use('dark_background')

        fig = plt.gcf()

        ax1 = plt.subplot2grid((3, 2), (0, 0), 2, 1)

        sc2, _, _ = self.sdata.scatter(axd=ax1,scalebar=scalebar,cmap=cmap,  ** kwargs)

        ax2 = plt.subplot2grid((3, 2), (0, 1), 2, 1)
        self.sdata.graph.plot_umap(cmap=cmap,**kwargs)



    def _untangle_text(self, cogs, untangle_rounds=50, min_distance=0.5):
        knn = NearestNeighbors(n_neighbors=2)

        np.random.seed(42)
        cogs_new = cogs+np.random.normal(size=cogs.shape,)*0.01

        for i in range(untangle_rounds):

            cogs = cogs_new.copy()
            knn = NearestNeighbors(n_neighbors=2)

            knn.fit(cogs)
            distances, neighbors = knn.kneighbors(cogs/[1.01, 1.0])
            too_close = (distances[:, 1] < min_distance)

            for i, c in enumerate(np.where(too_close)[0]):
                partner = neighbors[c, 1]
                cog = cogs[c]-cogs[partner]
                cog_new = cogs[c]+0.3*cog
                cogs_new[c] = cog_new

        return cogs_new
