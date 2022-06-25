
from turtle import color
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import Button, TextBox
import matplotlib.patheffects as PathEffects

import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets, interactive, HBox, VBox,Output,Layout

import pandas as pd
from scipy.stats import binom

from sklearn.neighbors import NearestNeighbors

from umap import UMAP

class SpatialGraph():

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
        if self._neighbors is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._neighbors[:,:self.n_neighbors]

    @property
    def distances(self):
        if self._distances is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._distances[:,:self.n_neighbors]

    @property
    def neighbor_types(self):
        if self._neighbor_types is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._neighbor_types[:,:self.n_neighbors]

    @property
    def umap(self):
        if self._umap is None:
            self.run_umap()
        return self._umap

    @property
    def tsne(self):
        if self._tsne is None:
            self.run_tsne()
        return self._tsne

    def __getitem__(self,*args):
        sg = SpatialGraph(self.sdata,self.n_neighbors)
        if self._distances is not None:
            sg._distances = self._distances.__getitem__(*args)
        if self._neighbors is not None:
            sg._neighbors = self._neighbors.__getitem__(*args)
        if self._neighbor_types is not None:
            sg._neighbor_types = self._neighbor_types.__getitem__(*args)

    def update_knn(self, n_neighbors, re_run=False):

        if self._neighbors is not None and (n_neighbors <
                                            self._neighbors.shape[1]):
            self.n_neighbors=n_neighbors
            # return (self._neighbors[:, :n_neighbors],
            #         self._distances[:, :n_neighbors],
            #         self._neighbor_types[:, :n_neighbors])
        else:

            coordinates = np.stack([self.sdata.x, self.sdata.y]).T
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            knn.fit(coordinates)
            self._distances, self._neighbors = knn.kneighbors(coordinates)
            self._neighbor_types = np.array(self.sdata.gene_ids)[self._neighbors]

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
            axd[plot_name].scatter(self.sdata.x, self.sdata.y, color=(0.5, 0.5, 0.5, 0.1))
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

    def _determine_counts(self,bandwidth=1, kernel=None):

        counts = np.zeros((len(self.sdata,),len(self.sdata.genes)))
        if kernel is None:
            kernel = lambda x: np.exp(-x**2/(2*bandwidth**2))

        for i in range(0,self.n_neighbors):
            counts[np.arange(len(self.sdata)),self.neighbor_types[:,i]]+=  kernel(self.distances[:,i])
        return counts

    def run_umap(self,bandwidth=1,kernel=None,metric='cosine', zero_weight=1,*args,**kwargs):        
        # print(kwargs)
        counts = self._determine_counts(bandwidth=bandwidth,kernel=kernel)
        assert (all(counts.sum(1))>0)
        counts[np.arange(len(self.sdata)),self.sdata.gene_ids]+=zero_weight-1
        umap=UMAP(metric=metric,*args,**kwargs)
        self._umap = umap.fit_transform(counts)

    # def run_tsne(self,bandwidth=1,kernel=None,*args,**kwargs):        
    #     counts = self._determine_counts(bandwidth=bandwidth,kernel=kernel)
    #     tsne=TSNE(*args,**kwargs)
    #     self._tsne = tsne.fit_transform(counts)

    # def plot_umap(self, text_prop=None, color_prop='genes', color_dict=None, c=None,text_distance=1, thlds_text=(1.0,0.0,None),text_kwargs={}, **kwargs):
    #     self.plot_embedding(self.umap, text_prop=text_prop, color_prop=color_prop, color_dict=color_dict, 
    #     c=c,text_distance=text_distance, thlds_text=thlds_text, text_kwargs=text_kwargs, **kwargs)

    # def plot_tsne(self,text_prop=None, color_prop='genes', color_dict=None, c=None,text_distance=1, **kwargs):
    #     self.plot_embedding(self.tsne, text_prop=text_prop, color_prop=color_prop, color_dict=color_dict, c=c,text_distance=text_distance, **kwargs)

    def plot_umap(self, color_prop='genes', text_prop=None, 
                    text_color_prop = None, c=None, color=None, color_dict=None,text_distance=1, 
                    thlds_text=(1.0,0.0,0.0), text_kwargs={}, legend=False, **kwargs):

        embedding=self.umap

        categories = self.sdata.obsc[color_prop].unique() 
        colors = self.sdata.obsc.project('c_'+color_prop)
        

        if color is not None:
            colors=(color,)*len(self.sdata)
        if c is not None:
            colors=c

        if legend:
            handlers = [plt.scatter([],[],color=self.sdata.obsc[self.sdata.obsc[color_prop]==c]['c_'+color_prop][0]) for c in sorted(categories)]
            plt.legend(handlers, sorted(categories))

        plt.scatter(*embedding.T,c=colors, **kwargs)

        if text_prop is not None:

            text_xs=[]
            text_ys=[]
            text_cs=[]
            text_is=[]
            text_zs=[]

            X, Y = np.mgrid[embedding[:,0].min():embedding[:,0].max():100j, 
                embedding[:,1].min():embedding[:,1].max():100j]
            positions = np.vstack([X.ravel(), Y.ravel()])

           
            for i,g in enumerate(self.sdata.obsc[text_prop].unique()):
                if text_color_prop is not None:
                    fontcolor = self.sdata.obsc[self.sdata.obsc[text_prop]==g]['c_'+text_color_prop][0]
                else: fontcolor='w'

                # try:
                embedding_subset = embedding[self.sdata.g.isin(self.sdata.obsc[self.sdata.obsc[text_prop]==g].index)].T
                if embedding_subset.shape[1]>2:
                    kernel = scipy.stats.gaussian_kde(embedding_subset)

                    Z = np.reshape(kernel(positions).T, X.shape)
                    localmaxs = scipy.ndimage.maximum_filter(Z,size=(10,10))
                    maxz = (localmaxs==Z)&(localmaxs>=(localmaxs.max()*thlds_text[0]))
                    maxs = np.where(maxz)
                    # maxz = Z.max()
                    maxx = X[:,0][maxs[0]]
                    maxy = Y[0][maxs[1]]
                
                    for j in range(len(maxx)):
                        # plt.suptitle(maxx[j])
                        text_xs.append(maxx[j])
                        text_ys.append(maxy[j])
                        text_is.append(g)
                        text_cs.append(fontcolor)
                        text_zs.append(Z[maxs[0][j],maxs[1][j]])

            cogs = self._untangle_text(np.array([text_xs,text_ys,]).T, min_distance=text_distance)
            for i,c in enumerate(cogs):
                # plt.suptitle(text_zs)
                if (text_zs[i]>(max(text_zs)*thlds_text[1])):
                    if (thlds_text[2] is not None) or ((text_is[i] in self.sdata.counts ) and (self.sdata.counts[text_is[i]]>thlds_text[2])):
                        txt = plt.text(c[0],c[1],text_is[i],color=text_cs[i],ha='center',**text_kwargs)
                        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])

                # except:
                #     print(f'Failed to assign text label to {g}')
    def umap_js(self, color_prop='c_genes'):
                
        n_bars=20

        f_scatter = go.FigureWidget(px.imshow(np.repeat(self.sdata.pixel_maps[0].data[:,:,None],3,axis=-1,),
                                x=np.linspace(self.sdata.background.extent[0],self.sdata.background.extent[1],self.sdata.background.data.shape[0]),
                                y=np.linspace(self.sdata.background.extent[2],self.sdata.background.extent[3],self.sdata.background.data.shape[1])
                                ),
                        layout=Layout(border='solid 4px',width='100%'))

        trace_scatter=go.Scattergl(x=self.sdata.x,
                                y=self.sdata.y,
                                mode='markers', 
                                marker=dict(color=self.sdata.obsc.project(color_prop)),
                                hoverinfo='none',meta={'name':'tissue-scatter'},
                                unselected={'marker':{'color':'black','opacity':0.2}},
                                )

        f_scatter.add_trace(trace_scatter)

        f_umap = go.FigureWidget(go.Scattergl(x=self.sdata.graph.umap[:,0],
                                            y=self.sdata.graph.umap[:,1],
                                            mode='markers', 
                                            marker=dict(color=self.sdata.obsc.project(color_prop)),
                                            unselected={'marker':{'color':'black','opacity':0.2}},
                                            hoverinfo='none',meta={'name':'umap-scatter'}),
                                )

        colors=self.sdata.obsc.loc[self.sdata.stats.sort_values('counts')[-n_bars:].index,color_prop].values

        w_bars=go.Bar(                  x=self.sdata.stats.sort_values('counts')[-n_bars:].index,
                                        y=self.sdata.stats.sort_values('counts')[-n_bars:]['counts'],
                                        marker={'color':['rgb'+str(tuple((np.array(c)*256).astype(int))) for c in  colors]},
                                    )
        f_bars = go.FigureWidget(w_bars)

        f_bars.data[0]['showlegend']=False



        w_bars_ratio_up=go.Bar(         x=self.sdata.stats.sort_values('counts')[-n_bars:].index,
                                        y=[0]*n_bars,
                                        marker={'color':['rgb'+str(tuple((np.array(c)*256).astype(int))) for c in  colors]},
                                    )
        f_bars_ratio_up = go.FigureWidget(w_bars_ratio_up)

        f_bars_ratio_up.data[0]['showlegend']=False



        w_bars_binom=go.Bar(         x=self.sdata.stats.sort_values('counts')[-n_bars:].index,
                                        y=[0]*n_bars,
                                        marker={'color':['rgb'+str(tuple((np.array(c)*256).astype(int))) for c in  colors]},
                                    )
        f_bars_binom = go.FigureWidget(w_bars_binom)

        f_bars_binom.data[0]['showlegend']=False

        out = widgets.Output(layout={'border': '1px solid black'})

        def update_bars(plot,points,selector):
            
            if plot['meta']['name']=='tissue-scatter':
                f_umap.data[0].selectedpoints=plot.selectedpoints
                
            else:
                f_scatter.data[1].selectedpoints=plot.selectedpoints
                
            subset=self.sdata[np.array(points.point_inds)]
            
            colors=subset.obsc.loc[subset.stats.sort_values('counts')[-n_bars:].index,color_prop].values
            colors=['rgb'+str(tuple((np.array(c)*256).astype(int))) for c in  colors]
            ys = subset.stats.sort_values('counts')[-n_bars:]['counts']
            xs = subset.stats.sort_values('counts')[-n_bars:].index
            
            f_bars.data[0].marker.color=colors
            f_bars.data[0].x=xs
            f_bars.data[0].y=ys
            
            vals= (subset.stats.counts)/ (self.sdata.stats.loc[subset.stats.index].counts)
            idcs = np.argsort(vals)
            
            colors=subset.obsc[color_prop][idcs]
            colors=['rgb'+str(tuple((np.array(c)*256).astype(int))) for c in  colors]
            ys = vals[idcs]
            xs = subset.stats.index[idcs]
            
            f_bars_ratio_up.data[0].marker.color=colors[-n_bars:]
            f_bars_ratio_up.data[0].x=xs[-n_bars:]
            f_bars_ratio_up.data[0].y=(ys[-n_bars:])
            
            vals= binom.cdf(subset.stats.counts,self.sdata.stats.loc[subset.stats.index].counts,len(subset)/len(self.sdata))
            idcs = np.argsort(vals)
            
            colors=subset.obsc[color_prop][idcs]
            colors=['rgb'+str(tuple((np.array(c)*256).astype(int))) for c in  colors]
            ys = vals[idcs]
            xs = subset.stats.index[idcs]
            
            f_bars_binom.data[0].marker.color=colors[-n_bars:]
            f_bars_binom.data[0].x=xs[-n_bars:]
            f_bars_binom.data[0].y=(ys[-n_bars:])
            
            
            
        f_scatter.data[1].on_selection(update_bars)
        f_umap.data[0].on_selection(update_bars)

        text_field=widgets.Text(
            value='selection1',
            placeholder='Label for storing',
            description='Name:',
            disabled=False
        )

        store_button = widgets.Button(
            description='store selection',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
        )
            
        def store_selection(event):
            self.sdata[text_field.value]=pd.Series(np.arange(len(self.sdata))).isin(f_umap.data[0].selectedpoints)
            []
        store_button.on_click(store_selection)
            
        reset_button = widgets.Button(
            description='reset selection',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
        )


        return widgets.VBox([widgets.HBox([f_scatter,f_umap] ,layout=Layout(display='flex',width='100%',height='80%',border='red solid 1px',align_items='stretch',justify_content='space-around',flex_direction='row')),
                            widgets.HBox([widgets.HBox([f_bars,f_bars_ratio_up,f_bars_binom],layout=Layout(display='flex',width='80%')),
                                        widgets.VBox([widgets.HBox([text_field,
                                                                    store_button
                                                                    ]),widgets.HBox([reset_button])],layout=Layout(display='flex',width='20%',height='100%',border='red solid 1px',justify_content='space-around',flex_direction='column')
                                                    )]
                                        )]
                        
                        ,layout=Layout(width='100%',height='80vh',background='red',border='solid 1px'))


    def umap_interactive(self, color_prop=None, umap_kwargs={'alpha':0.1, 'marker':'x','text_kwargs':{'fontsize':10},'legend':False},
                                scatter_kwargs={'marker':'x','alpha':0.5,}):


        if color_prop is None:
            color_prop='genes'
        else:
            umap_kwargs['color_prop']=color_prop
            scatter_kwargs['color_prop']=color_prop


        click_coords=np.array((0.0,0.0))

        plt.style.use('dark_background')

        fig = plt.gcf()

        dist=((self.sdata.graph.umap)**2).sum(1)**0.5
        radius=int(dist.max())+1

        ax1 = plt.subplot2grid((3, 2), (0, 0),2,1)
        
        sc2,_,_ = self.sdata.scatter(axd=ax1,s=np.zeros((len(self.sdata),))+10,**(scatter_kwargs|{'animated':False}))

        ax2 = plt.subplot2grid((3, 2), (0, 1),2,1)  
        self.sdata.graph.plot_umap(**umap_kwargs)

        circle=plt.Circle((0,0),radius, color=plt.rcParams['axes.edgecolor'],fill=False, linestyle='--',animated=False)
        ax2.add_artist(circle)

        ax3 = plt.subplot2grid((3, 2), (2, 0),1,1)

        bars=ax3.bar(range(10),np.zeros((10,)),animated=False)
        plt.xticks(range(10),['-']*10,rotation=90)
        ax3.set_ylim(0,1)
        ax3.set_ylabel('molecule counts')

        # ax4 = plt.subplot2grid((3, 2), (2, 1),1,1)
        ax4=plt.axes([0.6,0.28,0.2,0.05])
        ax5=plt.axes([0.6,0.2,0.2,0.05])
        ax6=plt.axes([0.8,0.2,0.1,0.05])
        ax7=plt.axes([0.6,0.12,0.2,0.05])


        text_box_radius = TextBox(ax4, 'radius', initial=1,color='lightgray',hovercolor='darkgray')
        text_box_radius.text_disp.set_color('k')

        text_box_label = TextBox(ax5, 'label', initial='mask_1',color='lightgray',hovercolor='darkgray')
        text_box_label.text_disp.set_color('k')

        txt = ax7.text(0.0,0.5,'',color='r',va='center')
        ax7.set_axis_off()

        def store_selection(event):
            # print('kewl',dist)
            txt.set_text(text_box_label.text)

            if (~(text_box_label.text in self.sdata.columns)) or ('Click again to overwrite.' in  txt.text):
                
                txt.set_text((dist<int(text_box_radius.text)).sum())
                self.sdata[text_box_label.text]=dist<int(text_box_radius.text)
                txt.set_text(f'Stored {(dist<int(text_box_radius.text)).sum()} points as {text_box_label.text}.')
            
            elif (text_box_label.text in self.sdata.columns):
                # txt.set_text('kewlest')
                txt.set_text(f'Label {text_box_label.text} already exists. Click again to overwrite.')
                

            # fig.canvas.draw()

        btn_submit=Button(ax6,'store selection',color='rosybrown',hovercolor='lightcoral')
        btn_submit.on_clicked(store_selection)

        bg = fig.canvas.copy_from_bbox(fig.bbox)

        fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.1)

        ax2.draw_artist(circle)
        ax2.figure.canvas.draw_idle()
        # fig.canvas.blit(fig.bbox)
        # fig.canvas.flush_events()

        def on_click(event):
            click_coords[0]=event.xdata
            click_coords[1]=event.ydata

        def update_bars(subset):

            # ax2.set_title('T')
            subset=self.sdata[subset]
            heights = subset.stats.counts.sort_values()[-10:]
            ax3.set_ylim(0,heights.max())

            [b.set_height(heights.iloc[i]) for i,b in enumerate(bars)]
            ax3.set_xticklabels(heights.index)      
            [b.set_color(subset.obsc['c_'+color_prop][heights.index[i]]) for i,b in enumerate(bars)]        
            
        def on_release(event):
            center = np.array((event.xdata,event.ydata))
            if all(click_coords==center) and (event.inaxes is ax2):
                # ax2.set_title(f'{event.xdata},{event.ydata}')
                circle.set_center(center)
                
                dist[:] = ((self.sdata.graph.umap-center)**2).sum(1)**0.5
                
                radius=int(text_box_radius.text)
                text_box_label.text
                btn_submit
                circle.radius=radius
                
                sc2._sizes[dist<radius]=20
                sc2._sizes[dist>radius]=0

                update_bars(dist<radius)
                        
                fig.canvas.blit(fig.bbox)
                fig.canvas.flush_events()
                # fig.canvas.draw()
                
       
        update_bars(dist<radius)

        plt.connect('button_press_event', on_click)
        plt.connect('button_release_event', on_release)
            


 
    def _untangle_text(self, cogs, untangle_rounds=50, min_distance=0.5):
        knn = NearestNeighbors(n_neighbors=2)

        np.random.seed(42)
        cogs_new = cogs+np.random.normal(size=cogs.shape,)*0.01

        for i in range(untangle_rounds):

            cogs = cogs_new.copy()
            knn = NearestNeighbors(n_neighbors=2)

            knn.fit(cogs)
            distances,neighbors = knn.kneighbors(cogs/[1.01,1.0])
            too_close = (distances[:,1]<min_distance)

            for i,c in enumerate(np.where(too_close)[0]):
                partner = neighbors[c,1]
                cog = cogs[c]-cogs[partner]
                cog_new = cogs[c]+0.3*cog
                cogs_new[c]= cog_new
                
        return cogs_new

