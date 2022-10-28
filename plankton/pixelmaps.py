from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from typing import Union
import collections

import numpy as np
import matplotlib.pyplot as plt


class PixelMap():

    def __init__(self,
                 pixel_data: np.ndarray,
                 px_p_um: float = 1.0,
                 x_shift=0.0,
                 y_shift=0.0,
                 cmap='Greys') -> None:

        self.data = pixel_data

        self.n_channels = 1 if len(
            pixel_data.shape) == 2 else pixel_data.shape[-1]

        if not isinstance(px_p_um, collections.abc.Iterable) or len(px_p_um) == 1:
            self.scale = (px_p_um, px_p_um)
        else:
            self.scale = px_p_um

        self.cmap=cmap

        self.extent = (x_shift, x_shift+pixel_data.shape[0] / self.scale[0],
                       y_shift, y_shift + pixel_data.shape[1] / self.scale[1])

    @property
    def shape(self):
        return self.extent[1] - self.extent[0], self.extent[3] - self.extent[2]

    def imshow(self, cmap=None, axd=None, **kwargs) -> None:
        extent = np.array(self.extent)

        if (len(self.data.shape) > 2) and (self.data.shape[2] > 4):
            data = self.data.sum(-1)
        else:
            data = self.data

        if axd is None:
            axd = plt.gca()

        if cmap is None:
            cmap = self.cmap

        axd.imshow(data, extent=extent[[0, 3, 1, 2]], cmap=cmap,**kwargs)

    def __getitem__(self, indices: Union[slice, collections.abc.Iterable[slice]]):

        # print(self.extent)
        if not isinstance(indices, collections.abc.Iterable):
            index_x = indices
            index_y = slice(0, None, None)

        else:
            index_x = indices[0]

            if len(indices) > 1:
                index_y = indices[1]
            else:
                index_y = slice(0, None, None)

        if (index_x.start is None):
            start_x = 0  # self.extent[0]
        else:
            start_x = index_x.start
        if (index_x.stop is None):
            stop_x = self.extent[3]-1
        else:
            stop_x = index_x.stop

        if (index_y.start is None):
            start_y = 0  # self.extent[2]
        else:
            start_y = index_y.start
        if (index_y.stop is None):
            stop_y = self.extent[1]-1
        else:
            stop_y = index_y.stop

        # print(self.data.shape, int(stop_x * self.scale[1]), int(stop_y * self.scale[0]), self.scale)
        data = self.data[int(start_y * self.scale[1]):int(stop_y *
                                                          self.scale[1]),
                         int(start_x * self.scale[0]):int(stop_x *
                                                          self.scale[0]), ]

        return PixelMap(
            data,
            px_p_um=self.scale,
        )

    def get_value(self, x, y, padding_value=-1):
        x_= np.array(y).flatten()
        y= np.array(x).flatten()
        x=x_
        x = np.round((x-self.extent[0])/self.extent[1]*self.data.shape[0]).astype(int)
        y = np.round((y-self.extent[2])/self.extent[3]*self.data.shape[1]).astype(int)


        values = np.empty_like(x)
        valid = (x>=0)&(y>=0)&(x<self.data.shape[0])&(y<self.data.shape[1])
        print(x    )
        values[~valid] = padding_value
        values[valid] = self.data[x[valid],y[valid]]
        return values





class KDEProjection(PixelMap):
    def __init__(self, sd,
                 bandwidth: float = 3.0,
                 threshold_vf_norm: float = 1.0,
                 threshold_p_corr: float = 0.5,
                 px_p_um: float = 1) -> None:

        self.sdata = sd
        self.bandwidth = bandwidth
        self.threshold_vf_norm = threshold_vf_norm
        self.threshold_p_corr = threshold_p_corr

        self.scale = px_p_um

        super().__init__(self.run_kde(), px_p_um)

    def run_kde(self) -> None:

        kernel = self.generate_kernel(self.bandwidth*3, self.scale)

        x_int = np.array(self.sdata.y * self.scale).astype(int)
        y_int = np.array(self.sdata.x * self.scale).astype(int)
        genes = self.sdata.gene_ids

        vf = np.zeros(
            (x_int.max()+kernel.shape[0]+1, y_int.max()+kernel.shape[0]+1, len(self.sdata.genes)))

        for x, y, g in zip(x_int, y_int, genes):
            # print(x,y,vf.shape,kernel.shape)
            vf[x:x+kernel.shape[0], y:y+kernel.shape[1], g] += kernel

        return vf[kernel.shape[0]//2:-kernel.shape[0]//2, kernel.shape[1]//2:-kernel.shape[1]//2]

    def generate_kernel(self, bandwidth: float, scale: float = 1) -> np.ndarray:

        kernel_width_in_pixels = int(bandwidth * scale *
                                     6)  # kernel is 3 sigmas wide.

        span = np.linspace(-3, 3, kernel_width_in_pixels)
        X, Y = np.meshgrid(span, span)

        return 1 / (2 * np.pi)**0.5 * np.exp(-0.5 * ((X**2 + Y**2)**0.5)**2)


class DensityMap(PixelMap):
    def __init__(self, data, *args, **kwargs):

        # .super().qd

        pass


class PixelMask(PixelMap):
    def __init__(self,
                 pixel_data: np.ndarray,
                 px_p_um: float = 1.0,
                 x_shift=0.0,
                 y_shift=0.0) -> None:

        super(PixelMask, self).__init__(
            pixel_data=pixel_data,
            px_p_um=px_p_um,
            x_shift=x_shift,
            y_shift=y_shift
        )

# decorate PixelMap to overload python operators:

def overload_operator(name):

    def apply(self,*args): 
        # getattr(self.data   , name)(*args)
        pixel_map_ = self[:]
        pixel_map_.data = getattr(self.data, name)(*args)
        return pixel_map_

    return apply
    # return lambda self, *args: PixelMap(getattr(self.data, name)(*args), self.info)

for name in ["__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__invert__", "__neg__", "__pos__",
            "__lt__","__le__","__eq__","__ne__","__ge__","__gt__",]:
    setattr(PixelMap, name, overload_operator(name))

