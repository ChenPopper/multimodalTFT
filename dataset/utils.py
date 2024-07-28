import copy

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import metpy.calc
import numpy as np
import xarray as xr
import os
import json
import shutil
from datetime import datetime, timedelta
import numpy as np


def calculate_vorticity(data, itimetamp):
    # 加载数据

    # 使用metpy计算涡度
    u850 = data['u_component_of_wind'].sel(level=850, time=itimetamp).values
    v850 = data['v_component_of_wind'].sel(level=850, time=itimetamp).values

    lat_arr = data.latitude.values
    lon_arr = data.longitude.values

    ds2 = xr.Dataset(
        {
            "u850": (["lat", "lon"], u850, {'units': 'm/s'}),
            "v850": (["lat", "lon"], v850, {'units': 'm/s'}),
        },
        coords={
            "lat": lat_arr,
            "lon": lon_arr,
        }
    )

    vor = metpy.calc.vorticity(ds2.u850, ds2.v850)

    vor_data = vor.values
    vor_data = np.expand_dims(vor_data, axis=0)
    t99 = np.percentile(vor_data, 99)
    t01 = np.percentile(vor_data, 1)
    vor_data[vor_data > t99] = t99
    vor_data[vor_data < t01] = t01
    vor_data = vor_data.astype(np.float32) * 100
    return vor_data


lat_ls_ = np.linspace(90, -90, 721)
lon_ls_ = np.linspace(0, 359.75, 1440)

font_size = 14


def __mkdir__(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except:
            pass


def calculate_vorticity(data, itimetamp):
    # 加载数据

    # 使用metpy计算涡度
    u850 = data['u_component_of_wind'].sel(level=850, time=itimetamp).values
    v850 = data['v_component_of_wind'].sel(level=850, time=itimetamp).values

    lat_arr = data.latitude.values
    lon_arr = data.longitude.values

    ds2 = xr.Dataset(
        {
            "u850": (["lat", "lon"], u850, {'units': 'm/s'}),
            "v850": (["lat", "lon"], v850, {'units': 'm/s'}),
        },
        coords={
            "lat": lat_arr,
            "lon": lon_arr,
        }
    )

    vor = metpy.calc.vorticity(ds2.u850, ds2.v850)

    vor_data = vor.values
    vor_data = np.expand_dims(vor_data, axis=0)
    t99 = np.percentile(vor_data, 99)
    t01 = np.percentile(vor_data, 1)
    vor_data[vor_data > t99] = t99
    vor_data[vor_data < t01] = t01
    vor_data = vor_data.astype(np.float32) * 100

    return vor_data

def nearst_index(indx_arr, val):
    # 计算数组中每个元素与给定值的差的绝对值
    diff = np.abs(indx_arr - val)

    # 使用numpy.argmin()找到最小差值的索引
    min_index = np.unravel_index(np.argmin(diff), indx_arr.shape)
    return min_index[0]


def extend_get(icenter):
    lat_ls = []
    lon_ls = []
    for icen in icenter:
        lat_ls.append(icen[0])
        lon_ls.append(icen[1])
    max_lat = max(lat_ls)
    min_lat = min(lat_ls)
    max_lon = max(lon_ls)
    min_lon = min(lon_ls)
    return max_lat, min_lat, max_lon, min_lon


def times_step_gene(beg_time: str, end_time: str, step: int, test=False):
    beg_time = datetime.strptime(beg_time, '%Y%m%d%H')
    end_time = datetime.strptime(end_time, '%Y%m%d%H')
    time_interval = timedelta(hours=step)
    time_ls = []
    current_date = beg_time
    while current_date <= end_time:
        time_ls.append(current_date)
        current_date += time_interval
        # time_ls.append(current_date)
    return time_ls


def times_select(beg_time: str, end_time: str, label: dict):
    beg_time = datetime.strptime(beg_time, '%Y%m%d%H')
    end_time = datetime.strptime(end_time, '%Y%m%d%H')
    time_interval = timedelta(hours=6)
    time_dict = {}
    current_date = beg_time
    time_ls = []
    while current_date <= end_time:
        curr_str = current_date.strftime('%Y%m%d%H')
        time_dict[curr_str] = 0
        current_date += time_interval
        # time_ls.append(current_date)
    label_select = {}
    for ikey in label.keys():
        if ikey in time_dict:
            label_select[ikey] = label[ikey]
            time_ls.append(datetime.strptime(ikey, '%Y%m%d%H'))
    return label_select, time_ls


def chunk_time(ds):
    '''
    chunk 可以控制数据春方式，将数据分块进行延迟加载和并行计算
    :param ds:
    :return:
    '''
    dims = {k: v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds


def inv_normalize(ds, mean, std, dim="level"):
    ds = ds.astype(np.float32)
    ds = ds * std + mean
    new_ds = chunk_time(ds)
    return new_ds


def deal_mean_std(file_name, vars):
    ds = xr.open_dataset(file_name)
    stat = None
    for ivar in vars:
        ids = ds.sel(channel=ivar).drop('channel').data.to_dataset()
        ids = ids.rename({'data': ivar})
        if stat is None:
            stat = ids
        else:
            stat = xr.merge([stat, ids])
    return stat


def merge_data_vars(zarr_ds, data_vars):
    """
    Merges multiple data variables into a single xarray.Dataset.

    Parameters:
    - zarr_ds (xarray.Dataset): The input dataset stored in zarr format.
    - data_vars (list of str): List of variable names to be merged.

    Returns:
    - xarray.Dataset: A dataset containing merged data variables.
    """
    ds = None
    for ivar in data_vars:
        # Select the variable and drop the 'channel' dimension
        ids = zarr_ds.sel(channel=ivar).drop('channel').data.to_dataset()
        # Rename the 'data' dimension to the name of the variable
        ids = ids.rename({'data': ivar})
        # Initialize or merge the datasets
        if ds is None:
            ds = ids
        else:
            ds = xr.merge([ds, ids])
    return ds


class ZarrRead(object):
    def __init__(self, zarr_file, data_vars, time_stamps, mean_file, std_file):
        # self.data_path = data_path
        self.file_name = zarr_file
        self.data_vars = data_vars
        self.time_stamps = time_stamps
        self.mean, self.std = self.load_statistic(mean_file, std_file)
        self.data = self.get_data(zarr_file)

    def get_data(self, zarr_file):
        mean_file = 'mean.nc'
        std_file = 'std.nc'
        mean = deal_mean_std(mean_file, self.data_vars)  # [self.data_vars]
        std = deal_mean_std(std_file, self.data_vars)

        zarr_ds = xr.open_zarr(zarr_file, consolidated=True)
        # zarr_ds = zarr_ds[self.data_vars].sel(time=self.time_stamps)

        ds = merge_data_vars(zarr_ds, self.data_vars)

        zarr_ds = inv_normalize(ds, mean, std)  #
        return zarr_ds

    def load_statistic(self, mean_file, std_file):
        mean = deal_mean_std(mean_file, self.data_vars)
        std = deal_mean_std(std_file, self.data_vars)
        return mean, std


def load_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
