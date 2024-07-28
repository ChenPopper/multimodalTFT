import os
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime, timedelta


class MockDataset(Dataset):

    def __init__(self, img_data, ts_data, hist_len, pred_len):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.sample_len = self.hist_len + self.pred_len
        self.data = self.get_data(ts_data, img_data)

    def __getitem__(self, index):
        return self.data[index]

    def get_data(self, ts_data, img_data):
        lifetime = len(ts_data)
        # assert lifetime > self.hist_len + self.pred_len, "time series is too short"
        context = []
        target = []
        labels = []
        for i in range(lifetime - self.hist_len - self.pred_len + 1):

            context.append([ts_data[i: i + self.hist_len], img_data[i: i + self.hist_len]])
            target.append(ts_data[i + self.hist_len: i + self.hist_len + self.pred_len])
        labels.extend(list(zip(context, target)))  # labels.append([context, target])

        return labels

    def __len__(self):
        return len(self.data)


def check_missing_data(timestamps, periods, fmt='%Y%m%d%H'):
    """
    to check whether the data is missing for some time periods
    :param timestamps:
    :param periods: Dict, time step, e.g., 6 hours for era5 records, {'hours': 6}
    :return: True or False
    """
    start_time = datetime.strptime(timestamps[0], fmt)
    end_time = datetime.strptime(timestamps[-1], fmt)
    return end_time == start_time + timedelta(**periods) * (len(timestamps) - 1)


class NPZDataset(Dataset):

    def __init__(self, tc_data, list_tc_name, hist_len, pred_len, logger):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.sample_len = self.hist_len + self.pred_len
        self.logger = logger
        self.data = self.get_data(tc_data, list_tc_name)

    def __getitem__(self, index):
        return self.data[index]

    def get_data(self, tc_data, list_tc_name):
        list_samples = []
        for tcn in list_tc_name:
            idx = tc_data['tc_names'] == tcn
            timestamps = tc_data['timestamps'][idx]
            lifetime = len(timestamps)
            if lifetime < self.sample_len:
                self.logger.info(f"the lifetime of typhoon {tcn} is {lifetime} ")
                continue
            target = tc_data['intensities'][idx].astype(np.float32)
            context = tc_data['era5_data'][idx].astype(np.float32)
            # sorting the data
            order_idx = [i for i, _ in sorted(enumerate(timestamps), key=lambda x: x[1])]
            if list(range(lifetime)) != order_idx:
                timestamps = timestamps[order_idx]
                target = target[order_idx]
                context = context[order_idx]

            future_target = target[order_idx]
            hist_target = future_target
            # hist_target = np.concatenate(
            #     [target[order_idx], 0.01 * np.arange(lifetime, dtype=np.float32).reshape(-1, 1)],
            #     axis=-1
            # )

            for i in range(lifetime - self.sample_len + 1):
                _timestamps = timestamps[i: i + self.sample_len]
                # self.logger.info(f"{_timestamps}")
                if not check_missing_data(_timestamps, periods={'hours': 6}):
                    self.logger.info(
                        f"There exists missing data, drop the current sample for {tcn} at {_timestamps[0]}")
                    continue
                list_samples.append(
                    (
                        [hist_target[i: i + self.hist_len], context[i: i + self.hist_len]],  # history features
                        future_target[i + self.hist_len: i + self.sample_len]  # target
                    )
                )
        return list_samples

    def __len__(self):
        return len(self.data)


class WindDataset(Dataset):

    def __init__(self, tc_data, list_tc_name, hist_len, pred_len, logger):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.sample_len = self.hist_len + self.pred_len
        self.logger = logger
        self.data = self.get_data(tc_data, list_tc_name)

    def __getitem__(self, index):
        return self.data[index]

    def get_data(self, tc_data, list_tc_name):
        list_samples = []
        for tcn in list_tc_name:
            idx = tc_data['tc_names'] == tcn
            timestamps = tc_data['timestamps'][idx]
            lifetime = len(timestamps)
            if lifetime < self.sample_len:
                self.logger.info(f"the lifetime of typhoon {tcn} is {lifetime} ")
                continue
            target = tc_data['intensities'][idx][:, :1].astype(np.float32)  # wind
            context = tc_data['era5_data'][idx].astype(np.float32)
            # sorting the data
            order_idx = [i for i, _ in sorted(enumerate(timestamps), key=lambda x: x[1])]
            if list(range(lifetime)) != order_idx:
                timestamps = timestamps[order_idx]
                target = target[order_idx]
                context = context[order_idx]

            future_target = target[order_idx]
            hist_target = np.concatenate(
                [target[order_idx], 0.00 * np.arange(lifetime, dtype=np.float32).reshape(-1, 1)],
                axis=-1
            )

            for i in range(lifetime - self.sample_len + 1):
                _timestamps = timestamps[i: i + self.sample_len]
                # self.logger.info(f"{_timestamps}")
                if not check_missing_data(_timestamps, periods={'hours': 6}):
                    self.logger.info(
                        f"There exists missing data, drop the current sample for {tcn} at {_timestamps[0]}")
                    continue
                list_samples.append(
                    (
                        [hist_target[i: i + self.hist_len], context[i: i + self.hist_len]],  # history features
                        future_target[i + self.hist_len: i + self.sample_len]  # target
                    )
                )
        return list_samples

    def __len__(self):
        return len(self.data)


class NewNPZDataset(Dataset):
    """
    tc_data: dict, typhoon data (ibtrac with era5 features)
    json_data: dict, typhoon json information, used to extract time series covariates
    list_tc_name: iterable, a sequence of tropical cyclone names
    land_sea_mask: xarray dataset, the data shows the land and sea mark with land=1, sea=0
    """

    def __init__(self, tc_data, json_data, list_tc_name, land_sea_mask, hist_len, pred_len, logger):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.sample_len = self.hist_len + self.pred_len
        self.logger = logger
        self.json_data = json_data
        self.land_sea_mask = land_sea_mask
        self.data = self.get_data(tc_data, list_tc_name)

    def __getitem__(self, index):
        return self.data[index]

    def get_data(self, tc_data, list_tc_name):
        list_samples = []
        for num_tc, tcn in enumerate(list_tc_name):
            idx = tc_data['tc_names'] == tcn
            timestamps = tc_data['timestamps'][idx]
            lifetime = len(timestamps)
            if lifetime < self.sample_len:
                self.logger.info(f"the lifetime of typhoon {tcn} is {lifetime} ")
                continue
            target = tc_data['intensities'][idx].astype(np.float32)
            context = tc_data['era5_data'][idx].astype(np.float32)
            # sorting the data
            order_idx = [i for i, _ in sorted(enumerate(timestamps), key=lambda x: x[1])]
            if list(range(lifetime)) != order_idx:
                timestamps = timestamps[order_idx]
                target = target[order_idx]
                context = context[order_idx]
            list_covariates = []  # list of list
            cvar_t = [num_tc, ]
            t = timestamps[0]
            cov_t = self.json_data[t]
            pre_info = cov_t['#&#'.join([tcn, t])]
            delta_t = datetime.strptime(t, '%Y%m%d%H') - datetime.strptime(t[:4], '%Y')
            rel_t = delta_t.days + delta_t.seconds / 86400  # num days from year beginning
            cvar_t.extend([np.sin(2 * np.pi * rel_t / 366), np.cos(2 * np.pi * rel_t / 366)])
            cvar_t.extend([0, 1])
            cvar_t.append(0)
            lat = pre_info['c_era5'][0]
            lon = pre_info['c_era5'][1]
            cvar_t.append(self.land_sea_mask.sel(latitude=lat, longitude=lon).to_array().values.squeeze())
            if lat > 66.5 or lat < -66.5:
                cat = 0
            elif lat >= 23.5 or lat <= -23.5:
                cat = 1
            else:
                cat = 2
            cvar_t.append(cat)
            list_covariates.append(cvar_t)
            _timestamps = [t]  # record the time when the tc stay in the ocean

            for i in range(lifetime-1):
                cvar_t = [num_tc,]
                t = timestamps[i+1]
                cov_t = self.json_data[t]
                current_info = cov_t['#&#'.join([tcn, t])]
                lat = current_info['c_era5'][0]
                lon = current_info['c_era5'][1]
                lsm_t = self.land_sea_mask.sel(latitude=lat, longitude=lon).to_array().values.squeeze()
                if lsm_t < 1e-3:
                    _timestamps.append(t)
                else:
                    continue  # drop the data
                delta_t = datetime.strptime(t, '%Y%m%d%H') - datetime.strptime(t[:4], '%Y')
                rel_t = delta_t.days + delta_t.seconds / 86400  # num days from year beginning
                cvar_t.extend([np.sin(2*np.pi*rel_t/366), np.cos(2*np.pi*rel_t/366)])  # time embedding covariates
                time_from_start = datetime.strptime(t, '%Y%m%d%H') - datetime.strptime(timestamps[0], '%Y%m%d%H')
                hourfstart = time_from_start.days * 24 + time_from_start.seconds / 3600
                cvar_t.extend([np.sin(2*np.pi*hourfstart/(24 * 30)), np.cos(2*np.pi*hourfstart/(24 * 30))])  # assume the lifetime is less than 30 days

                # moving velocity
                cvar_t.append(self._calc_velocity(np.array(current_info['c_era5']), np.array(pre_info['c_era5'])))


                cvar_t.append(lsm_t)
                if lat > 66.5 or lat < -66.5:
                    cat = 0
                elif lat >= 23.5 or lat <= -23.5:
                    cat = 1
                else:
                    cat = 2
                cvar_t.append(cat)
                list_covariates.append(cvar_t)
                pre_info = current_info

            future_target = target[order_idx]
            hist_target = future_target
            # hist_target = np.concatenate(
            #     [target[order_idx], 0.01 * np.arange(lifetime, dtype=np.float32).reshape(-1, 1)],
            #     axis=-1
            # )
            lifetime = len(_timestamps)
            for i in range(lifetime - self.sample_len + 1):
                _t_stamps = _timestamps[i: i + self.sample_len]
                # self.logger.info(f"{_timestamps}")
                if not check_missing_data(_t_stamps, periods={'hours': 6}):
                    self.logger.info(
                        f"There exists missing data, drop the current sample for {tcn} at {_t_stamps[0]}")
                    continue
                list_samples.append(
                    (
                        [
                            hist_target[i: i + self.hist_len],
                            context[i: i + self.hist_len],
                            np.array(list_covariates[i: i + self.hist_len], dtype=np.float32)
                        ],  # history features
                        [
                            future_target[i + self.hist_len: i + self.sample_len],  # target
                            np.array(
                                [time_feats[1:5] for time_feats in list_covariates[i + self.hist_len: i + self.sample_len]],
                                dtype=np.float32
                            )
                        ]
                    )
                )
        return list_samples  # hist_target, context, covariates (timestamps embedding, velocity, tropical_cat, land_sea)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _calc_velocity(x, y):
        """
        :param x: nd.array (lat, lon) , lat \in [-90, 90], lon \in [0, 360)
        :param y: nd.array (lat, lon)
        :return: vel
        """
        if np.abs(x[1] - y[1]) > 180:
            if x[1] > y[1]:
                x[1] -= 360
            else:
                y[1] -= 360
        delta = np.sqrt(np.sum((x - y) ** 2))
        return delta / 6
