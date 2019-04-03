#! /usr/bin/env python
"""Steady-state lat-by-lat RCE throughout the annual cycle using climlab."""

import contextlib
from multiprocessing import cpu_count
import os

import climlab
from climlab.convection import ConvectiveAdjustment
from climlab.radiation import (
    AnnualMeanInsolation,
    FixedInsolation,
    ManabeWaterVapor,
    RRTMG,
)
from climlab.solar.insolation import daily_insolation
import dask
import dask.bag as db
import distributed
import numpy as np
import xarray as xr


ALBEDO = 0.3
DAY_TYPE = 2  # 2: `day` is solar longitude, 0-360 degrees
DAY_OF_YEAR = 45.  # 90 = NH summer solstice (if `day_type` is 2)
DIR_OUTPUT = 'output'
DLAT_DEG = 30.
DT_IN_DAYS = 1.
LAPSE_RATE = 6.5
LAT_STR = 'lat'
LEV_STR = 'lev'
MIXED_LAYER_DEPTH = 1.
NUM_ANN_CYCLE_POINTS = 12
NUM_VERT_LEVELS = 5
NUM_DAYS = 2.
SOLAR_CONST = 1365.2
STEF_BOLTZ_CONST = 5.6704e-8
TEMP_MIN_VALID = 150.
TEMP_MAX_VALID = 340.
TEMP_SFC_INIT = None
TIME_STR = 'time'
WRITE_TO_DISK = False


def _coord_arr_1d(start, stop, spacing, dim):
    """Create xr.DataArray of an evenly spaced 1D coordinate ."""
    arr_np = np.arange(start, stop + 0.1*spacing, spacing)
    return xr.DataArray(arr_np, name=dim, dims=[dim],
                        coords={dim: arr_np})


def lat_arr(start=None, stop=None, spacing=1., dim=LAT_STR):
    """Convenience function to create an array of latitudes."""
    if start is None and stop is None:
        start = -90 + 0.5*spacing
        stop = 90 - 0.5*spacing
    return _coord_arr_1d(start, stop, spacing, dim)


def time_arr(start=0, stop=100, spacing=1., dim=TIME_STR):
    """Convenience function to create an array of times."""
    return _coord_arr_1d(start, stop, spacing, dim)


def coszen_from_insol(lats, insol):
    """Get cosine of zenith angle."""
    state = climlab.column_state(num_lev=1, lat=lats)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        rad = RRTMG(state=state, insolation=insol)
    return rad.coszen


def rad_equil(solar_const=SOLAR_CONST, albedo=ALBEDO, ghg_layer=True,
              stef_boltz_const=STEF_BOLTZ_CONST):
    """Radiative equilibrium with optional single greenhouse layer."""
    temp = (solar_const*(1-albedo) / (4*STEF_BOLTZ_CONST))**0.25
    if ghg_layer:
        temp *= 2**0.25
    return temp


def ann_mean_insol(lats):
    """Get annual mean insolation."""
    state = climlab.column_state(num_lev=1, lat=lats)
    return AnnualMeanInsolation(domains=state.Ts.domain).insolation.to_xarray()


def create_rce_model(lat, insol, coszen, num_vert_levels=NUM_VERT_LEVELS,
                     albedo=ALBEDO, rad_model=RRTMG,
                     water_vapor=ManabeWaterVapor,
                     convec_model=ConvectiveAdjustment, lapse_rate=LAPSE_RATE,
                     temp_sfc_init=TEMP_SFC_INIT, quiet=True):
    """Create a column model for a single latitude."""
    state = climlab.column_state(num_lev=num_vert_levels,
                                 water_depth=MIXED_LAYER_DEPTH)
    if temp_sfc_init is None:
        # Leading 2^(1/4) power is from simple 1 layer greenhouse.
        temp_rad_eq = rad_equil(4*insol, albedo)
        temp_rad_eq = max(temp_rad_eq, TEMP_MIN_VALID)
        temp_rad_eq = min(temp_rad_eq, TEMP_MAX_VALID)
        state['Ts'][:] = temp_rad_eq
    else:
        state['Ts'][:] = temp_sfc_init
    # TODO: initialize with adiabatic lapse rate.
    state['Tatm'][:] = state['Ts'][:]

    model = climlab.TimeDependentProcess(state=state)
    convec_proc = convec_model(state=state, adj_lapse_rate=lapse_rate)
    h2o_proc = water_vapor(state=state)

    # Solar constant is four times the insolation.
    insol_proc = FixedInsolation(S0=insol*4, domains=model.Ts.domain,
                                 coszen=coszen)
    if quiet:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            rad_proc = rad_model(state=state, specific_humidity=h2o_proc.q,
                                 albedo=albedo, insolation=insol_proc,
                                 coszen=coszen, S0=insol*4)
    else:
        rad_proc = rad_model(state=state, specific_humidity=h2o_proc.q,
                             albedo=albedo, insolation=insol_proc,
                             coszen=coszen, S0=insol*4)

    model.add_subprocess('Radiation', rad_proc)
    model.add_subprocess('WaterVapor', h2o_proc)
    model.add_subprocess('Convection', convec_proc)
    return model


def run_rce_model(rce_model, times=None, dt_in_days=DT_IN_DAYS,
                  temp_min_valid=TEMP_MIN_VALID,
                  temp_max_valid=TEMP_MAX_VALID):
    """Integrate one RCE model in time and combine results into one Dataset."""
    if times is None:
        times = time_arr(stop=NUM_DAYS, spacing=dt_in_days)
    results = []
    for time in times:
        # Prevent model from crashing in weakly insolated columns.
        outside_valid_range = ((rce_model.Ts > temp_max_valid) or
                               (rce_model.Ts < temp_min_valid))
        if outside_valid_range:
            ds = xr.ones_like(rce_model.to_xarray().copy(deep=True))*np.nan
        else:
            # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            rce_model.integrate_days(dt_in_days)
            ds = rce_model.to_xarray().copy(deep=True)

        # Drop unneeded mixed-layer depth and include useful attributes.
        ds = ds.squeeze('depth').drop('depth_bounds')
        ds.coords['coszen'] = rce_model.subprocess['Radiation'].coszen
        ds.coords['S0'] = rce_model.subprocess['Radiation'].S0
        ds.coords['albedo'] = rce_model.subprocess['Radiation'].asdir
        results.append(ds)
    return xr.concat(results, dim=times)


def _run_rce_models(rce_models, lats, times, dt_in_days=DT_IN_DAYS,
                    temp_min_valid=TEMP_MIN_VALID):
    """Integrate each model in time and combine results into one Dataset."""
    results = []
    for lat, col_mod in zip(lats, rce_models):
        # click.echo('Latitude: {}'.format(float(lat)))
        results.append(run_rce_model(
            col_mod, times, dt_in_days=dt_in_days,
            temp_min_valid=temp_min_valid))
    return xr.concat(results, dim=lats).transpose()


def run_rce_models(ann_mean=False, day_type=DAY_TYPE, day_of_year=DAY_OF_YEAR,
                   albedo=ALBEDO, lapse_rate=LAPSE_RATE, lat_spacing=DLAT_DEG,
                   dt_in_days=DT_IN_DAYS, num_days_run=NUM_DAYS,
                   num_vert_levels=NUM_VERT_LEVELS,
                   temp_min_valid=TEMP_MIN_VALID,
                   temp_max_valid=TEMP_MAX_VALID, temp_sfc_init=TEMP_SFC_INIT,
                   write_to_disk=WRITE_TO_DISK):
    """Solve for lat-by-lat RCE at a fixed point in the annual cycle."""
    lats = lat_arr(spacing=lat_spacing)
    times = time_arr(stop=num_days_run, spacing=dt_in_days)

    if ann_mean:
        insolation = ann_mean_insol(lats)
    else:
        insolation = daily_insolation(lat=lats, day_type=day_type,
                                      day=day_of_year)

    coszen = coszen_from_insol(lats, insolation)

    column_models = []
    for lat, insol, cz in zip(lats.values, insolation.values, coszen.values):
        column_models.append(create_rce_model(
            lat, insol, cz, num_vert_levels=num_vert_levels,
            temp_sfc_init=temp_sfc_init, albedo=albedo, lapse_rate=lapse_rate))

    def _run_rce_model_one_arg(model):
        """dask.bag 'map' requires a function with single argument."""
        return run_rce_model(model, times=times, dt_in_days=dt_in_days,
                             temp_min_valid=temp_min_valid,
                             temp_max_valid=temp_max_valid)

    ds = _run_rce_models(column_models, lats, times, dt_in_days=dt_in_days,
                         temp_min_valid=temp_min_valid)
    return ds
    # num_workers = min(len(lats), cpu_count())
    # with distributed.LocalCluster(n_workers=num_workers) as cluster:
    #     with distributed.Client(cluster) as client:
    #         with dask.config.set(get=client.get):
    #             results = db.from_sequence(
    #                  column_models).map(_run_rce_model_one_arg).compute()

    # print(results)
    # return results


if __name__ == '__main__':
    ds = run_rce_models()
