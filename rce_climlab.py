#! /usr/bin/env python
"""Steady-state lat-by-lat RCE throughout the annual cycle using climlab."""

import contextlib
import os

import climlab
from climlab.convection import ConvectiveAdjustment
from climlab.radiation import (
    AnnualMeanInsolation,
    FixedInsolation,
    RRTMG,
)
from climlab.radiation.water_vapor import (
    FixedRelativeHumidity,
    ManabeWaterVapor,
)
from climlab.solar.insolation import daily_insolation
import numpy as np
import xarray as xr


ALBEDO = 0.3
DAY_TYPE = 2  # 2: `day` is solar longitude, 0-360 degrees
DAY_OF_YEAR = 90.  # 90 = NH summer solstice (if `day_type` is 2)
DIR_OUTPUT = 'output'
DIR_TMP = 'tmp'
DT_IN_DAYS = 20.
LAPSE_RATE = 6.5
LAPSE_RATE_DRY = 10.
LAT_STR = 'lat'
LEV_STR = 'lev'
MIXED_LAYER_DEPTH = 1.
NUM_VERT_LEVELS = 30
NUM_DAYS = 1000.
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


def ann_mean_insol(lat):
    """Get annual mean insolation."""
    state = climlab.column_state(num_lev=1, lat=lat)
    return AnnualMeanInsolation(
        domains=state.Ts.domain).insolation.to_xarray().drop('depth').squeeze()


def time_avg_insol(lat, start_day, end_day, day_type=2):
    """Average insolation over the subset of the annual cycle."""
    days = np.arange(start_day, end_day+0.1, 1)
    insolation = [daily_insolation(lat=lat, day_type=day_type, day=day)
                  for day in days]
    return np.array(insolation).mean(axis=0)


def create_rce_model(lat, day_type=DAY_TYPE, day_of_year=DAY_OF_YEAR,
                     insol_avg_window=1, num_vert_levels=NUM_VERT_LEVELS,
                     albedo=ALBEDO, dry_atmos=False, rad_model=RRTMG,
                     water_vapor=ManabeWaterVapor,
                     convec_model=ConvectiveAdjustment, lapse_rate=LAPSE_RATE,
                     temp_sfc_init=TEMP_SFC_INIT, quiet=True):
    """Create a column model for a single latitude."""
    if day_of_year == 'ann':
        insolation = ann_mean_insol(lat)
    else:
        if insol_avg_window > 1:
            dday = 0.5*insol_avg_window,
            insolation = time_avg_insol(lat, day_of_year - dday,
                                        day_of_year + dday,
                                        day_type=day_type)
        else:
            insolation = daily_insolation(lat=lat, day_type=day_type,
                                          day=day_of_year)
    coszen = coszen_from_insol(lat, insolation)

    state = climlab.column_state(num_lev=num_vert_levels,
                                 water_depth=MIXED_LAYER_DEPTH)
    if temp_sfc_init is None:
        # Leading 2^(1/4) power is from simple 1 layer greenhouse.
        temp_rad_eq = rad_equil(4*insolation, albedo)
        temp_rad_eq = max(temp_rad_eq, TEMP_MIN_VALID)
        temp_rad_eq = min(temp_rad_eq, TEMP_MAX_VALID)
        state['Ts'][:] = temp_rad_eq
    else:
        state['Ts'][:] = temp_sfc_init
    # TODO: initialize with adiabatic lapse rate.  Currently isothermal.
    state['Tatm'][:] = state['Ts'][:]

    model = climlab.TimeDependentProcess(state=state)
    if dry_atmos:
        h2o_proc = FixedRelativeHumidity(relative_humidity=0., qStrat=0.,
                                         state=state)
        print("Dry atmosphere specified, so overriding given lapse rate of "
              "'{}' with dry adiabatic lapse rate.".format(lapse_rate))
        lapse_rate = LAPSE_RATE_DRY
    else:
        h2o_proc = water_vapor(state=state)
    convec_proc = convec_model(state=state, adj_lapse_rate=lapse_rate)

    # Solar constant is four times the insolation.
    insol_proc = FixedInsolation(S0=insolation*4, domains=model.Ts.domain,
                                 coszen=coszen)
    if quiet:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            rad_proc = rad_model(state=state, specific_humidity=h2o_proc.q,
                                 albedo=albedo, insolation=insol_proc,
                                 coszen=coszen, S0=insolation*4)
    else:
        rad_proc = rad_model(state=state, specific_humidity=h2o_proc.q,
                             albedo=albedo, insolation=insol_proc,
                             coszen=coszen, S0=insolation*4)

    model.add_subprocess('Radiation', rad_proc)
    model.add_subprocess('WaterVapor', h2o_proc)
    model.add_subprocess('Convection', convec_proc)
    return model


def run_rce_model(rce_model, num_days_run=NUM_DAYS, dt_in_days=DT_IN_DAYS,
                  temp_min_valid=TEMP_MIN_VALID,
                  temp_max_valid=TEMP_MAX_VALID, quiet=False):
    """Integrate one RCE model in time and combine results into one Dataset."""
    times = time_arr(stop=num_days_run, spacing=dt_in_days)
    results = []
    for time in times:
        # Prevent model from crashing in weakly insolated columns.
        outside_valid_range = ((rce_model.Ts > temp_max_valid) or
                               (rce_model.Ts < temp_min_valid))
        if outside_valid_range:
            ds = xr.ones_like(rce_model.to_xarray().copy(deep=True))*np.nan
        else:
            if quiet:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    rce_model.integrate_days(dt_in_days)
                    ds = rce_model.to_xarray().copy(deep=True)
            else:
                rce_model.integrate_days(dt_in_days)
                ds = rce_model.to_xarray().copy(deep=True)

        # Drop unneeded mixed-layer depth and include useful attributes.
        ds = ds.squeeze('depth').drop('depth_bounds')
        ds.coords['coszen'] = rce_model.subprocess['Radiation'].coszen
        ds.coords['S0'] = rce_model.subprocess['Radiation'].S0
        ds.coords['albedo'] = rce_model.subprocess['Radiation'].asdir
        results.append(ds)
    return xr.concat(results, dim=times)


def create_and_run_rce_model(lat, day_type=DAY_TYPE, day_of_year=DAY_OF_YEAR,
                             dry_atmos=False, insol_avg_window=1,
                             num_vert_levels=NUM_VERT_LEVELS, albedo=ALBEDO,
                             rad_model=RRTMG, water_vapor=ManabeWaterVapor,
                             convec_model=ConvectiveAdjustment,
                             lapse_rate=LAPSE_RATE,
                             temp_sfc_init=TEMP_SFC_INIT,
                             dt_in_days=DT_IN_DAYS, num_days_run=NUM_DAYS,
                             temp_min_valid=TEMP_MIN_VALID,
                             temp_max_valid=TEMP_MAX_VALID,
                             write_to_disk=WRITE_TO_DISK, path_output=None,
                             quiet=True):
    """Create and run a column model for a single latitude."""
    print("lat value: {}".format(lat))
    model = create_rce_model(
        lat, day_type=day_type, day_of_year=day_of_year,
        insol_avg_window=insol_avg_window, num_vert_levels=num_vert_levels,
        albedo=albedo, dry_atmos=dry_atmos, rad_model=rad_model,
        water_vapor=water_vapor, convec_model=convec_model,
        lapse_rate=lapse_rate, temp_sfc_init=temp_sfc_init, quiet=quiet)
    ds = run_rce_model(model,
                       num_days_run=num_days_run,
                       dt_in_days=dt_in_days,
                       temp_min_valid=temp_min_valid,
                       temp_max_valid=temp_max_valid)
    ds.coords[LAT_STR] = lat

    if write_to_disk:
        if path_output == ".":
            path = os.path.join(DIR_TMP, 'rce_climlab_lat{}.nc'.format(lat))
        else:
            path = path_output

        # print(path)
        print(path)
        ds.to_netcdf(path)

    return ds


if __name__ == '__main__':
    create_and_run_rce_model()
