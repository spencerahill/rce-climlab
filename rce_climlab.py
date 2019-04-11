#! /usr/bin/env python
"""Steady-state lat-by-lat RCE throughout the annual cycle using climlab."""

from contextlib import contextmanager, redirect_stdout
import os

import climlab
from climlab.convection import ConvectiveAdjustment
from climlab.radiation import (
    AnnualMeanInsolation,
    DailyInsolation,
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
DAY_TYPE = 1  # 1: `day` is calendar day, starting January 1
DAY_OF_YEAR = 80.  # 80 = NH spring equinox
DAY_OF_YEAR_STR = 'day_of_year'
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


@contextmanager
def suppress_stdout(yes=True):
    if yes:
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            yield
    else:
        yield


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


def days_of_year_arr(start=1, stop=365, spacing=1., dim=DAY_OF_YEAR_STR):
    """Convenience function to create an array of times."""
    return _coord_arr_1d(start, stop, spacing, dim)


def coszen_from_insol(lats, insol, quiet=True):
    """Get cosine of zenith angle."""
    state = climlab.column_state(num_lev=1, lat=lats)
    with suppress_stdout(yes=quiet):
        return RRTMG(state=state, insolation=insol).coszen


def ann_mean_insol(lat):
    """Get annual mean insolation."""
    state = climlab.column_state(num_lev=1, lat=lat)
    return AnnualMeanInsolation(
        domains=state.Ts.domain).insolation.to_xarray().drop('depth').squeeze()


def time_avg_insol(lat, start_day, end_day, day_type=1):
    """Average insolation over the subset of the annual cycle."""
    days = np.arange(start_day, end_day+0.1, 1)
    insolation = [daily_insolation(lat=lat, day_type=day_type, day=day)
                  for day in days]
    return np.array(insolation).mean(axis=0)


def _gen_column_state(lat,
                      insol_type: str = 'fixed_day',
                      num_vert_levels: int = NUM_VERT_LEVELS,
                      mixed_layer_depth: float = MIXED_LAYER_DEPTH):
    """Generate column state for climlab."""
    # Hack to get climlab's DailyInsolation to work at a single latitude: just
    # create a len-2 array with the same latitude.  Later in the pipeline,
    # strip off that duplicate value.
    do_duplicate = (insol_type == 'ann_cycle' and
                    (np.isscalar(lat) or np.size(lat) == 1))
    if do_duplicate:
        state_lat = [lat, lat]
    else:
        state_lat = lat
    return climlab.column_state(lat=state_lat,
                                num_lev=num_vert_levels,
                                water_depth=mixed_layer_depth)


def _gen_h2o_proc(state,
                  dry_atmos: bool = False,
                  water_vapor=ManabeWaterVapor):
    """Generate the water vapor process for climlab."""
    if dry_atmos:
        return FixedRelativeHumidity(relative_humidity=0., qStrat=0.,
                                     state=state)
    return water_vapor(state=state)


def _gen_convec_proc(state, convec_model=ConvectiveAdjustment,
                     dry_atmos: bool = False,
                     lapse_rate: float = LAPSE_RATE):
    """Generate the convection process for climlab."""
    if dry_atmos:
        print("Dry atmosphere specified, so overriding given lapse rate of "
              "'{}' with dry adiabatic lapse rate.".format(lapse_rate))
        lapse_rate = LAPSE_RATE_DRY
    return convec_model(state=state, adj_lapse_rate=lapse_rate)


def _gen_insol_proc(lat, state,
                    insol_type: str = 'fixed_day',
                    day_of_year: int = DAY_OF_YEAR,
                    day_type: int = DAY_TYPE,
                    insol_avg_window: int = 1):
    """Generate insolation and corresponding zenith angle arrays."""
    if insol_type == 'ann_cycle':
        return DailyInsolation(state=state, domains=state.Ts.domain)
    if insol_type == 'ann_mean':
        insolation = ann_mean_insol(lat)
    elif insol_type == 'fixed_day':
        if insol_avg_window > 1:
            dday = 0.5*insol_avg_window,
            insolation = time_avg_insol(lat, day_of_year - dday,
                                        day_of_year + dday,
                                        day_type=day_type)
        else:
            insolation = daily_insolation(lat=lat, day_type=day_type,
                                          day=day_of_year)
    else:
        VALID_INSOL_TYPES = ('fixed_day', 'ann_mean', 'ann_cycle')
        raise ValueError("insol type must be one of {0}.  "
                         "Got {1}".format(VALID_INSOL_TYPES, insol_type))

    coszen = coszen_from_insol(lat, insolation)

    # Solar constant is insolation divided by cosine of the zenith angle.
    return FixedInsolation(S0=insolation/coszen,
                           domains=state.Ts.domain,
                           coszen=coszen)


def _gen_rad_proc(state, insol_proc, h2o_proc, albedo,
                  rad_model=RRTMG,
                  quiet: bool = True):
    """Generate the radiation process for climlab."""
    with suppress_stdout(yes=quiet):
        return rad_model(state=state,
                         specific_humidity=h2o_proc.q,
                         albedo=albedo,
                         insolation=insol_proc.insolation,
                         coszen=insol_proc.coszen,
                         S0=insol_proc.S0)


def _gen_col_model(state, h2o_proc, convec_proc, insol_proc, rad_proc):
    """Generate a climlab model by combining the given subprocesses."""
    model = climlab.TimeDependentProcess(state=state)
    model.add_subprocess('WaterVapor', h2o_proc)
    model.add_subprocess('Convection', convec_proc)
    model.add_subprocess('Insolation', insol_proc)
    model.add_subprocess('Radiation', rad_proc)
    return model


def rad_equil(solar_const: float = SOLAR_CONST,
              albedo: float = ALBEDO,
              ghg_layers: int = 1,
              stef_boltz_const: float = STEF_BOLTZ_CONST) -> float:
    """Radiative equilibrium with optional single greenhouse layer."""
    temp = (solar_const*(1-albedo) / (4*STEF_BOLTZ_CONST))**0.25
    temp *= (2**0.25)**int(ghg_layers)
    return temp


def _init_temps(state, rad_proc, temp_sfc_init=TEMP_SFC_INIT):
    """Initialize surface and atmospheric temperatures for climlab."""
    if temp_sfc_init is None:
        temp_rad_eq = rad_equil(rad_proc.insolation*rad_proc.coszen,
                                rad_proc.albedo)
        temp_rad_eq = max(temp_rad_eq, TEMP_MIN_VALID)
        temp_rad_eq = min(temp_rad_eq, TEMP_MAX_VALID)
        state['Ts'][:] = temp_rad_eq
    else:
        state['Ts'][:] = temp_sfc_init
    # TODO: initialize with adiabatic lapse rate.  Currently isothermal.
    # TODO: simpler intermediate solution: just drop temperature uniformly
    # across the levels over say 100 K from surface to TOA.
    state['Tatm'][:] = state['Ts'][:]
    return state


def create_rce_model(lat,
                     day_type=DAY_TYPE,
                     insol_type='fixed_day',
                     day_of_year=DAY_OF_YEAR,
                     insol_avg_window=1,
                     num_vert_levels=NUM_VERT_LEVELS,
                     albedo=ALBEDO,
                     mixed_layer_depth=MIXED_LAYER_DEPTH,
                     dry_atmos=False,
                     rad_model=RRTMG,
                     water_vapor=ManabeWaterVapor,
                     convec_model=ConvectiveAdjustment,
                     lapse_rate=LAPSE_RATE,
                     temp_sfc_init=TEMP_SFC_INIT,
                     quiet=True):
    """Create a column model for a single latitude."""
    state = _gen_column_state(lat,
                              insol_type=insol_type,
                              num_vert_levels=num_vert_levels,
                              mixed_layer_depth=mixed_layer_depth)
    h2o_proc = _gen_h2o_proc(state,
                             dry_atmos=dry_atmos,
                             water_vapor=water_vapor)
    convec_proc = _gen_convec_proc(state,
                                   convec_model=convec_model,
                                   lapse_rate=lapse_rate,
                                   dry_atmos=dry_atmos)
    insol_proc = _gen_insol_proc(lat, state,
                                 insol_type=insol_type,
                                 day_of_year=day_of_year,
                                 day_type=day_type,
                                 insol_avg_window=insol_avg_window)
    rad_proc = _gen_rad_proc(state, insol_proc, h2o_proc, albedo,
                             rad_model=rad_model,
                             quiet=quiet)
    state = _init_temps(state, rad_proc, temp_sfc_init=temp_sfc_init)
    return _gen_col_model(state, h2o_proc, convec_proc, insol_proc, rad_proc)


def _add_metadata(ds, model):
    """Drop unneeded mixed-layer depth bounds and include useful attributes."""
    ds = ds.squeeze('depth').drop('depth_bounds')
    ds.coords['coszen'] = model.subprocess['Radiation'].coszen
    ds.coords['S0'] = model.subprocess['Radiation'].S0
    ds.coords['albedo'] = model.subprocess['Radiation'].asdir
    return ds


def _advance_one_dt(rce_model, dt_in_days,
                    check_temps_valid: bool = True,
                    temp_min_valid: float = TEMP_MIN_VALID,
                    temp_max_valid: float = TEMP_MAX_VALID,
                    quiet: bool = False):
    """Advance the model by one timestep."""
    if check_temps_valid:
        # Prevent model from crashing in weakly insolated columns.
        outside_valid_range = ((np.any(rce_model.Ts > temp_max_valid)) or
                               (np.any(rce_model.Ts < temp_min_valid)))
    else:
        outside_valid_range = False
    if outside_valid_range:
        ds = xr.ones_like(rce_model.to_xarray().copy(deep=True))*np.nan
    else:
        with suppress_stdout(yes=quiet):
            rce_model.integrate_days(dt_in_days)
            ds = rce_model.to_xarray().copy(deep=True)
    # Hack to use time-varying daily insolation for single latitude: previously
    # duplicated the latitude; now remove the duplicate.
    if rce_model.Ts.domain.shape[0] == 2:
        ds = ds.isel(**{LAT_STR: 0})
    else:
        # TODO: fix hack logic re: annual cycle so that metadata gets appended
        # for annual cycle too.
        ds = _add_metadata(ds, rce_model)
    return ds


def run_rce_model(model,
                  num_days_run: int = NUM_DAYS,
                  dt_in_days: int = DT_IN_DAYS,
                  check_temps_valid: bool = True,
                  temp_min_valid: float = TEMP_MIN_VALID,
                  temp_max_valid: float = TEMP_MAX_VALID,
                  quiet: bool = False):
    """Integrate one RCE model in time, combining results into one Dataset."""
    times = time_arr(stop=num_days_run, spacing=dt_in_days)
    results = []
    for time in times:
        results.append(_advance_one_dt(model, dt_in_days,
                                       check_temps_valid=check_temps_valid,
                                       temp_min_valid=temp_min_valid,
                                       temp_max_valid=temp_max_valid))
    return xr.concat(results, dim=times)


def create_and_run_rce_model(lat,
                             insol_type='fixed_day',
                             day_type=DAY_TYPE,
                             day_of_year=DAY_OF_YEAR,
                             dry_atmos=False,
                             insol_avg_window=1,
                             mixed_layer_depth=MIXED_LAYER_DEPTH,
                             num_vert_levels=NUM_VERT_LEVELS,
                             albedo=ALBEDO,
                             rad_model=RRTMG,
                             water_vapor=ManabeWaterVapor,
                             convec_model=ConvectiveAdjustment,
                             lapse_rate=LAPSE_RATE,
                             temp_sfc_init=TEMP_SFC_INIT,
                             dt_in_days=DT_IN_DAYS,
                             num_days_run=NUM_DAYS,
                             check_temps_valid=True,
                             temp_min_valid=TEMP_MIN_VALID,
                             temp_max_valid=TEMP_MAX_VALID,
                             write_to_disk=WRITE_TO_DISK,
                             path_output=None,
                             quiet=True):
    """Create and run a column model for a single latitude."""
    print("lat value: {}".format(lat))
    model = create_rce_model(
        lat,
        insol_type=insol_type,
        day_type=day_type,
        day_of_year=day_of_year,
        insol_avg_window=insol_avg_window,
        num_vert_levels=num_vert_levels,
        albedo=albedo,
        dry_atmos=dry_atmos,
        rad_model=rad_model,
        water_vapor=water_vapor,
        convec_model=convec_model,
        lapse_rate=lapse_rate,
        temp_sfc_init=temp_sfc_init,
        quiet=quiet,
    )
    ds = run_rce_model(
        model,
        num_days_run=num_days_run,
        dt_in_days=dt_in_days,
        check_temps_valid=check_temps_valid,
        temp_min_valid=temp_min_valid,
        temp_max_valid=temp_max_valid
    )
    ds.coords[LAT_STR] = lat

    if write_to_disk:
        if path_output is None:
            path = os.path.join(DIR_TMP, 'rce_climlab_lat{}.nc'.format(lat))
        else:
            path = path_output

        print(path)
        ds.to_netcdf(path)

    return ds


if __name__ == '__main__':
    pass
