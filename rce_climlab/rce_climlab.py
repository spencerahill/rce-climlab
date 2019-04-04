#! /usr/bin/env python
"""Steady-state lat-by-lat RCE throughout the annual cycle using climlab."""

import contextlib
import os
import shutil
from subprocess import Popen, PIPE
from time import sleep

import click
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
DLAT_DEG = 5.
DT_IN_DAYS = 20.
LAPSE_RATE = 6.5
LAPSE_RATE_DRY = 10.
LAT_STR = 'lat'
LEV_STR = 'lev'
MIXED_LAYER_DEPTH = 1.
NUM_ANN_CYCLE_POINTS = 12
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


def create_rce_model(lat, day_type=DAY_TYPE, day_of_year=DAY_OF_YEAR,
                     num_vert_levels=NUM_VERT_LEVELS, albedo=ALBEDO,
                     dry_atmos=False, rad_model=RRTMG,
                     water_vapor=ManabeWaterVapor,
                     convec_model=ConvectiveAdjustment, lapse_rate=LAPSE_RATE,
                     temp_sfc_init=TEMP_SFC_INIT, quiet=True):
    """Create a column model for a single latitude."""
    if day_of_year == 'ann':
        insolation = ann_mean_insol(lat)
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
                  temp_max_valid=TEMP_MAX_VALID):
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


def _run_rce_models(rce_models, lats, num_days_run=NUM_DAYS,
                    dt_in_days=DT_IN_DAYS, temp_min_valid=TEMP_MIN_VALID):
    """Integrate each model in time and combine results into one Dataset."""
    results = []
    for lat, col_mod in zip(lats, rce_models):
        click.echo('Latitude: {}'.format(float(lat)))
        results.append(run_rce_model(
            col_mod, num_days_run=num_days_run, dt_in_days=dt_in_days,
            temp_min_valid=temp_min_valid))
    return xr.concat(results, dim=lats).transpose()


@click.command()
@click.option('--lat', default=0.)
@click.option('--day-type', 'day_type', default=DAY_TYPE)
@click.option('--day-of-year', 'day_of_year', default=DAY_OF_YEAR)
@click.option('--albedo', 'albedo', default=ALBEDO)
@click.option('--dry-atmos', 'dry_atmos', default=False, is_flag=True)
@click.option('--lapse-rate', 'lapse_rate', default=LAPSE_RATE)
@click.option('--dt-in-days', 'dt_in_days', default=DT_IN_DAYS)
@click.option('--num-days-run', 'num_days_run', default=NUM_DAYS)
@click.option('--num-vert-levels', 'num_vert_levels', default=NUM_VERT_LEVELS)
@click.option('--temp-min-valid', 'temp_min_valid', default=TEMP_MIN_VALID)
@click.option('--temp-max-valid', 'temp_max_valid', default=TEMP_MAX_VALID)
@click.option('--temp-sfc-init', 'temp_sfc_init', default=TEMP_SFC_INIT)
@click.option('--path-output', 'path_output', default=".")
@click.option('--write-to-disk', 'write_to_disk',
              default=WRITE_TO_DISK, is_flag=True)
def create_and_run_rce_model(lat, day_type=DAY_TYPE, day_of_year=DAY_OF_YEAR,
                             dry_atmos=False, num_vert_levels=NUM_VERT_LEVELS,
                             albedo=ALBEDO, rad_model=RRTMG,
                             water_vapor=ManabeWaterVapor,
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
        num_vert_levels=num_vert_levels, albedo=albedo, dry_atmos=dry_atmos,
        rad_model=rad_model, water_vapor=water_vapor,
        convec_model=convec_model, lapse_rate=lapse_rate,
        temp_sfc_init=temp_sfc_init, quiet=quiet)
    ds = run_rce_model(model, num_days_run=num_days_run, dt_in_days=dt_in_days,
                       temp_min_valid=temp_min_valid,
                       temp_max_valid=temp_max_valid)
    ds.coords[LAT_STR] = lat

    if write_to_disk:
        if path_output == ".":
            path = os.path.join(DIR_TMP, 'rce_climlab_lat{}.nc'.format(lat))
        else:
            path = path_output

        click.echo(path)
        ds.to_netcdf(path)

    return ds


def popen_call_and_monitor(popen_args, popen_kwargs):
    """Call popen to submit jobs and monitor the results."""
    running_procs = [Popen(args, **popen_kwargs) for args in popen_args]

    while running_procs:
        for proc in running_procs:
            retcode = proc.poll()
            if retcode == -11:
                click.echo('Segmentation fault for proc {}'.format(proc))
                running_procs.remove(proc)
                break
            elif retcode == 0:
                running_procs.remove(proc)
                click.echo('Successful integration for proc {}'.format(proc))
                break
            elif retcode is not None:
                running_procs.remove(proc)
                click.echo('Retcode {0} for proc {1}'.format(retcode, proc))
                break
            else:
                # Process is not done; wait and then check again.
                sleep(.1)
                continue


def _create_and_run_rce_models(lats, insolation, coszen, albedo=ALBEDO,
                               dry_atmos=False, lapse_rate=LAPSE_RATE,
                               dt_in_days=DT_IN_DAYS, num_days_run=NUM_DAYS,
                               num_vert_levels=NUM_VERT_LEVELS,
                               temp_min_valid=TEMP_MIN_VALID,
                               temp_max_valid=TEMP_MAX_VALID,
                               temp_sfc_init=TEMP_SFC_INIT):
    """Integrate each model in time and combine results into one Dataset."""
    def _popen_args(lat, insol, cz):
        args = ['python', 'rce_climlab_single_lat.py']
        if dry_atmos:
            args += ['--dry-atmos']
        args += ['--lat={:f}'.format(lat),
                 '--insol={:f}'.format(insol),
                 '--coszen={:f}'.format(cz),
                 '--albedo={:f}'.format(albedo),
                 '--lapse-rate={:f}'.format(lapse_rate),
                 '--dt-in-days={:f}'.format(dt_in_days),
                 '--num-days-run={:f}'.format(num_days_run),
                 '--num-vert-levels={:d}'.format(num_vert_levels),
                 '--temp-min-valid={:f}'.format(temp_min_valid),
                 '--temp-max-valid={:f}'.format(temp_max_valid)]
        return args

    popen_args = [_popen_args(lat, insol, cz) for lat, insol, cz in
                  zip(lats.values, insolation.values, coszen.values)]
    popen_kwargs = dict(stdout=PIPE, stderr=PIPE)
    popen_call_and_monitor(popen_args, popen_kwargs)

    ds = xr.open_mfdataset(os.path.join(DIR_TMP, '*.nc'),
                           concat_dim=LAT_STR).transpose()
    return ds.sortby(LAT_STR)


@click.command()
@click.option('--ann-mean', 'ann_mean', default=False, is_flag=True)
@click.option('--dry-atmos', 'dry_atmos', default=False, is_flag=True)
@click.option('--day-type', 'day_type', default=DAY_TYPE)
@click.option('--day-of-year', 'day_of_year', default=DAY_OF_YEAR)
@click.option('--albedo', 'albedo', default=ALBEDO)
@click.option('--lapse-rate', 'lapse_rate', default=LAPSE_RATE)
@click.option('--dt-in-days', 'dt_in_days', default=DT_IN_DAYS)
@click.option('--num-days-run', 'num_days_run', default=NUM_DAYS)
@click.option('--lat-spacing', 'lat_spacing', default=DLAT_DEG)
@click.option('--num-vert-levels', 'num_vert_levels', default=NUM_VERT_LEVELS)
@click.option('--temp-min-valid', 'temp_min_valid', default=TEMP_MIN_VALID)
@click.option('--temp-max-valid', 'temp_max_valid', default=TEMP_MAX_VALID)
@click.option('--temp-sfc-init', 'temp_sfc_init', default=TEMP_SFC_INIT)
@click.option('--write-to-disk', 'write_to_disk',
              default=WRITE_TO_DISK, is_flag=True)
def run_rce_models(ann_mean=False, day_type=DAY_TYPE, day_of_year=DAY_OF_YEAR,
                   albedo=ALBEDO, dry_atmos=False, lapse_rate=LAPSE_RATE,
                   lat_spacing=DLAT_DEG, dt_in_days=DT_IN_DAYS,
                   num_days_run=NUM_DAYS, num_vert_levels=NUM_VERT_LEVELS,
                   temp_min_valid=TEMP_MIN_VALID,
                   temp_max_valid=TEMP_MAX_VALID, temp_sfc_init=TEMP_SFC_INIT,
                   write_to_disk=WRITE_TO_DISK):
    """Solve for lat-by-lat RCE at a fixed point in the annual cycle."""
    # Clear contents of directory that holds the intermediate netCDF files.
    shutil.rmtree(DIR_TMP)
    os.mkdir(DIR_TMP)

    lats = lat_arr(spacing=lat_spacing)
    if ann_mean:
        insolation = ann_mean_insol(lats)
    else:
        insolation = daily_insolation(lat=lats, day_type=day_type,
                                      day=day_of_year)
    coszen = coszen_from_insol(lats, insolation)

    ds = _create_and_run_rce_models(
        lats, insolation, coszen, albedo=albedo,
        dry_atmos=dry_atmos, lapse_rate=lapse_rate, dt_in_days=dt_in_days,
        num_days_run=num_days_run, temp_min_valid=temp_min_valid,
        num_vert_levels=num_vert_levels, temp_sfc_init=temp_sfc_init)

    if ann_mean:
        filename = 'rce_ann_mean'
    else:
        filename = 'rce_solar_lon{0:03d}'.format(day_of_year)
    if dry_atmos:
        filename += '_dry'
    filename += '_albedo{:0.1f}.nc'.format(albedo)
    path_out = os.path.join(DIR_OUTPUT, filename)

    if write_to_disk:
        ds.to_netcdf(path_out)
        click.echo('Written to: {}'.format(path_out))
    else:
        click.echo("'--write-to-disk' option set to False.  If True, would "
                   "have written to: {}".format(path_out))
    return ds


# @click.command()
# @click.option('--ann-mean', 'ann_mean', default=False, is_flag=True)
# @click.option('--day-type', 'day_type', default=DAY_TYPE)
# @click.option('--day-of-year', 'day_of_year', default=DAY_OF_YEAR)
# @click.option('--albedo', 'albedo', default=ALBEDO)
# @click.option('--lapse-rate', 'lapse_rate', default=LAPSE_RATE)
# @click.option('--dt-in-days', 'dt_in_days', default=DT_IN_DAYS)
# @click.option('--num-days-run', 'num_days_run', default=NUM_DAYS)
# @click.option('--lat-spacing', 'lat_spacing', default=DLAT_DEG)
# @click.option('--num-vert-levels', 'num_vert_levels', default=NUM_VERT_LEVELS)
# @click.option('--temp-min-valid', 'temp_min_valid', default=TEMP_MIN_VALID)
# @click.option('--temp-max-valid', 'temp_max_valid', default=TEMP_MAX_VALID)
# @click.option('--temp-sfc-init', 'temp_sfc_init', default=TEMP_SFC_INIT)
# @click.option('--write-to-disk', 'write_to_disk',
#               default=WRITE_TO_DISK, is_flag=True)
# def run_rce_models(ann_mean=False, day_type=DAY_TYPE, day_of_year=DAY_OF_YEAR,
#                    albedo=ALBEDO, lapse_rate=LAPSE_RATE, lat_spacing=DLAT_DEG,
#                    dt_in_days=DT_IN_DAYS, num_days_run=NUM_DAYS,
#                    num_vert_levels=NUM_VERT_LEVELS,
#                    temp_min_valid=TEMP_MIN_VALID,
#                    temp_max_valid=TEMP_MAX_VALID, temp_sfc_init=TEMP_SFC_INIT,
#                    write_to_disk=WRITE_TO_DISK):
#     """Solve for lat-by-lat RCE at a fixed point in the annual cycle."""
#     lats = lat_arr(spacing=lat_spacing)
#     times = time_arr(stop=num_days_run, spacing=dt_in_days)
#     if ann_mean:
#         insolation = ann_mean_insol(lats)
#     else:
#         insolation = daily_insolation(lat=lats, day_type=day_type,
#                                       day=day_of_year)
#     coszen = coszen_from_insol(lats, insolation)

#     column_models = []
#     for lat, insol, cz in zip(lats.values, insolation.values, coszen.values):
#         column_models.append(create_rce_model(
#             lat, insol, cz, num_vert_levels=num_vert_levels,
#             temp_sfc_init=temp_sfc_init, albedo=albedo, lapse_rate=lapse_rate))

#     ds = _run_rce_models(column_models, lats, times, dt_in_days=dt_in_days,
#                          temp_min_valid=temp_min_valid)

#     if write_to_disk:
#         if ann_mean:
#             filename = 'rce_ann_mean_albedo{:.1f}.nc'.format(albedo)
#         else:
#             filename = 'rce_solar_lon{0:03d}_albedo{1:0.1f}.nc'.format(
#                 day_of_year, albedo)
#         path_out = os.path.join(DIR_OUTPUT, filename)
#         ds.to_netcdf(path_out)
#         click.echo('Written to: {}'.format(path_out))
#     return ds


@click.command()
@click.option('-n', '--num-ann-cycle-points', 'num_ann_cycle_points',
              default=NUM_ANN_CYCLE_POINTS,
              help='Number of evenly spaced points in the annual cycle.')
@click.option('--dry-atmos', 'dry_atmos', default=False, is_flag=True)
@click.option('--albedo', 'albedo', default=ALBEDO)
@click.option('--lapse-rate', 'lapse_rate', default=LAPSE_RATE)
@click.option('--dt-in-days', 'dt_in_days', default=DT_IN_DAYS)
@click.option('--num-days-run', 'num_days_run', default=NUM_DAYS)
@click.option('--lat-spacing', 'lat_spacing', default=DLAT_DEG)
@click.option('--num-vert-levels', 'num_vert_levels', default=NUM_VERT_LEVELS)
@click.option('--temp-min-valid', 'temp_min_valid', default=TEMP_MIN_VALID)
@click.option('--temp-max-valid', 'temp_max_valid', default=TEMP_MAX_VALID)
@click.option('--temp-sfc-init', 'temp_sfc_init', default=TEMP_SFC_INIT)
def rce_ann_cycle(num_ann_cycle_points=NUM_ANN_CYCLE_POINTS, dry_atmos=False,
                  albedo=ALBEDO, lapse_rate=LAPSE_RATE, lat_spacing=DLAT_DEG,
                  dt_in_days=DT_IN_DAYS, num_days_run=NUM_DAYS,
                  num_vert_levels=NUM_VERT_LEVELS,
                  temp_min_valid=TEMP_MIN_VALID, temp_max_valid=TEMP_MAX_VALID,
                  temp_sfc_init=TEMP_SFC_INIT, write_to_disk=WRITE_TO_DISK):
    """Compute lat-by-lat RCE at fixed points throughout annual cycle."""
    dt_ann_cyc = 360 / num_ann_cycle_points
    days = np.arange(0.5*dt_ann_cyc, 12 - 0.49*dt_ann_cyc, dt_ann_cyc)

    # Logic adapted from https://stackoverflow.com/a/636601/1706640
    popen_args = [['python', 'rce_climlab.py', '--day-type=2',
                   '--day-of-year={:f}'.format(day),
                   '--albedo={:f}'.format(albedo),
                   '--lapse-rate={:f}'.format(lapse_rate),
                   '--lat-spacing={:f}'.format(lat_spacing),
                   '--dt-in-days={:f}'.format(dt_in_days),
                   '--num-days-run={:f}'.format(num_days_run),
                   '--num-vert-levels={:d}'.format(num_vert_levels),
                   '--temp-min-valid={:f}'.format(temp_min_valid),
                   '--temp-max-valid={:f}'.format(temp_max_valid)]
                  for day in days]

    def _popen_args(day):
        args = ['python', 'rce_climlab.py', '--day-type=2']
        if dry_atmos:
            args += ['--dry-atmos']
        args += ['--day-of-year={:f}'.format(day),
                 '--albedo={:f}'.format(albedo),
                 '--lapse-rate={:f}'.format(lapse_rate),
                 '--lat-spacing={:f}'.format(lat_spacing),
                 '--dt-in-days={:f}'.format(dt_in_days),
                 '--num-days-run={:f}'.format(num_days_run),
                 '--num-vert-levels={:d}'.format(num_vert_levels),
                 '--temp-min-valid={:f}'.format(temp_min_valid),
                 '--temp-max-valid={:f}'.format(temp_max_valid)]
        return args

    popen_args = [_popen_args(day) for day in days]
    popen_kwargs = dict(stdout=PIPE, stderr=PIPE)
    popen_call_and_monitor(popen_args, popen_kwargs)


def rce_ann_mean(lat_spacing=DLAT_DEG, dt_days=DT_IN_DAYS,
                 num_days_run=NUM_DAYS):
    """Lat-by-lat RCE using annual mean insolation."""
    lats = lat_arr(spacing=lat_spacing)
    times = time_arr(0, num_days_run, dt_days)

    state = climlab.column_state(num_lev=NUM_VERT_LEVELS, lat=lats,
                                 water_depth=MIXED_LAYER_DEPTH)
    rce = climlab.TimeDependentProcess(state=state)
    insol = AnnualMeanInsolation(domains=rce.Ts.domain)
    h2o = ManabeWaterVapor(state=state)
    conv_adj = ConvectiveAdjustment(state=state, adj_lapse_rate=LAPSE_RATE)
    rad = RRTMG(state=state, specific_humidity=h2o.q, albedo=ALBEDO,
                S0=insol.S0, insolation=insol.insolation,
                coszen=insol.coszen)

    rce.add_subprocess('Insolation', insol)
    rce.add_subprocess('Radiation', rad)
    rce.add_subprocess('WaterVapor', h2o)
    rce.add_subprocess('Convection', conv_adj)

    rce.integrate_years(0)
    solutions = []
    for time in times:
        solutions.append(rce.to_xarray().copy(deep=True))
        rce.integrate_days(dt_days)

    ds = xr.concat(solutions, dim=times)

    ds.to_netcdf('output/rce_ann_mean.nc', format='NETCDF4')
    return ds


if __name__ == '__main__':
    # ds = run_rce_models()
    create_and_run_rce_model()
