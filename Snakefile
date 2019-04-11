"""
Example usage:

snakemake --cluster "sbatch --output=logs/slurm-%A_%a.out
--error=logs/slurm-%A_%a.err --time=00:10:00 --export=ALL" --jobs
50 --latency-wait 60 out/moist/albedo_0.4/ann_cycle.nc --keep-going

"""

configfile: "config.yaml"


import numpy as np


ANN_CYC_SPACING = 360. / config['grid']['num_ann_cycle_points']
DAYS_OF_YEAR = np.arange(0.5*ANN_CYC_SPACING, 360-0.49*ANN_CYC_SPACING,
                         ANN_CYC_SPACING)
DLAT = config['grid']['dlat_deg']
LATS = np.arange(-90+0.5*DLAT, 90-0.49*DLAT, DLAT)


rule daily_insolation:
    output:
        "out/daily_insol/day_of_year_{day_of_year}/"
        "kyr_bp_{kyr_bp}/daily_insol.nc"
    run:
        from climlab.solar.insolation import daily_insolation
        from climlab.solar.orbital import OrbitalTable
        import numpy as np
        orb = OrbitalTable.interp(kyear=-1*float(wildcards['kyr_bp']))
        insol = daily_insolation(lat=LATS, day=int(wildcards['day_of_year']),
                                 orb=orb)
        insol.name = 'insolation'
        insol.attrs['units'] = 'W/m^2'
        insol.attrs['day_of_year'] = int(wildcards['day_of_year'])
        insol.attrs['day_type'] = 1
        insol.to_netcdf(output[0])


rule rce_single_lat:
    output:
        "tmp/{dry_moist}/albedo_{albedo}/day_of_year_{day_of_year}"
        "/lat_{lat}.nc"
    run:
        from rce_climlab import create_and_run_rce_model

        if wildcards['dry_moist'] == 'dry':
            dry_atmos = True
        elif wildcards['dry_moist'] == 'moist':
            dry_atmos = False
        else:
            raise ValueError("Dry/moist flag must be either 'dry' or 'moist'. "
                             "Value given: {}".format(wildcards['dry_moist']))

        create_and_run_rce_model(
            float(wildcards['lat']),
            insol_type='fixed_day',
            day_type=1,
            day_of_year=float(wildcards['day_of_year']),
            insol_avg_window=config['model']['insol_avg_window'],
            dry_atmos=dry_atmos,
            num_vert_levels=config['grid']['num_vert_levels'],
            albedo=float(wildcards['albedo']),
            lapse_rate=config['model']['lapse_rate'],
            dt_in_days=config['grid']['dtime_days'],
            num_days_run=config['grid']['num_days'],
            temp_min_valid=config['runtime']['temp_min_valid'],
            temp_max_valid=config['runtime']['temp_max_valid'],
            write_to_disk=True,
            quiet=config['runtime']['quiet'],
            path_output=output[0],
        )


rule rce_mult_lats:
    input:
        expand("tmp/{{dry_moist}}/albedo_{{albedo}}/"
               "day_of_year_{{day_of_year}}/lat_{lat}.nc", lat=LATS)
    output:
        "out/{dry_moist}/albedo_{albedo}/day_of_year_{day_of_year}.nc"
    run:
        import xarray as xr
        ds = xr.open_mfdataset(input[:],
                               concat_dim=config['str']['lat_str']).transpose()
        ds.to_netcdf(output[0])


rule rce_ann_cycle:
    input:
        expand("out/{{dry_moist}}/albedo_{{albedo}}/day_of_year_{day}.nc",
               day=DAYS_OF_YEAR)
    output:
        "out/{dry_moist}/albedo_{albedo}/rce_ann_cycle.nc"
    run:
        import xarray as xr
        ds = xr.open_mfdataset(input[:], concat_dim=config['str']['day_str'])
        ds.to_netcdf(output[0])


rule transient_ann_cycle_single_lat:
    output:
        "tmp/{dry_moist}/albedo_{albedo}/ml_depth_{ml_depth}/"
        "transient_ann_cycle_lat_{lat}.nc"
    run:
        from rce_climlab import create_and_run_rce_model

        if wildcards['dry_moist'] == 'dry':
            dry_atmos = True
        elif wildcards['dry_moist'] == 'moist':
            dry_atmos = False
        else:
            raise ValueError("Dry/moist flag must be either 'dry' or 'moist'. "
                             "Value given: {}".format(wildcards['dry_moist']))

        create_and_run_rce_model(
            float(wildcards['lat']),
            insol_type='ann_cycle',
            mixed_layer_depth=wildcards['ml_depth'],
            dry_atmos=dry_atmos,
            num_vert_levels=config['grid']['num_vert_levels'],
            albedo=float(wildcards['albedo']),
            lapse_rate=config['model']['lapse_rate'],
            dt_in_days=20,
            num_days_run=365*5,
            check_temps_valid=False,
            temp_min_valid=config['runtime']['temp_min_valid'],
            temp_max_valid=config['runtime']['temp_max_valid'],
            temp_sfc_init=280.,
            write_to_disk=True,
            quiet=config['runtime']['quiet'],
            path_output=output[0],
        )


rule transient_ann_cycle_mult_lats:
    input:
        expand("tmp/{{dry_moist}}/albedo_{{albedo}}/ml_depth_{{ml_depth}}/"
               "transient_ann_cycle_lat_{lat}.nc", lat=LATS)
    output:
        "out/{dry_moist}/albedo_{albedo}/ml_depth_{ml_depth}/"
        "transient_ann_cycle.nc"
    run:
        import xarray as xr
        ds = xr.open_mfdataset(input[:],
                               concat_dim=config['str']['lat_str']).transpose()
        ds.to_netcdf(output[0])
