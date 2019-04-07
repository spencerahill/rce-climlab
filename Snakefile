configfile: "config.yaml"

import numpy as np

ANN_CYC_SPACING = 360. / config['grid']['num_ann_cycle_points']
DAYS_OF_YEAR = np.arange(0.5*ANN_CYC_SPACING, 360-0.49*ANN_CYC_SPACING,
                         ANN_CYC_SPACING)
DLAT = config['grid']['dlat_deg']
LATS = np.arange(-90+0.5*DLAT, 90-0.49*DLAT, DLAT)

rule rce_single_lat:
    output:
        "tmp/{dry_moist}/albedo_{albedo}/solar_lon_{day_of_year}/lat_{lat}.nc"
    log:
        "logs/single_lat/{dry_moist}/albedo_{albedo}/"
        "solar_lon_{day_of_year}/lat_{lat}.log"
    benchmark:
        "benchmarks/single_lat/{dry_moist}/albedo_{albedo}/"
        "solar_lon_{day_of_year}/lat_{lat}.benchmark.txt"
    run:
        from rce_climlab import create_and_run_rce_model

        if wildcards['dry_moist'] == 'dry':
            dry_atmos = True
        elif wildcards['dry_moist'] = 'moist':
            dry_atmos = False
        else:
            raise ValueError("Dry/moist flag must be either 'dry' or 'moist'. "
                             "Value given: {}".format(wildcards['dry_moist'])

        create_and_run_rce_model(
            float(wildcards['lat']),
            day_type=config['model']['day_type'],
            day_of_year=float(wildcards['day_of_year']),
            insol_avg_window: config['model']['insol_avg_window']
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
        expand("tmp/{{dry_moist}}/albedo_{{albedo}}/solar_lon_{{day_of_year}}"
               "/lat_{lat}.nc", lat=LATS)
    output:
        protected("out/{dry_moist}/albedo_{albedo}/solar_lon_{day_of_year}.nc")
    log:
        "logs/mult_lat/{dry_moist}/albedo_{albedo}/solar_lon_{day_of_year}.log"
    benchmark:
        "benchmarks/mult_lat/{dry_moist}/albedo_{albedo}"
        "/solar_lon_{day_of_year}.benchmark.txt"
    run:
        import xarray as xr
        ds = xr.open_mfdataset(input[:],
                               concat_dim=config['str']['lat_str']).transpose()
        ds.to_netcdf(output[0])

rule rce_ann_cycle:
    input:
        expand("out/{{dry_moist}}/albedo_{{albedo}}/solar_lon_{day}.nc",
               day=DAYS_OF_YEAR)
    output:
        protected("out/{{dry_moist}}/albedo_{albedo}/ann_cycle.nc")
    run:
        import xarray as xr
        ds = xr.open_mfdataset(input[:], concat_dim=config['str']['day_str'])
        ds.to_netcdf(output[0])
