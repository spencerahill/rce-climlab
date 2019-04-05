configfile: "config.yaml"


import numpy as np


ANN_CYC_SPACING = 360. / config['grid']['num_ann_cycle_points']
DAYS_OF_YEAR = np.arange(0.5*ANN_CYC_SPACING, 360-0.49*ANN_CYC_SPACING,
                         ANN_CYC_SPACING)
DLAT = config['grid']['dlat_deg']
LATS = np.arange(-90+0.5*DLAT, 90-0.49*DLAT, DLAT)


rule rce_single_lat:
    output:
        temp("tmp/solar_lon_{day_of_year}/lat_{lat}.nc")
    params:
        lat="{lat}",
        day_of_year="{day_of_year}"
    log:
        "logs/single_lat/solar_lon_{day_of_year}/lat_{lat}.log"
    benchmark:
        "benchmarks/single_lat/solar_lon_{day_of_year}/lat_{lat}.benchmark.txt"
    run:
        from rce_climlab import create_and_run_rce_model
        create_and_run_rce_model(
            float(params['lat']),
            day_type=config['model']['day_type'],
            day_of_year=float(params['day_of_year']),
            dry_atmos=config['model']['dry_atmos'],
            num_vert_levels=config['grid']['num_vert_levels'],
            albedo=config['model']['albedo'],
            lapse_rate=config['model']['lapse_rate'],
            dt_in_days=config['grid']['dtime_days'],
            num_days_run=config['grid']['num_days'],
            temp_min_valid=config['model']['temp_min_valid'],
            temp_max_valid=config['model']['temp_max_valid'],
            write_to_disk=True,
            path_output=output[0],
        )

rule rce_mult_lats:
    input:
        expand("tmp/solar_lon_{{day_of_year}}/lat_{lat}.nc", lat=LATS)
    output:
        protected("out/solar_lon_{day_of_year}.nc")
    log:
        "logs/mult_lat/solar_lon_{day_of_year}.log"
    benchmark:
        "benchmarks/mult_lat/solar_lon_{day_of_year}.benchmark.txt"
    run:
        import xarray as xr
        ds = xr.open_mfdataset(input[:],
                               concat_dim=config['str']['lat_str']).transpose()
        ds.to_netcdf(output[0])


rule rce_ann_cycle:
    input:
        expand("out/solar_lon_{day}.nc", day=DAYS_OF_YEAR)
    output:
        protected("out/ann_cycle.nc")
    run:
        import xarray as xr
        ds = xr.open_mfdataset(input[:], concat_dim=config['str']['day_str'])
        ds.to_netcdf(output[0])
