# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build hydroelectric inflow time-series for each country.

Outputs
-------

- ``resources/profile_hydro.nc``:

    ===================  ================  =========================================================
    Field                Dimensions        Description
    ===================  ================  =========================================================
    inflow               countries, time   Inflow to the state of charge (in MW),
                                           e.g. due to river inflow in hydro reservoir.
    ===================  ================  =========================================================

    .. image:: img/inflow-ts.png
        :scale: 33 %

    .. image:: img/inflow-box.png
        :scale: 33 %
"""

import logging

import country_converter as coco
import geopandas as gpd
import pandas as pd
import xarray as xr
from numpy.polynomial import Polynomial

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    load_cutout,
    set_scenario_config,
)

cc = coco.CountryConverter()


def get_eia_annual_hydro_generation(
    fn: str, countries: list[str], capacities: bool = False
) -> pd.DataFrame:
    # in billion kWh/a = TWh/a
    df = pd.read_csv(fn, skiprows=2, index_col=1, na_values=[" ", "--"]).iloc[1:, 1:]
    df.index = df.index.str.strip()
    df.columns = df.columns.astype(int)

    former_countries = {
        "Former Czechoslovakia": dict(
            countries=["Czechia", "Slovakia"], start=1980, end=1992
        ),
        "Former Serbia and Montenegro": dict(
            countries=["Serbia", "Montenegro", "Kosovo"], start=1992, end=2005
        ),
        "Former Yugoslavia": dict(
            countries=[
                "Slovenia",
                "Croatia",
                "Bosnia and Herzegovina",
                "Serbia",
                "Kosovo",
                "Montenegro",
                "North Macedonia",
            ],
            start=1980,
            end=1991,
        ),
    }

    for k, v in former_countries.items():
        period = [i for i in range(v["start"], v["end"] + 1)]
        ratio = df.loc[v["countries"]].T.dropna().sum()
        ratio /= ratio.sum()
        for country in v["countries"]:
            df.loc[country, period] = df.loc[k, period] * ratio[country]

    baltic_states = ["Latvia", "Estonia", "Lithuania"]
    df.loc[baltic_states] = (
        df.loc[baltic_states].T.fillna(df.loc[baltic_states].mean(axis=1)).T
    )

    df.loc["Germany"] = df.filter(like="Germany", axis=0).sum()
    df = df.loc[~df.index.str.contains("Former")]
    df.drop(["Europe", "Germany, West", "Germany, East"], inplace=True)

    df.index = cc.convert(df.index, to="iso2")
    df.index.name = "countries"

    # convert to MW of MWh/a
    factor = 1e3 if capacities else 1e6
    df = df.T[countries] * factor

    df.ffill(axis=0, inplace=True)

    return df


def correct_eia_stats_by_capacity(
    eia_stats: pd.DataFrame, fn: str, countries: list[str], baseyear: int = 2019
) -> None:
    cap = get_eia_annual_hydro_generation(fn, countries, capacities=True)
    ratio = cap / cap.loc[baseyear]
    eia_stats_corrected = eia_stats / ratio
    to_keep = ["AL", "AT", "CH", "DE", "GB", "NL", "RS", "XK", "RO", "SK"]
    to_correct = eia_stats_corrected.columns.difference(to_keep)
    eia_stats.loc[:, to_correct] = eia_stats_corrected.loc[:, to_correct]


def approximate_missing_eia_stats(
    eia_stats: pd.DataFrame, runoff_fn: str, countries: list[str]
) -> pd.DataFrame:
    runoff = pd.read_csv(runoff_fn, index_col=0, parse_dates=True)[countries]
    runoff = runoff.groupby(runoff.index.year).sum().index

    # fix outliers; exceptional floods in 1977-1979 in ES & PT
    if "ES" in runoff:
        runoff.loc[1978, "ES"] = runoff.loc[1979, "ES"]
    if "PT" in runoff:
        runoff.loc[1978, "PT"] = runoff.loc[1979, "PT"]

    runoff_eia = runoff.loc[eia_stats.index]

    eia_stats_approximated = {}

    for c in countries:
        X = runoff_eia[c]
        Y = eia_stats[c]

        to_predict = runoff.index.difference(eia_stats.index)
        X_pred = runoff.loc[to_predict, c]

        p = Polynomial.fit(X, Y, 1)
        Y_pred = p(X_pred)

        eia_stats_approximated[c] = pd.Series(Y_pred, index=to_predict)

    eia_stats_approximated = pd.DataFrame(eia_stats_approximated)
    return pd.concat([eia_stats, eia_stats_approximated]).sort_index()

def add_qmax_turb(hydro_ppls, efficiency, q_min_factor=0.2):
    
    #function that adds the maximum and minimum turbinable flow rates for each hydro plant

    hydro_ppls["technology"] = hydro_ppls["technology"].fillna("Run-Of-River")

    #Compute country-level median heads
    ror_mask = hydro_ppls["technology"] == "Run-Of-River"
    res_mask = hydro_ppls["technology"].isin(["Reservoir", "Pumped Storage"])

    median_ror_country = (
        hydro_ppls[ror_mask]
        .groupby("country")["damheight_m"]
        .median()
    )

    median_res_country = (
        hydro_ppls[res_mask]
        .groupby("country")["damheight_m"]
        .median()
    )

    # Global fallback medians
    median_ror_global = hydro_ppls.loc[ror_mask, "damheight_m"].median(skipna=True)
    median_res_global = hydro_ppls.loc[res_mask, "damheight_m"].median(skipna=True)

    #Replace NaN heads using country medians for safety
    mask_nan = hydro_ppls["damheight_m"].isna()

    for country in hydro_ppls["country"].unique():

        # ROR replacement
        mask_country_ror = (
            mask_nan &
            (hydro_ppls["country"] == country) &
            ror_mask
        )

        country_median = median_ror_country.get(country, median_ror_global)

        hydro_ppls.loc[mask_country_ror, "damheight_m"] = country_median

        # Reservoir + PHS replacement
        mask_country_res = (
            mask_nan &
            (hydro_ppls["country"] == country) &
            res_mask
        )

        country_median = median_res_country.get(country, median_res_global)

        hydro_ppls.loc[mask_country_res, "damheight_m"] = country_median


    #Replace zero or negative heads with non zero value
    mask_zero_or_neg = hydro_ppls["damheight_m"] <= 0
    hydro_ppls.loc[mask_zero_or_neg, "damheight_m"] = 1.0

    #Compute qmax and qmin
    hydro_ppls["qmax_turb"] = (
        hydro_ppls["p_nom"] /
        (9.81 * hydro_ppls["damheight_m"] * 1e-3 * efficiency)
    )

    hydro_ppls["qmin_turb"] = q_min_factor * hydro_ppls["qmax_turb"]

    return hydro_ppls

def ror_conversion(ror_plants, inflow_m3s):
    
    #function that converts the inflows from ROR plants into relative output (p_max_pu)
    
    plant_ids = ror_plants.index.astype(str).values
    inflow_sel = inflow_m3s.sel(plant=plant_ids).load()

    out = xr.zeros_like(inflow_sel)

    for pid in plant_ids:

        qmax = float(ror_plants.loc[pid, "qmax_turb"])
        qmin = float(ror_plants.loc[pid, "qmin_turb"])

        s = inflow_sel.sel(plant=pid)

        denom = (qmax - qmin) if (qmax - qmin) != 0 else 1.0

        p = xr.where(
            s < qmin,
            0.0,
            xr.where(
                s > qmax,
                1.0,
                (s - qmin) / denom
            )
        )

        out.loc[dict(plant=pid)] = p

    df = out.to_dataframe(name="p_max_pu").reset_index()
    return df.pivot(index="time", columns="plant", values="p_max_pu")

def reservoir_conversion(res_plants, inflow_m3s, eff):
    
    #function that converts the inflows of reservoir plants from flow rate to MW

    plant_ids = res_plants.index.astype(str).values
    inflow_sel = inflow_m3s.sel(plant=plant_ids).load()

    converted = {}

    for pid in plant_ids:

        head = float(res_plants.loc[pid, "damheight_m"])

        q = inflow_sel.sel(plant=pid).values

        converted[pid] = q * 9.81 * head * eff * 1e-3

    return pd.DataFrame(converted, index=inflow_sel.coords["time"].values)

def override_eia_stats_with_entsoe(
    eia_stats: pd.DataFrame,
    fn: str,
    countries: list[str],
) -> pd.DataFrame:
    """
    Override EIA annual hydro generation statistics with processed
    ENTSO-E / Electricity Maps annual country totals.

    Expected input columns:
    - year
    - country
    - hydro_generation_mwh

    Units:
    - hydro_generation_mwh must be annual generation in MWh.
    """

    entsoe = pd.read_csv(fn)

    required_columns = {"year", "country", "hydro_generation_mwh"}
    missing_columns = required_columns.difference(entsoe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {fn}: {sorted(missing_columns)}"
        )

    entsoe = entsoe.copy()
    entsoe["year"] = entsoe["year"].astype(int)
    entsoe["country"] = entsoe["country"].astype(str)

    entsoe = entsoe[entsoe["country"].isin(countries)]

    if entsoe.empty:
        logger.warning(
            "ENTSO-E hydro annual production file is empty after filtering "
            "for configured countries. EIA statistics are left unchanged."
        )
        return eia_stats

    entsoe_wide = entsoe.pivot_table(
        index="year",
        columns="country",
        values="hydro_generation_mwh",
        aggfunc="first",
    )

    for year in entsoe_wide.index:
        if year not in eia_stats.index:
            eia_stats.loc[year, :] = pd.NA

        common_countries = entsoe_wide.columns.intersection(eia_stats.columns)
        eia_stats.loc[year, common_countries] = entsoe_wide.loc[
            year, common_countries
        ]

    eia_stats = eia_stats.sort_index()

    logger.info(
        "Overrode EIA hydro generation statistics with ENTSO-E/Electricity Maps "
        "for years %s.",
        sorted(entsoe_wide.index.tolist()),
    )

    return eia_stats

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_hydro_profile")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params_hydro = snakemake.params.hydro

    time = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)

    cutout = load_cutout(snakemake.input.cutout)

    years_in_time = pd.DatetimeIndex(time).year.unique()
    cutout_time = pd.DatetimeIndex(cutout.coords["time"].values)

    full_years_available = all(
        pd.Timestamp(f"{year}-01-01") in cutout_time
        and pd.Timestamp(f"{year}-12-31") in cutout_time
        for year in years_in_time
    )

    if full_years_available:
        mask = [pd.Timestamp(t).year in years_in_time for t in cutout_time]
        cutout = cutout.sel(time=cutout_time[mask])
    else:
        cutout = cutout.sel(time=time)

    countries = snakemake.params.countries
    country_shapes = (
        gpd.read_file(snakemake.input.country_shapes)
        .set_index("name")["geometry"]
        .reindex(countries)
    )
    country_shapes.index.name = "countries"

    fn = snakemake.input.eia_hydro_generation
    eia_stats = get_eia_annual_hydro_generation(fn, countries)
    
    fn = snakemake.input.eia_hydro_generation
    eia_stats = get_eia_annual_hydro_generation(fn, countries)

    if "entsoe_hydro_annual_production" in snakemake.input:
        eia_stats = override_eia_stats_with_entsoe(
            eia_stats=eia_stats,
            fn=snakemake.input.entsoe_hydro_annual_production,
            countries=countries,
        )

    config_hydro = snakemake.config["renewable"]["hydro"]

    if config_hydro.get("eia_correct_by_capacity"):
        fn = snakemake.input.eia_hydro_capacity
        correct_eia_stats_by_capacity(eia_stats, fn, countries)

    if config_hydro.get("eia_approximate_missing"):
        fn = snakemake.input.era5_runoff
        eia_stats = approximate_missing_eia_stats(eia_stats, fn, countries)

    norm_year = config_hydro.get("eia_norm_year")
    missing_years = years_in_time.difference(eia_stats.index)
    if norm_year:
        eia_stats.loc[years_in_time] = eia_stats.loc[norm_year]
    elif missing_years.any():
        eia_stats.loc[missing_years] = eia_stats.median()

    # Hydro inflow calculation method
    config_hydro = snakemake.config["renewable"]["hydro"]
    GloFAS_ERA5 = config_hydro.get("GloFAS_ERA5", {})
    method = GloFAS_ERA5.get("methods", False)

    if method == "GloFAS":
        eff = GloFAS_ERA5.get("eff", 0.85)
        q_min = GloFAS_ERA5.get("q_min", 0.2)

        # 1️⃣ Load powerplants
        ppls = pd.read_csv(snakemake.input.powerplants, index_col=0)

        ppls.columns = ppls.columns.str.strip().str.lower()
        ppls.index = ppls.index.astype(str)

        hydro_ppls = ppls[ppls["fueltype"].str.lower() == "hydro"].copy()
        hydro_ppls["technology"] = hydro_ppls["technology"].fillna("Run-Of-River")
        hydro_ppls = hydro_ppls.rename(columns={"capacity": "p_nom"})
        hydro_ppls = add_qmax_turb(hydro_ppls, eff, q_min)

        #Load inflow
        logger.info("Opening SABER inflow file: %s", snakemake.input.saber_inflow)
        inflow_saber = xr.open_dataarray(snakemake.input.saber_inflow)

        inflow_saber = inflow_saber.assign_coords(
            plant=inflow_saber.coords["plant"].astype(str)
        )

        if "clip_min_inflow" in params_hydro:
            inflow_saber = inflow_saber.where(
                inflow_saber > params_hydro["clip_min_inflow"], 0
            )

        hydro_ppls["source_id"] = hydro_ppls["source_id"].astype(str).str.strip()

        duplicated_source_id = hydro_ppls["source_id"].duplicated(keep=False)
        if duplicated_source_id.any():
            duplicated = hydro_ppls.loc[
                duplicated_source_id,
                [
                    c for c in [
                        "name",
                        "country",
                        "technology",
                        "p_nom",
                        "source_id",
                        "bus",
                        "lon",
                        "lat",
                    ]
                    if c in hydro_ppls.columns
                ],
            ]

            logger.error(
                "Duplicated hydro source_id values found before SABER inflow matching:\n%s",
                duplicated.to_string(),
            )

            raise ValueError(
                "Duplicated hydro source_id values found before SABER inflow matching."
            )

        hydro_index = pd.Index(hydro_ppls["source_id"].astype(str))
        inflow_index = pd.Index(inflow_saber.coords["plant"].values.astype(str))

        common_index = hydro_index.intersection(inflow_index)

        logger.info("Hydro plants in powerplants: %s", len(hydro_index))
        logger.info("Plants in SABER inflow: %s", len(inflow_index))
        logger.info("Common SABER hydro plants: %s", len(common_index))

        missing_inflow = hydro_index.difference(inflow_index)

        if len(missing_inflow) > 0:
            missing_ppls = hydro_ppls.loc[
                hydro_ppls["source_id"].isin(missing_inflow),
                [
                    c for c in [
                        "name",
                        "country",
                        "technology",
                        "p_nom",
                        "source_id",
                        "bus",
                        "lon",
                        "lat",
                    ]
                    if c in hydro_ppls.columns
                ],
            ]

            logger.error(
                "%s hydro plants have no SABER inflow data.",
                len(missing_inflow),
            )
            logger.error("Missing SABER source_id values: %s", list(missing_inflow))
            logger.error("Missing plants:\n%s", missing_ppls.to_string())

            raise ValueError(
                f"{len(missing_inflow)} hydro plants have no inflow data."
            )

        hydro_matched = (
            hydro_ppls
            .set_index("source_id", drop=False)
            .loc[common_index]
            .copy()
        )

        inflow_matched = inflow_saber.sel(plant=common_index)
        #Split technologies
        ror_plants = hydro_matched[
            hydro_matched["technology"] == "Run-Of-River"
        ]

        reservoir_plants = hydro_matched[
            hydro_matched["technology"].isin(["Reservoir", "Pumped Storage"])
        ]

        inflow_ror = inflow_matched.sel(plant=ror_plants.index)
        inflow_res = inflow_matched.sel(plant=reservoir_plants.index)

        #Convert inflow
        if not ror_plants.empty:
            p_max_pu_ror = ror_conversion(
                ror_plants,
                inflow_ror,
            )
        else:
            p_max_pu_ror = pd.DataFrame()
        
        if not reservoir_plants.empty:
            inflow_res_mw = reservoir_conversion(
                reservoir_plants,
                inflow_res,
                eff=eff
            )
        else:
            inflow_res_mw = pd.DataFrame()

        outputs = {}

        if not p_max_pu_ror.empty:
            outputs["p_max_pu_ror"] = xr.DataArray(
                p_max_pu_ror.values,
                dims=("time", "plant"),
                coords={
                    "time": p_max_pu_ror.index,
                    "plant": p_max_pu_ror.columns.values
                }
            )
        
        if not inflow_res_mw.empty:
            outputs["inflow_reservoir"] = xr.DataArray(
                inflow_res_mw.values,
                dims=("time", "plant"),
                coords={
                    "time": inflow_res_mw.index,
                    "plant": inflow_res_mw.columns.values,
                }
            )

        inflow = xr.Dataset(outputs)

        logger.info("Final dataset dims: %s", inflow.dims)
        
    else:

        #Default PyPSA-Eur behaviour
        inflow = cutout.runoff(
            shapes=country_shapes,
            smooth=True,
            lower_threshold_quantile=True,
            normalize_using_yearly=eia_stats,
        )

        if full_years_available:
            inflow = inflow.sel(time=time)

        if "clip_min_inflow" in params_hydro:
            inflow = inflow.where(inflow > params_hydro["clip_min_inflow"], 0)

    inflow.to_netcdf(snakemake.output.profile)
