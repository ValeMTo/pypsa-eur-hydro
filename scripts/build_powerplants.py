# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


"""
Retrieves conventional powerplant capacities and locations from
`powerplantmatching <https://github.com/PyPSA/powerplantmatching>`_, assigns
these to buses and creates a ``.csv`` file. It is possible to amend the
powerplant database with custom entries provided in
``data/custom_powerplants.csv``.
Lastly, for every substation, powerplants with zero-initial capacity can be added for certain fuel types automatically.

Outputs
-------

- ``resource/powerplants_s_{clusters}.csv``: A list of conventional power plants (i.e. neither wind nor solar) with fields for name, fuel type, technology, country, capacity in MW, duration, commissioning year, retrofit year, latitude, longitude, and dam information as documented in the `powerplantmatching README <https://github.com/PyPSA/powerplantmatching/blob/master/README.md>`_; additionally it includes information on the closest substation/bus in ``networks/base_s_{clusters}.nc``.

    .. image:: img/powerplantmatching.png
        :scale: 30 %

    **Source:** `powerplantmatching on GitHub <https://github.com/PyPSA/powerplantmatching>`_

Description
-----------

The configuration options ``electricity: powerplants_filter`` and ``electricity: custom_powerplants`` can be used to control whether data should be retrieved from the original powerplants database or from custom amendments. These specify `pandas.query <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_ commands.
In addition the configuration option ``electricity: everywhere_powerplants`` can be used to place powerplants with zero-initial capacity of certain fuel types at all substations.

1. Adding all powerplants from custom:

    .. code:: yaml

        powerplants_filter: false
        custom_powerplants: true

2. Replacing powerplants in e.g. Germany by custom data:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: true

    or

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: Country in ['Germany']


3. Adding additional built year constraints:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany'] and YearCommissioned <= 2015
        custom_powerplants: YearCommissioned <= 2015

4. Adding powerplants at all substations for 4 conventional carrier types:

    .. code:: yaml

        everywhere_powerplants: ['Natural Gas', 'Coal', 'nuclear', 'OCGT']
"""

import itertools
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import powerplantmatching as pm
import pypsa
from shapely.geometry import MultiPolygon, Polygon

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def add_custom_powerplants(ppl, custom_powerplants, custom_ppl_query=False):
    if not custom_ppl_query:
        return ppl
    add_ppls = pd.read_csv(custom_powerplants, dtype={"bus": "str"})
    add_ppls["source_dataset"] = "custom"
    add_ppls["source_id"] = add_ppls.index.astype(str)
    if isinstance(custom_ppl_query, str):
        add_ppls.query(custom_ppl_query, inplace=True)
    return pd.concat(
        [ppl, add_ppls], sort=False, ignore_index=True, verify_integrity=True
    )


def add_everywhere_powerplants(ppl, substations, everywhere_powerplants):
    # Create a dataframe with "everywhere_powerplants" of stated carriers at the location of all substations
    everywhere_ppl = (
        pd.DataFrame(
            itertools.product(substations.index.values, everywhere_powerplants),
            columns=["substation_index", "Fueltype"],
        ).merge(
            substations[["x", "y", "country"]],
            left_on="substation_index",
            right_index=True,
        )
    ).drop(columns="substation_index")

    # PPL uses different columns names compared to substations dataframe -> rename
    everywhere_ppl = everywhere_ppl.rename(
        columns={"x": "lon", "y": "lat", "country": "Country"}
    )

    # Add default values for the powerplants
    everywhere_ppl["Name"] = (
        "Automatically added everywhere-powerplant " + everywhere_ppl.Fueltype
    )
    everywhere_ppl["Set"] = "PP"
    everywhere_ppl["Technology"] = everywhere_ppl["Fueltype"]
    everywhere_ppl["Capacity"] = 0.0

    # Assign plausible values for the commissioning and decommissioning years
    # required for multi-year models
    everywhere_ppl["DateIn"] = ppl["DateIn"].min()
    everywhere_ppl["DateOut"] = ppl["DateOut"].max()

    # NaN values for efficiency will be replaced by the generic efficiency by attach_conventional_generators(...) in add_electricity.py later
    everywhere_ppl["Efficiency"] = np.nan
    
    everywhere_ppl["source_dataset"] = "everywhere"
    everywhere_ppl["source_id"] = everywhere_ppl.index.astype(str)

    return pd.concat(
        [ppl, everywhere_ppl], sort=False, ignore_index=True, verify_integrity=True
    )


def replace_natural_gas_technology(df):
    mapping = {
        "Steam Turbine": "CCGT",
        "Combustion Engine": "OCGT",
        "Not Found": "CCGT",
    }
    tech = df.Technology.replace(mapping).fillna("CCGT")
    return df.Technology.mask(df.Fueltype == "Natural Gas", tech)


def replace_natural_gas_fueltype(df: pd.DataFrame) -> pd.Series:
    return df.Fueltype.mask(
        (df.Technology == "OCGT") | (df.Technology == "CCGT"), "Natural Gas"
    )


def fill_unoccupied_holes(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    def _fill_poly(poly, idx):
        if not poly.interiors:
            return poly
        kept = [h for h in poly.interiors if gdf.drop(idx).intersects(Polygon(h)).any()]
        return Polygon(poly.exterior, kept)

    result = gdf.geometry.copy()
    for idx in gdf.index:
        g = gdf.geometry[idx]
        if g.geom_type == "Polygon":
            result[idx] = _fill_poly(g, idx)
        elif g.geom_type == "MultiPolygon":
            result[idx] = MultiPolygon([_fill_poly(p, idx) for p in g.geoms])
    return result


def map_to_country_bus(
    ppl: gpd.GeoDataFrame, regions: gpd.GeoDataFrame, max_distance: float = 10000
) -> gpd.GeoDataFrame:
    """
    Assign power plants to region buses of the same country.

    First, spatial join is performed per country to avoid cross-border
    misassignment. Remaining unmatched plants are assigned via nearest
    neighbor (max 10000m) within the same country.
    """
    assigned = []
    unmatched = []

    for country, plants in ppl.groupby("Country"):
        country_regions = regions[regions.index.str[:2] == country]
        joined = (
            plants.sjoin(country_regions)
            .rename(columns={"name": "bus"})
            .reindex(plants.index)
        )
        assigned.append(joined.dropna(subset=["bus"]))
        missing = joined[joined["bus"].isna()]
        if not missing.empty:
            unmatched.append(plants.loc[missing.index])

    if unmatched:
        unmatched = pd.concat(unmatched)
        for country, plants in unmatched.groupby("Country"):
            country_regions = regions[regions.index.str[:2] == country]
            nearest = (
                plants.to_crs(3035)
                .sjoin_nearest(country_regions.to_crs(3035), max_distance=max_distance)
                .rename(columns={"name": "bus"})
                .to_crs(4326)
            )
            missing = plants.index.difference(nearest.index)
            print(country, missing)
            nearest = pd.concat([nearest, plants.loc[missing]])
            assigned.append(nearest)

    return pd.concat(assigned)


def load_and_prepare_glohydrores(input_path: str, ppl: pd.DataFrame) -> pd.DataFrame:
    #Load GloHydroRes and convert it directly to the same column structure as ppl
    df = pd.read_csv(input_path)

    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()

    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.strip()

    if "plant_type" in df.columns:
        df["plant_type"] = df["plant_type"].astype(str).str.strip()

    out = pd.DataFrame(index=df.index)

    technology_mapping = {
        "ROR": "Run-Of-River",
        "STO": "Reservoir",
        "PS": "Pumped Storage",
        "Canal": "Run-Of-River",
    }

    out["Name"] = df["name"]
    out["Fueltype"] = "Hydro"
    out["Technology"] = df["plant_type"].map(technology_mapping)
    out["Technology"] = out["Technology"].replace({"": np.nan, "nan": np.nan}).fillna("Run-Of-River")

    out["Set"] = np.where(df["plant_type"] == "PS", "Store", "PP")
    out["Country"] = df["country"]

    out["Capacity"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    out["Efficiency"] = np.nan
    out["DateIn"] = pd.to_numeric(df["year"], errors="coerce")
    out["DateRetrofit"] = pd.to_numeric(df["year"], errors="coerce")
    out["DateOut"] = pd.to_numeric(df["year"], errors="coerce") + 150

    out["lat"] = pd.to_numeric(df["plant_lat"], errors="coerce")
    out["lon"] = pd.to_numeric(df["plant_lon"], errors="coerce")

    out["Duration"] = np.nan
    out["Volume_Mm3"] = pd.to_numeric(df["res_vol_km3"], errors="coerce") * 1000.0
    out["DamHeight_m"] = pd.to_numeric(df["head_m"], errors="coerce")
    out["StorageCapacity_MWh"] = np.nan

    out["EIC"] = ""
    out["projectID"] = df["plant_source_id"]
    out["bus"] = ""

    out["source_dataset"] = "glohydrores"
    out["source_id"] = (df.index + 1).astype(str) #out["source_id"] = df["ID"].astype(str).str.strip()

    for col in ppl.columns:
        if col not in out.columns:
            out[col] = np.nan

    out = out.reindex(columns=ppl.columns)

    return out


def convert_glohydrores_country_to_iso2(df: pd.DataFrame) -> pd.DataFrame:
    #Convert GloHydroRes country names to ISO-2 country codes. move to ppmatching

    out = df.copy()
    out["Country"] = out["Country"].astype(str).str.strip()

    country_to_code = {
        "Albania": "AL",
        "Austria": "AT",
        "Bosnia and Herzegovina": "BA",
        "Belgium": "BE",
        "Bulgaria": "BG",
        "Switzerland": "CH",
        "Czech Republic": "CZ",
        "Germany": "DE",
        "Denmark": "DK",
        "Estonia": "EE",
        "Spain": "ES",
        "Finland": "FI",
        "France": "FR",
        "United Kingdom": "GB",
        "Greece": "GR",
        "Croatia": "HR",
        "Hungary": "HU",
        "Ireland": "IE",
        "Italy": "IT",
        "Lithuania": "LT",
        "Luxembourg": "LU",
        "Latvia": "LV",
        "Montenegro": "ME",
        "North Macedonia": "MK",
        "Netherlands": "NL",
        "Norway": "NO",
        "Poland": "PL",
        "Portugal": "PT",
        "Romania": "RO",
        "Serbia": "RS",
        "Sweden": "SE",
        "Slovenia": "SI",
        "Slovakia": "SK",
        "Kosovo": "XK",
    }

    out["Country"] = out["Country"].map(country_to_code)

    missing_after = out["Country"].isna().sum()
    logger.info("Plants with missing ISO-2 country after conversion: %s", missing_after)

    return out


def filter_to_pypsa_countries(df: pd.DataFrame, countries: list) -> pd.DataFrame:
    """
    Filter hydro plants to the countries used in PyPSA.
    """

    out = df.copy()
    out["Country"] = out["Country"].astype(str).str.strip().str.upper()
    countries = [c.upper() for c in countries]

    out = out[out["Country"].isin(countries)].copy()

    logger.info("Total plants before filtering: %s", len(df))
    logger.info("Total plants after filtering: %s", len(out))

    return out

def apply_manual_glohydrores_corrections(df: pd.DataFrame) -> pd.DataFrame:
    #duplicated plant and wrong dam height value for Dossi plant
    out = df.copy()

    name = out["Name"].astype(str).str.lower()

    dossi_mask = (
        (out["Country"] == "IT")
        & name.str.contains("dossi", na=False)
    )

    out.loc[dossi_mask, "DamHeight_m"] = 1000.0

    remove_by_name_mask = (
        (out["Country"] == "CH")
        & (
            name.str.contains("innertkirchen 2", na=False)
            | name.str.contains("albignawerk löbbia", na=False)
        )
    )

    remove_tierfehd_duplicate_mask = (
        (out["Country"] == "CH")
        & out["source_id"].astype(str).eq("6121")
    )

    remove_mask = remove_by_name_mask | remove_tierfehd_duplicate_mask

    logger.info("Manual GloHydroRes corrections:")
    logger.info("Dossi DamHeight_m corrected: %s", int(dossi_mask.sum()))
    logger.info("Removed plants:")
    logger.info(
        out.loc[
            remove_mask,
            ["Name", "Country", "Capacity", "Technology", "source_id"],
        ]
    )

    return out.loc[~remove_mask].copy()


def fill_hydro_nan_parameters(
    glohydro_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    #Fill hydro NaN parameters using median values by (Country, Technology).
    
    out = glohydro_df.copy()

    out["Country"] = out["Country"].astype(str).str.strip()
    out["Technology"] = out["Technology"].astype(str).str.strip()

    for col in ["DamHeight_m", "Volume_Mm3", "Capacity", "lat", "lon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    def _fill_column_for_subset(
        df: pd.DataFrame,
        subset_mask: pd.Series,
        target_col: str,
    ) -> pd.DataFrame:
        work = df.copy()

        eligible = work.loc[subset_mask].copy()
        if eligible.empty:
            return work

        country_tech_median = (
            eligible.groupby(["Country", "Technology"])[target_col]
            .median()
        )

        tech_median = (
            eligible.groupby("Technology")[target_col]
            .median()
        )

        global_median = eligible[target_col].median()

        missing_mask = subset_mask & work[target_col].isna()

        for idx in work.index[missing_mask]:
            country = work.at[idx, "Country"]
            tech = work.at[idx, "Technology"]

            value = np.nan

            if (country, tech) in country_tech_median.index:
                value = country_tech_median.loc[(country, tech)]

            if pd.isna(value) and tech in tech_median.index:
                value = tech_median.loc[tech]

            if pd.isna(value):
                value = global_median

            work.at[idx, target_col] = value

        return work

    #ror plants
    ror_mask = out["Technology"] == "Run-Of-River"

    if verbose:
        logger.debug("\n--- ROR ---")
        logger.debug("Missing DamHeight before: %s", out.loc[ror_mask, "DamHeight_m"].isna().sum())

    out = _fill_column_for_subset(
        df=out,
        subset_mask=ror_mask,
        target_col="DamHeight_m",
    )

    if verbose:
        logger.debug("Missing DamHeight after: %s", out.loc[ror_mask, "DamHeight_m"].isna().sum())

    #storage plants
    storage_mask = out["Technology"].isin(["Reservoir", "Pumped Storage"])

    if verbose:
        logger.debug("\n--- STORAGE ---")
        logger.debug("Missing DamHeight before: %s", out.loc[storage_mask, "DamHeight_m"].isna().sum())
        logger.debug("Missing Volume before: %s", out.loc[storage_mask, "Volume_Mm3"].isna().sum())

    out = _fill_column_for_subset(
        df=out,
        subset_mask=storage_mask,
        target_col="DamHeight_m",
    )

    out = _fill_column_for_subset(
        df=out,
        subset_mask=storage_mask,
        target_col="Volume_Mm3",
    )

    if verbose:
        logger.debug("Missing DamHeight after: %s", out.loc[storage_mask, "DamHeight_m"].isna().sum())
        logger.debug("Missing Volume after: %s", out.loc[storage_mask, "Volume_Mm3"].isna().sum())

    return out


def append_glohydro_to_powerplantmatching(
    ppl: pd.DataFrame,
    glohydro_filled: pd.DataFrame,
    ppm_index_start: int = 10000,
    verbose: bool = False,
) -> pd.DataFrame:

    ppl_out = ppl.copy()
    glo_out = glohydro_filled.copy()

    # shift ppl index
    ppl_out.index = pd.RangeIndex(
        start=ppm_index_start,
        stop=ppm_index_start + len(ppl_out),
        step=1,
    )

    # check schema
    if list(glo_out.columns) != list(ppl_out.columns):
        raise ValueError("glohydro_filled columns do not match ppl columns")

    combined = pd.concat([ppl_out, glo_out])

    if combined.index.duplicated().any():
        raise ValueError("Duplicated index after append")

    if verbose:
        logger.info("PPM plants: %s", len(ppl_out))
        logger.info("GloHydro plants: %s", len(glo_out))
        logger.info("Combined plants: %s", len(combined))
        logger.info("PPM index range: %s -> %s", int(ppl_out.index.min()), int(ppl_out.index.max()))
        logger.info("GloHydro index range: %s -> %s", int(glo_out.index.min()), int(glo_out.index.max()))

    return combined

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_powerplants", clusters=256)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.network)
    countries = snakemake.params.countries

    fn_onshore = snakemake.input.regions_onshore
    fn_offshore = snakemake.input.regions_offshore

    regions = pd.concat([gpd.read_file(fn_onshore), gpd.read_file(fn_offshore)])
    regions = regions.dissolve("name")
    regions["geometry"] = fill_unoccupied_holes(regions)

    # Steps copied from PPM: Usually run by PPM when using pm.powerplants(...) from cache
    ppl = (
        pd.read_csv(snakemake.input.powerplants, index_col=0, header=[0])
        .pipe(pm.collection.parse_string_to_dict, ["projectID", "EIC"])
        .pipe(pm.collection.set_column_name, "Matched Data")
    )
    ppl = (
        ppl.powerplant.convert_country_to_alpha2()
        .query("Country in @countries")
        .assign(Technology=replace_natural_gas_technology)
        .assign(Fueltype=replace_natural_gas_fueltype)
        .replace({"Solid Biomass": "Bioenergy", "Biogas": "Bioenergy"})
    )
    ppl_query = snakemake.params.powerplants_filter
    if isinstance(ppl_query, str):
        ppl.query(ppl_query, inplace=True)
    
    ppl["source_dataset"] = "ppm"
    ppl["source_id"] = ppl.index.astype(str)
        
    #GloHydroRes
    glohydro_df = load_and_prepare_glohydrores(snakemake.input.glohydrores_powerplants, ppl)
    glohydro_df = convert_glohydrores_country_to_iso2(glohydro_df)
    glohydro_df = filter_to_pypsa_countries(glohydro_df, countries)
    glohydro_df = apply_manual_glohydrores_corrections(glohydro_df)
    glohydro_filled = fill_hydro_nan_parameters(glohydro_df, verbose=True)
    
    #add glohydro_filled to ppl
    ppl = append_glohydro_to_powerplantmatching(
        ppl=ppl,
        glohydro_filled=glohydro_filled,
        ppm_index_start=10000,
        verbose=True,
    )
    
    ppl = ppl[((ppl["DateOut"] >= 2020) | (ppl["DateOut"].isna()))
    & ((ppl["DateIn"] <= 2019) | (ppl["DateIn"].isna()))
    ].copy()

    # add carriers from own powerplant files:
    custom_ppl_query = snakemake.params.custom_powerplants
    ppl = add_custom_powerplants(
        ppl, snakemake.input.custom_powerplants, custom_ppl_query
    )

    if countries_wo_ppl := set(countries) - set(ppl.Country.unique()):
        logger.warning(f"No powerplants known in: {', '.join(countries_wo_ppl)}")

    # Add "everywhere powerplants" to all bus locations
    ppl = add_everywhere_powerplants(
        ppl, n.buses, snakemake.params.everywhere_powerplants
    )

    ppl = ppl.dropna(subset=["lat", "lon"])

    ppl = gpd.GeoDataFrame(ppl, geometry=gpd.points_from_xy(ppl.lon, ppl.lat), crs=4326)

    ppl = map_to_country_bus(ppl, regions)

    bus_null_b = ppl["bus"].isnull()
    if bus_null_b.any():
        stats = (
            ppl.loc[bus_null_b]
            .groupby(by=["Country", "Fueltype"])
            .Capacity.sum()
            .sort_values(ascending=False)
        )
        logger.warning(
            f"Couldn't assign sufficiently close region for {bus_null_b.sum()} powerplants.\n"
            f"Removing the following capacities (MW) from the powerplants dataset:\n {stats}"
        )
        ppl = ppl[~bus_null_b]

    ppl.reset_index(drop=True).to_csv(snakemake.output[0])
