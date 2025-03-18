# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 22:21:52 2025

@author: Asus
"""

import pandas as pd


def extract(filename, countries):
    """
    Reads a csv file, cleans it, and returns 2 dataframes for specific countries in a list
    
    Parameters:
    - filename: str, name of the CSV file
    - countries: list
    
    Returns:
    - df_years: Cleaned DataFrame with wide format (years as columns)
    - df_countries: DataFrame with long format (years as rows)
    - Summary statistics
    """
    df = pd.read_csv(filename, skiprows=4)  


    df_clean = df.drop(columns=["Country Code", "Indicator Code", "Indicator Name"], errors="ignore")
    df_clean = df_clean.rename(columns={"Country Name": "Country"})   
    df_clean = df_clean[df_clean["Country"].isin(countries)]
    df_clean = df_clean.dropna(axis=1, how="all")


    df_melted = df_clean.melt(id_vars=["Country"], var_name="Year", value_name="Value").dropna()
    df_melted["Year"] = pd.to_numeric(df_melted["Year"], errors="coerce").astype("Int64")
    df_countries = df_melted.pivot(index="Year", columns="Country", values="Value")
    
    summary_stats = df_countries.describe()

    return df_clean, df_countries, summary_stats

#%%
rich = ["Canada", "France"]
middle = ["Brazil", "Argentina"]
poor = ["Mali", "Korea, Dem. People's Rep."]

poorpop = extract("urbanpopulation.csv", poor)
poorco2 = extract ("co2emissionsfromwaste.csv", poor)
richpop = extract("urbanpopulation.csv", rich)
richco2 = extract ("co2emissionsfromwaste.csv", rich)
middlepop = extract("urbanpopulation.csv", middle)
middleco2 = extract ("co2emissionsfromwaste.csv", middle)

#%%





























