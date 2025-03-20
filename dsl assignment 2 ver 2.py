# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 22:21:52 2025

@author: Asus
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stats import *

def extract(filename, country):
    """
    Reads a csv file, cleans it, and returns 2 dataframes for a specific country and the summary statistics
    
    Parameters:
        - filename: str, name of the CSV file
        - country: str, name of the country
        
    Returns:
        - df_years: Cleaned DataFrame with wide format (years as columns)
        - df_country: DataFrame with long format (years as rows)
        - summary_stats: Uses .describe() to explore the data

    """
    df = pd.read_csv(filename, skiprows=4)  

    df_clean = df.drop(columns=["Country Code", "Indicator Code", "Indicator Name"], errors="ignore")
    df_clean = df_clean.rename(columns={"Country Name": "Country"})   
    df_clean = df_clean[df_clean["Country"] == country]
    df_years = df_clean.dropna(axis=1, how="all") 

    df_melted = df_clean.melt(id_vars=["Country"], var_name="Year", value_name="Value").dropna()
    df_melted["Year"] = pd.to_numeric(df_melted["Year"], errors="coerce").astype("Int64")
    
    df_country = df_melted.pivot(index="Year", columns="Country", values="Value")
    
    summary_stats = df_country.describe()

    return df_years, df_country, summary_stats


def analyze_country(co2_file, pop_file, country):
    """
    Reads two csv files for a specific country and returns correlations for the first and recent 20 years, and the average rate of change
        
    Returns:
        - first_corr: correlation coefficient for the first 20 years
        - recent_corr: correlation coefficient for the recent 20 years
        - first_rate: the average rate of change for the first 20 years
        - recent_rate: the aerage rate of change for the recent 20 years

    """
    _, co2,_ = extract(co2_file, country)
    _, pop,_ = extract(pop_file, country)
    
    pop_filtered = pop[pop.index >= 1970]
    df = pd.DataFrame({
        "CO2_Emissions": co2[country],
        "Urban_Population": pop_filtered[country]
    }).dropna()
    
    first_20 = df.loc[df.index[:20]]
    recent_20 = df.loc[df.index[-20:]]
    
    first_corr = first_20["CO2_Emissions"].corr(first_20["Urban_Population"])
    recent_corr = recent_20["CO2_Emissions"].corr(recent_20["Urban_Population"])
    
    first_rate = (first_20["CO2_Emissions"].iloc[-1] - first_20["CO2_Emissions"].iloc[0]) / 20
    recent_rate = (recent_20["CO2_Emissions"].iloc[-1] - recent_20["CO2_Emissions"].iloc[0]) / 20
    
    return first_corr, recent_corr, first_rate, recent_rate



#%%
richco2 = extract ("co2emissionsfromwaste.csv", "Australia")
richpop = extract("urbanpopulation.csv", "Australia")
middleco2 = extract ("co2emissionsfromwaste.csv", "Brazil")
middlepop = extract ("urbanpopulation.csv", "Brazil")
poorco2 = extract ("co2emissionsfromwaste.csv", "Mali")
poorpop = extract ("urbanpopulation.csv", "Mali")

print (richco2, richpop, middleco2, middlepop, poorco2, poorpop)

#High income country: australia
_, australiaco2,_ = extract("co2emissionsfromwaste.csv", "Australia")
_, australiapop,_ = extract("urbanpopulation.csv", "Australia")
australiapop_filtered = australiapop[australiapop.index >= 1970]
australia = pd.DataFrame({
    "CO2_Emissions": australiaco2["Australia"],
    "Urban_Population": australiapop_filtered["Australia"]
})
australia = australia.dropna()
australiacorr = australia["CO2_Emissions"].corr(australia["Urban_Population"])

#Middle income country 1: Brazil
_, brazilco2,_ = extract("co2emissionsfromwaste.csv", "Brazil")
_, brazilpop,_ = extract("urbanpopulation.csv", "Brazil")
brazilpop_filtered = brazilpop[brazilpop.index >= 1970]
brazil = pd.DataFrame({
    "CO2_Emissions": brazilco2["Brazil"],
    "Urban_Population": brazilpop_filtered["Brazil"]
})
brazil = brazil.dropna()
brazilcorr = brazil["CO2_Emissions"].corr(brazil["Urban_Population"])

#Lower income country: Mali
_, malico2,_ = extract("co2emissionsfromwaste.csv", "Mali")
_, malipop,_ = extract("urbanpopulation.csv", "Mali")
malipop_filtered = malipop[malipop.index >= 1970]
mali = pd.DataFrame({
    "CO2_Emissions": malico2["Mali"],
    "Urban_Population": malipop_filtered["Mali"]
})
mali = mali.dropna()
malicorr = mali["CO2_Emissions"].corr(mali["Urban_Population"])

#%%

australia["Country"] = "Australia"
brazil["Country"] = "Brazil"
mali["Country"] = "Mali"

combined_df = pd.concat([australia, brazil, mali], ignore_index=True)

summary_stats = combined_df.groupby("Country").agg({
    "CO2_Emissions": ["mean", "median", "std"],
    "Urban_Population": ["mean", "median", "std"]
})

print("Summary Statistics by Country:", summary_stats)

#%%
australia_results = analyze_country("co2emissionsfromwaste.csv", "urbanpopulation.csv", "Australia")
brazil_results = analyze_country("co2emissionsfromwaste.csv", "urbanpopulation.csv", "Brazil")
mali_results = analyze_country("co2emissionsfromwaste.csv", "urbanpopulation.csv", "Mali")

for country, results in zip(["Australia", "Brazil", "Mali"], [australia_results, brazil_results, mali_results]):
    print(f"{country}:")
    print(f"  First 20-year correlation: {results[0]:.4f}")
    print(f"  Recent 20-year correlation: {results[1]:.4f}")
    print(f"  First 20-year average rate of change: {results[2]:.4f}")
    print(f"  Recent 20-year average rate of change: {results[3]:.4f}\n")
    
#%%
for country, data in zip(["Australia", "Brazil", "Mali"], [australia, brazil, mali]):
    print(f"{country}:")
    print(f"  CO2 Emissions Skewness: {skew(data['CO2_Emissions']):.4f}")
    print(f"  CO2 Emissions Kurtosis: {kurtosis(data['CO2_Emissions']):.4f}")
    print()

#%%
countries = ["Australia", "Brazil", "Mali"]
co2_emissions_data = [australia["CO2_Emissions"], brazil["CO2_Emissions"], mali["CO2_Emissions"]]

means = []
confidence_intervals = []

for data in co2_emissions_data:
    mean = np.mean(data)
    low, high = bootstrap(data, np.mean, confidence_level=0.90)
    means.append(mean)
    confidence_intervals.append([low, high])  

for country, mean, ci in zip(countries, means, confidence_intervals):
    print(f"{country}:")
    print(f"  Mean CO2 Emissions: {mean:.4f}")
    print(f"  90% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print()

#%%
plt.figure(figsize=(10, 6))
plt.bar(countries, means, color=["blue", "green", "purple"])
plt.xlabel("Country")
plt.ylabel("Mean CO2 Emissions")
plt.title("Mean CO2 Emissions From Waste")
plt.show()

#%%
countries = ["Australia", "Brazil", "Mali"]
first_corrs = [australia_results[0], brazil_results[0], mali_results[0]]
recent_corrs = [australia_results[1], brazil_results[1], mali_results[1]]

first_rates = [australia_results[2], brazil_results[2], mali_results[2]]
recent_rates = [australia_results[3], brazil_results[3], mali_results[3]]

x = np.arange(len(countries))
width = 0.35  
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].bar(x - width/2, first_corrs, width, label="First 20 Years")
axes[0].bar(x + width/2, recent_corrs, width, label="Recent 20 Years")
axes[0].set_title("Correlation Between CO2 Emissions From Waste & Urban Population")
axes[0].set_xticks(x)
axes[0].set_xticklabels(countries)
axes[0].set_ylabel("Correlation")
axes[0].set_ylim(min(min(first_corrs), min(recent_corrs)) - 0.1, 1.05)
axes[0].legend()

axes[1].bar(x - width/2, first_rates, width, label="First 20 Years")
axes[1].bar(x + width/2, recent_rates, width, label="Recent 20 Years")
axes[1].set_title("Average Rate of Change in CO2 Emissions From Waste")
axes[1].set_xticks(x)
axes[1].set_xticklabels(countries)
axes[1].set_ylabel("Rate of Change")
axes[1].legend()

plt.show()

#%%
plt.figure(figsize=(10, 6))
plt.boxplot(
    [combined_df[combined_df["Country"] == country]["CO2_Emissions"] for country in ["Australia", "Brazil", "Mali"]],
    labels=["Australia", "Brazil", "Mali"],
    )

plt.xlabel("Country")
plt.ylabel("CO2 Emissions")
plt.title("Distribution of CO2 Emissions From Waste by Country")

plt.show()

#%%

plt.figure()

plt.plot(australia.index, australia["CO2_Emissions"], label="Australia", color="blue")
plt.plot(brazil.index, brazil["CO2_Emissions"], label="Brazil", color="green")
plt.plot(mali.index, mali["CO2_Emissions"], label="Mali", color="purple")

plt.xlabel("Year")
plt.ylabel("CO2 Emissions")
plt.title("CO2 Emissions From Waste Over Time by Country")
plt.legend()

plt.show()



































