import pandas as pd
path = "/data2/zqr/CAMELS/CAMELS-GB"
new_path = "/data2/zx/dataset/CAMELS-GB/static"
df1 = pd.read_csv(path + "/CAMELS_GB_climatic_attributes.csv")
df2 = pd.read_csv(path + "/CAMELS_GB_humaninfluence_attributes.csv")
df6 = pd.read_csv(path + "/CAMELS_GB_landcover_attributes.csv")
df7 = pd.read_csv(path + "/CAMELS_GB_soil_attributes.csv")
df8 = pd.read_csv(path + "/CAMELS_GB_topographic_attributes.csv")

gauge_id = df1.iloc[:, 0:1]
area = df8.iloc[:, 7:8]
elev_mean = df8.iloc[:, 9:10]
dpsbar = df8.iloc[:, 8:9]
sand_perc = df7.iloc[:, 1:2]
silt_perc = df7.iloc[:, 3:4]
clay_perc = df7.iloc[:, 5:6]
porosity_hypres = df7.iloc[:, 24:25]
conductivity_hypres = df7.iloc[:, 34:35]
soil_depth_pelletier = df7.iloc[:, 44:45]
frac_snow = df1.iloc[:, 5:6]
dwood_perc = df6.iloc[:, 1:2]
ewood_perc = df6.iloc[:, 2:3]
crop_perc = df6.iloc[:, 5:6]
urban_perc = df6.iloc[:, 6:7]
reservoir_cap = df2.iloc[:, 12:13]
p_mean = df1.iloc[:, 1:2]
pet_mean = df1.iloc[:, 2:3]
p_seasonality = df1.iloc[:, 4:5]
high_prec_freq = df1.iloc[:, 6:7]
low_prec_freq = df1.iloc[:, 9:10]
high_prec_dur = df1.iloc[:, 7:8]
low_prec_dur = df1.iloc[:, 10:11]

pd.concat([gauge_id, area, elev_mean, dpsbar, sand_perc, silt_perc, clay_perc,
                       porosity_hypres, conductivity_hypres, soil_depth_pelletier, frac_snow,
                       dwood_perc, ewood_perc, crop_perc, urban_perc, reservoir_cap,
                       p_mean, pet_mean, p_seasonality, high_prec_freq, low_prec_freq,
                       high_prec_dur, low_prec_dur], axis=1).to_csv(new_path + "/static_attributes_gb.csv" , index=False)
