Title of the dataset: SOLETE
SOLETE, a 15-month long holistic dataset including: Meteorology, co-located 
wind and solar PV power from Denmark with various resolutions
_______________________________________________________________
Creators:
Daniel Vázquez Pombo https://orcid.org/0000-0001-5664-9421
_______________________________________________________________
Contributors:
Oliver Gehrke https://orcid.org/0000-0001-8184-6435
Henrik W. Bindner https://orcid.org/0000-0003-1545-6994
_______________________________________________________________
Related publications:
https://doi.org/10.11583/DTU.17040626 -> The SOLETE Platform, codes that will help you using the dataset
https://doi.org/10.1016/j.dib.2022.108046 -> paper describing the dataset [1]
https://doi.org/10.3390/s22030749 -> paper using the dataset [2]
https://doi.org/10.1016/j.egyr.2022.05.006 -> paper using the dataset [3]
https://doi.org/10.1016/j.segan.2022.100943 -> paper using the dataset [4]
All of them are open access so you should be able to get them easily.
_______________________________________________________________
Description:
SOLETE includes 15 months of measurements at different sampling rates (1sec, 1min, 5 min, 60min)
from the 1st June 2018 to 1st September 2019 covering: Timestamp, air temperature, relative humidity,
pressure, wind speed, wind direction, global horizontal irradiance, plane of array irradiance, 
atmospheric pressure, active power recorded from an 11 kW Gaia wind turbine and a 10 kW PV inverter, 
Sun's azimuth and elevation angles.

The origin of the data is SYSLAB, part of the Wind and Energy Systems department at the Technical University
of Denmark (DTU). If you want to learn more about the dataset, you should check out [1].

The SOLETE dataset was originally disclosed in [1] to increase the transparency and replicability
of [2], [3], and [4]. Yet, SOLETE's enormous potential to facilitate derived research motivated
the development of the SOLETE platform (see Related publications above). Said platform started
as a few Python scripts allowing to import and plot pieces of the dataset. But nowadays is a 
comprehensive platform facilitating training of physics-informed machine learning models for time-
series forecasting.
_______________________________________________________________
Keywords:
#WindPower #SolarPower #PVpower #Meteorology #Temperature #WindDirection #WindSpeed
#Humidity #Irradiance #Gaia #HybridPowerPlant #ColocatedWindSolar #HybridPowerSystem
_______________________________________________________________
Spatial coverage:
Latitude: 55.6867, Longitude: 12.0985
More details available in [1].
_______________________________________________________________
Temporal coverage:
From 1st June 2018 to 1st September 2019 at different sampling rates.
_______________________________________________________________
This dataset contains the following files:
README.txt -> this description
SOLETE_Pombo_1sec.h5 -> full SOLETE dataset with the original 1 sec sampling
SOLETE_Pombo_1min.h5 -> full SOLETE dataset with a processed sampling of 1 min
SOLETE_Pombo_5min.h5 -> full SOLETE dataset with a processed sampling of 5 min 
SOLETE_Pombo_60min.h5 -> full SOLETE dataset with a processed sampling of 60 min
SOLETE_short.h5 -> preliminary version with only a few samples used for the review process of [1]
_______________________________________________________________
Explanation of variables:
'TEMPERATURE[degC]': ambient temperature in Celsius 
'HUMIDITY[%]': ambient relative humidity in %
'WIND_SPEED[m1s]': wind speed in m/s
'WIND_DIR[deg]': wind direction in deg
'GHI[kW1m2]': global horizontal irradiance in kW/m2
'POA Irr[kW1m2]': plane of array irradiance for the PVs in kW/m2
'P_Gaia[kW]': power output from a 11 kW Gaia turbine
'P_Solar[kW]': power output from a 10 kW PV array
'Pressure[mbar]': atmospheric pressure in mbar
'Azimuth[deg]': Sun's azimuth angle in deg
'Elevation[deg]': Sun's elevation angle in deg
_______________________________________________________________
How to cite:
Pombo, Daniel Vazquez (2022): The SOLETE dataset. Technical University of Denmark. Dataset. https://doi.org/10.11583/DTU.17040767

-version 3
@misc{Pombo2023SOLETE,
author = "Daniel Vazquez Pombo",
title = "{The SOLETE dataset v3}", 
year = "2023",
month = "Mar",
url = "https://doi.org/10.11583/DTU.17040767",
doi = "10.11583/DTU.17040767",
note = {Retrieved from: \url{https://doi.org/10.11583/DTU.17040767 }, {DOI}: {https://doi.org/10.11583/DTU.17040767}},
}

-version 2
@misc{Pombo2022SOLETE,
author = "Daniel Vazquez Pombo",
title = "{The SOLETE dataset v2}", 
year = "2022",
month = "Feb",
url = "https://doi.org/10.11583/DTU.17040767",
doi = "10.11583/DTU.17040767",
note = {Retrieved from: \url{https://doi.org/10.11583/DTU.17040767}, {DOI}: {https://doi.org/10.11583/DTU.17040767}},
}

You can also refer to [1]
_______________________________________________________________
References:
[1] Pombo, D. V., Gehrke, O., & Bindner, H. W. (2022). SOLETE, a 15-month  
    long holistic dataset including: Meteorology, co-located wind and solar PV
    power from Denmark with various resolutions. Data in Brief, 42, 108046.
        
[2] Pombo, D. V., Bindner, H. W., Spataru, S. V., Sørensen, P. E., & Bacher, P. 
    (2022). Increasing the accuracy of hourly multi-output solar power forecast  
    with physics-informed machine learning. Sensors, 22(3), 749.
    
[3] Pombo, D. V., Bacher, P., Ziras, C., Bindner, H. W., Spataru, S. V., & 
    Sørensen, P. E. (2022). Benchmarking physics-informed machine learning-based 
    short term PV-power forecasting tools. Energy Reports, 8, 6512-6520.
    
[4] Pombo, D. V., Rincón, M. J., Bacher, P., Bindner, H. W., Spataru, S. V., 
    & Sørensen, P. E. (2022). Assessing stacked physics-informed machine learning 
    models for co-located wind–solar power forecasting. Sustainable Energy, Grids 
    and Networks, 32, 100943.
_______________________________________________________________
This dataset is published under the CC BY 4.0 license. https://creativecommons.org/licenses/by/4.0/
This license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator.

