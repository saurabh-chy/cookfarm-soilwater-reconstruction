# Input data

Raw data are not redistributed in this repository. Download them from the original sources, then check the checksums.

## 1. Cook Agronomy Farm sensor network data package
- **Catalog:** https://catalog.data.gov/dataset/data-from-a-field-scale-sensor-network-data-set-for-monitoring-and-modeling-the-spatial-an
- **Direct download** (`CAF_Sensor_Dataset_2.zip`): https://ndownloader.figshare.com/files/44335805
- **Dataset DOI:** https://doi.org/10.15482/USDA.ADC/1349683
- **Unzip to:** `data/raw/CAF/`
- **SHA-256 of the zip:** `b4098c5a5b6499477794f2d0bdb797fb4fa495b80dd8ed97d0befa531e1eb5ff`

## 2. Daymet v4 R1 daily weather at the farm (single pixel)
- **URL** (requires a free NASA Earthdata login if prompted):
  https://daymet.ornl.gov/single-pixel/api/data?lat=46.7815&lon=-117.082&vars=prcp,tmax,tmin,srad,vp,swe,dayl&start=2007-01-01&end=2016-12-31
- **Save as:** `data/raw/daymet_caf_2007_2016.csv`
- **SHA-256 of the file used:** `af31593b9c394a4cb76f14af47d4d60077221a57b228e5eee3c0e5bec733f9f6`

  A fresh download can differ in its header lines (for example the download date), which changes the checksum without changing the data. In that case compare the data rows instead.

## Derived data
`make panel` writes the cleaned panel and covariate tables to `data/processed/`. These files are not committed.
