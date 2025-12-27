from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt

from src.config.paths import (
    RAW_DATA_DIR,
    HOUSING_TARBALL,
    HOUSING_CSV,
)

def load_housing_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not HOUSING_TARBALL.is_file():
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, HOUSING_TARBALL)

        with tarfile.open(HOUSING_TARBALL) as housing_tarball:
            housing_tarball.extractall(
                path=RAW_DATA_DIR, 
                filter="data"
            )
    return pd.read_csv(HOUSING_CSV)

def main():
    housing_full = load_housing_data()
    print(housing_full.head())
    print(housing_full.info())
    print(housing_full["ocean_proximity"].value_counts())
    print(housing_full.describe())

    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    
    housing_full.hist(bins=50, figsize=(12, 8))
    #plt.show()

if __name__ == "__main__":
    main()