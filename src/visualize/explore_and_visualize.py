import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import urllib.request 

from src.config.paths import (
    HOUSING_CSV,
    CALIFORNIA_IMAGE
)

def main():
    housing_full = pd.read_csv(HOUSING_CSV)
    
    # stratfied sampling
    housing_full["income_cat"] = pd.cut(
        housing_full["median_income"],
        bins=[0., 1.5, 3., 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    strat_train_set, strat_test_set = train_test_split(
        housing_full, 
        test_size=0.2, 
        stratify=housing_full["income_cat"], 
        random_state=42
    )
   
    #print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

    for set__ in (strat_train_set, strat_test_set):
        set__.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()

    housing.plot(
        kind="scatter", x="longitude", y="latitude", grid=True, 
        s=housing["population"] / 100, label = "population",
        c="median_house_value", cmap="jet", colorbar=True,
        legend=True, sharex=False, figsize=(10,7)
    )
    plt.show()

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()

    housing.plot(
        kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True
    )
    plt.show()

    if not CALIFORNIA_IMAGE.is_file():
        homlp_root = "https://github.com/ageron/handson-mlp/raw/main/"
        url = homlp_root + "images/end_to_end_project/california.png"
        print("downloading image")
        urllib.request.urlretrieve(url, CALIFORNIA_IMAGE)

    housing_renamed = housing.rename(columns={
        "latitude": "Latitude", "longitude": "Longitude",
        "population": "Population", 
        "median_house_value": "Median house value (usd)"
    })
    housing_renamed.plot(
        kind="scatter", x="Longitude", y="Latitude",
        s=housing_renamed["Population"] / 100, label="Population",
        c="Median house value (usd)", cmap="jet", colorbar=True,
        legend=True, sharex=False, figsize=(10, 7)
    )

    california_img = plt.imread(CALIFORNIA_IMAGE)
    axis = -124.55, -113.95, 32.45, 42.05
    plt.axis(axis)
    plt.imshow(california_img, extent=axis)

    plt.show()

if __name__ == "__main__":
    main()