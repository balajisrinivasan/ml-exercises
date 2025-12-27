from pathlib import Path
from zlib import crc32
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.config.paths import (
    HOUSING_CSV
)

def shuffle_and_split_data(data, test_ratio, rng):
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2 ** 32;

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def main():
    housing_full = pd.read_csv(HOUSING_CSV)
    housing_with_id = housing_full.reset_index()
    housing_with_id["id"] = (housing_full["longitude"] * 1000 + housing_full["latitude"])

    #rng = np.random.default_rng()
    train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
    print(len(train_set))
    print(len(test_set))

    # similar to random sampling, uses a seed as starting point
    train_set, test_set = train_test_split(housing_full, test_size=0.2, random_state=42)
    print(len(train_set))
    print(len(test_set))
    print(test_set["total_bedrooms"].isnull().sum())

    # stratified sampling based on income category
    housing_full["income_cat"] = pd.cut(
        housing_full["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1,2,3,4,5]
    )
    # cat_counts = housing_full["income_cat"].value_counts().sort_index();
    # cat_counts.plot.bar(rot=0, grid=True)
    # plt.xlabel("Income categroy")
    # plt.ylabel("Number of districts")
    # plt.show()

    strat_train_set, strat_test_set = train_test_split(
        housing_full, test_size=0.2, stratify=housing_full["income_cat"], random_state=42
    )
    print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

    for set__ in (strat_train_set, strat_test_set):
        set__.drop("income_cat", axis=1, inplace=True)

if __name__ == "__main__":
    main()