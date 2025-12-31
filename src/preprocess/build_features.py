from pathlib import Path
from zlib import crc32
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

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

    # housing_with_id = housing_full.reset_index()
    # housing_with_id["id"] = (housing_full["longitude"] * 1000 + housing_full["latitude"])
    #rng = np.random.default_rng()
    # train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
    # print(len(train_set))
    # print(len(test_set))

    # similar to random sampling, uses a seed as starting point
    # train_set, test_set = train_test_split(housing_full, test_size=0.2, random_state=42)
    # print(len(train_set))
    # print(len(test_set))
    # print(test_set["total_bedrooms"].isnull().sum())

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
    
    housing = strat_train_set.drop("median_house_value", axis = 1)
    housing_labels = strat_train_set["median_house_value"]

    null_rows_idx = housing.isnull().any(axis=1)
    ### Impute
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.select_dtypes(include=[np.number])
    imputer.fit(housing_num)

    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    print(housing_tr.loc[null_rows_idx].head())

    ## Encode
    housing_cat = housing[["ocean_proximity"]]
    print(housing_cat.head(8))
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print(housing_cat_encoded[:8])
    print(ordinal_encoder.categories_)

if __name__ == "__main__":
    main()