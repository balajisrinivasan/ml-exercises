from pathlib import Path
import numpy as np
import pandas as pd

def shuffle_and_split_data(data, test_ratio, rng):
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def main():
    housing_full = pd.read_csv(Path("datasets/housing/housing.csv"))
    rng = np.random.default_rng()
    train_set, test_set = shuffle_and_split_data(housing_full, 0.2, rng)
    print(len(train_set))
    print(len(test_set))

if __name__ == "__main__":
    main()