
import click
import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def fetch_and_save_ucimlrepo_dataset(dataset_id=2, out_file="data/raw/adult_census_data.csv"):
    """
    Fetch a ucimlrepo dataset by ID, combine features and targets, and save as a CSV.

    Parameters
    ----------
    dataset_id : int
        ucimlrepo dataset ID (Adult/Census Income is 2).
    out_file : str
        Output CSV path. Parent directories are created if needed.

    Returns
    -------
    pandas.DataFrame
        Combined DataFrame (features + target) that was saved to out_file.
    """
    out_dir = "/".join(out_file.split("/")[:-1])
    if out_dir:
        import os

        os.makedirs(out_dir, exist_ok=True)

    ds = fetch_ucirepo(id=dataset_id)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(out_file, index=False)
    return df


@click.command()
@click.option("--out_file", default="data/raw/adult_census_data.csv", show_default=True)
def main(out_file):
    df = fetch_and_save_ucimlrepo_dataset(2, out_file)
    print(f"Saved {df.shape[0]} rows and {df.shape[1]} columns to {out_file}")


if __name__ == "__main__":
    main()
