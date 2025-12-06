import click
import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

@click.command()
@click.option('--out_file', default="data/raw/adult_census_data.csv")
def main(out_file):
    print("Fetching data from ucimlrepo (ID=2)......")
    adult = fetch_ucirepo(id=2) 

    # Data as pandas dataframe
    df = pd.concat([adult.data.features, adult.data.targets], axis=1)
    
    # Save to CSV
    df.to_csv(out_file, index=False)
    print(f"Data saved to {out_file}")

if __name__ == "__main__":
    main()