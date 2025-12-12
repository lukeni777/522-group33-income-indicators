import os
import pandas as pd
import src.read_data as rd


class DummyData:
    def __init__(self):
        self.features = pd.DataFrame({"age": [30, 40]})
        self.targets = pd.DataFrame({"income": ["<=50K", ">50K"]})


class DummyDataset:
    def __init__(self):
        self.data = DummyData()


def test_fetch_and_save_ucimlrepo_dataset(monkeypatch):
    monkeypatch.setattr(rd, "fetch_ucirepo", lambda id: DummyDataset())

    out_file = "data/raw/adult_census_data.csv"
    df = rd.fetch_and_save_ucimlrepo_dataset(dataset_id=2, out_file=out_file)

    assert os.path.exists(out_file)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["age", "income"]

    df_read = pd.read_csv(out_file)
    assert df_read["age"].tolist() == [30, 40]
    assert df_read["income"].tolist() == ["<=50K", ">50K"]
