import pandas as pd
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.geocoders import ArcGIS

class PropertyDataLoader:
    STREET_NAME = ["PropAddress_StreetName", "PropAddr_2"]
    STREET_TYPE = ["PropAddress_StreetType", "PropAddr_3"]
    ADDRESS_COLUMNS = [
        "PropAddress_Full", "PropAddress_Num", "PropAddress_PreDir",
        "PropAddress_StreetName", "PropAddress_StreetType",
        "PropAddress_PostDir", "PropAddress_UnitType", "PropAddress_UnitNum",
        "PropAddress_City", "PropAddress_State", "PropAddress_Zip",
        "PropAddres", "PropAddr_1", "PropAddr_2", "PropAddr_3",
        "PropAddr_5", "PropAddr_6", "PropAddr_7", "PropAddr_8",
        "PropAddr_9", "PropAddr10"
    ]
    ALTERNATE_NAMES = {
        "FairMarket": "Fair Market Value",
        "FairMarketValue": "Fair Market Value",
        "Sale_Price": "Sale Price",
        "Acres": "Acres",
        "YearBuilt": "Year Built",
        "FMV_Land": "FMV Land",
        "FMV_Building": "FMV Building"
    }
    REQUIRED_COLUMNS = ["Fair Market Value", "Sale Price", "Acres", "Year Built"]


    @classmethod
    def clean_csv_list(cls, paths: List[str]) -> pd.DataFrame:
        dfs = []
        for p in paths:
            try:
                df = pd.read_csv(p)
                df.columns = df.columns.str.strip()
            except:
                continue
            name = None
            for c in cls.STREET_NAME:
                if c in df.columns:
                    name = c
                    break
            stype = None
            for c in cls.STREET_TYPE:
                if c in df.columns:
                    stype = c
                    break
            if not name:
                continue
            streets = []
            for _, row in df.iterrows():
                name = row.get(name, "")
                type = row.get(stype, "")
                if isinstance(name, str):
                    name = name.strip()
                    if isinstance(type, str):
                        type = type.strip()
                    else:
                        type = ""
                    streets.append(f"{name} {type}".strip())
                else:
                    streets.append("")
            df["Street Name"] = streets
            addr_cols = []
            for c in cls.ADDRESS_COLUMNS:
                if c in df.columns:
                    addr_cols.append(c)
            full_addr = []
            for _, row in df.iterrows():
                parts = []
                for c in addr_cols:
                    v = row.get(c)
                    if pd.notnull(v):
                        parts.append(str(v).strip())
                full_addr.append(" ".join(parts))
            df["Property Address"] = full_addr
            for old, new in cls.ALTERNATE_NAMES.items():
                if old in df.columns:
                    df.rename(columns={old: new}, inplace=True)
            found_all = True
            for c in cls.REQUIRED_COLUMNS:
                if c not in df.columns:
                    found_all = False
                    break
            if not found_all:
                continue
            use_cols = []
            for c in cls.REQUIRED_COLUMNS:
                use_cols.append(c)
            use_cols.append("Property Address")
            use_cols.append("Street Name")
            out = df[use_cols].dropna()
            dfs.append(out)
        if not dfs:
            cols = []
            for c in cls.REQUIRED_COLUMNS:
                cols.append(c)
            cols.append("Property Address")
            cols.append("Street Name")
            return pd.DataFrame(columns=cols)
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates()
        combined = combined.reset_index(drop=True)
        return combined


    @classmethod
    def order_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        ordered_cols = ["Property Address", "Street Name"] + cls.REQUIRED_COLUMNS
        return df.loc[:, ordered_cols]


    @classmethod
    def add_coordinates(cls, df: pd.DataFrame, num_workers: int = 10) -> pd.DataFrame:
        cache = {}
        def geocode(street):
            if street in cache:
                return cache[street]
            loc = ArcGIS(timeout=10).geocode(f"{street} TYBEE ISLAND, GA")
            result = (loc.latitude, loc.longitude) if loc else (None, None)
            cache[street] = result
            return result
        streets = df["Street Name"].tolist()
        results = [None] * len(streets)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(geocode, s): idx for idx, s in enumerate(streets)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = (None, None)
        latitude, longitude = zip(*results)
        df["Latitude"] = latitude
        df["Longitude"] = longitude
        return df


    @classmethod
    def load_data_to_csv(cls, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False)


def main():
    paths = ["data\Tybee-Parcels-2023(in).csv", "data\Tybee-Parcels-2020.csv", "data\Tybee-Parcels-2021.csv", "data\Tybee-Parcels-2022.csv"]
    df = PropertyDataLoader.clean_csv_list(paths)
    df = PropertyDataLoader.order_df(df)
    df = PropertyDataLoader.add_coordinates(df, num_workers=10)
    df.drop(columns=['Street Name'], inplace=True)
    PropertyDataLoader.load_data_to_csv(df, "preprocessed_parcels_data.csv")
    print(df.head())


if __name__ == "__main__":
    main()
