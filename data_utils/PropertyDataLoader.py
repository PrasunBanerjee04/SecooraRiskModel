import pandas as pd
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.geocoders import ArcGIS
import re 

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
                match = re.search(r'(\d{4})', p)
                year = int(match.group(1)) if match else pd.NA
                
                df = pd.read_csv(p)
                df.columns = df.columns.str.strip()
            except Exception as e:
                print(f"Failed to read {p}: {e}")
                continue
            
            # --- Add year to dataframe ---
            df['Year'] = year

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
                name_val = row.get(name, "")
                type_val = row.get(stype, "")
                if isinstance(name_val, str):
                    name_val = name_val.strip()
                    if isinstance(type_val, str):
                        type_val = type_val.strip()
                    else:
                        type_val = ""
                    streets.append(f"{name_val} {type_val}".strip())
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
            use_cols.append("Year") 
            out = df[use_cols].dropna()
            dfs.append(out)
            
        if not dfs:
            cols = []
            for c in cls.REQUIRED_COLUMNS:
                cols.append(c)
            cols.append("Property Address")
            cols.append("Street Name")
            cols.append("Year") 
            return pd.DataFrame(columns=cols)
            
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.drop_duplicates()
        combined = combined.reset_index(drop=True)
        return combined


    @classmethod
    def order_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        ordered_cols = ["Property Address", "Street Name", "Year"] + cls.REQUIRED_COLUMNS
        final_ordered_cols = [col for col in ordered_cols if col in df.columns]
        return df.loc[:, final_ordered_cols]


    @classmethod
    def add_coordinates(cls, df: pd.DataFrame, num_workers: int = 10) -> pd.DataFrame:
        if df.empty:
            df["Latitude"] = []
            df["Longitude"] = []
            return df
            
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
        
        if not streets:
            df["Latitude"] = []
            df["Longitude"] = []
            return df

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(geocode, s): idx for idx, s in enumerate(streets)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = (None, None)

        if not results:
             df["Latitude"] = []
             df["Longitude"] = []
             return df

        latitude, longitude = zip(*results)
        df["Latitude"] = latitude
        df["Longitude"] = longitude
        return df


    @classmethod
    def load_data_to_csv(cls, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False)


def main():
    paths = [
        "../data/parcels/Tybee-Parcels-2023.csv", 
        "../data/parcels/Tybee-Parcels-2020.csv", 
        "../data/parcels/Tybee-Parcels-2021.csv", 
        "../data/parcels/Tybee-Parcels-2022.csv"
    ]
    
    df = PropertyDataLoader.clean_csv_list(paths)
    if df.empty:
        print("No data was loaded. Please check your file paths.")
        return

    df = PropertyDataLoader.order_df(df)
    df = PropertyDataLoader.add_coordinates(df, num_workers=10)
    
    if 'Street Name' in df.columns:
        df.drop(columns=['Street Name'], inplace=True)
        
    PropertyDataLoader.load_data_to_csv(df, "preprocessed_parcels_data.csv")
    print("Data processing complete. Output saved to 'preprocessed_parcels_data.csv'")
    print(df.head())


if __name__ == "__main__":
    main()