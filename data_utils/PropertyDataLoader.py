import pandas as pd
from typing import List

class PropertyDataLoader:
    ADDRESS_COLUMNS = [
        "PropAddress_Full", "PropAddress_Num", "PropAddress_PreDir", "PropAddress_StreetName",
        "PropAddress_StreetType", "PropAddress_PostDir", "PropAddress_UnitType", "PropAddress_UnitNum",
        "PropAddress_City", "PropAddress_State", "PropAddress_Zip",
        "PropAddres", "PropAddr_1", "PropAddr_2", "PropAddr_3", "PropAddr_5",
        "PropAddr_6", "PropAddr_7", "PropAddr_8", "PropAddr_9", "PropAddr10"
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
    def clean_csv_list(cls, csv_paths: List[str]) -> pd.DataFrame:
        all_csv_files = []
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                continue
            df.columns = [col.strip() for col in df.columns]
            address_parts = []
            for idx in range(len(df)):
                parts = []
                for col in cls.ADDRESS_COLUMNS:
                    if col in df.columns and pd.notnull(df.at[idx, col]):
                        parts.append(str(df.at[idx, col]).strip())
                address_parts.append(' '.join(parts))
            df["Property Address"] = address_parts
            for original, standard in cls.ALTERNATE_NAMES.items():
                if original in df.columns:
                    df.rename(columns={original: standard}, inplace=True)
            missing_cols = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                continue
            subset_cols = cls.REQUIRED_COLUMNS + ["Property Address"]
            df_subset = df[subset_cols].copy()

            for idx, row in df_subset.iterrows():
                if row.isnull().any():
                    missing_fields = row.index[row.isnull()].tolist()
            cleaned = df_subset.dropna()
            all_csv_files.append(cleaned)

        combined_dataframe = pd.concat(all_csv_files, ignore_index=True)
        final_dataframe = combined_dataframe.drop_duplicates().reset_index(drop=True)
        return final_dataframe
    
    @classmethod
    def order_df(cls, df):
        #order the columns of the dataframe
        columns = ["Property Address",
                   "Fair Market Value",
                   "Sale Price",
                   "Acres",
                   "Year Built"
                    ]

        df = df[columns]
        return df

    @classmethod
    def load_data_to_csv(cls, df, path):
        #take pandas dataframe --> csv file 
        df.to_csv(path, index=False)
    
    
    @classmethod
    def load_data_to_parquet(cls, df, path):
        #take pandas dataframe --> parquet file 
        df.to_parquet(path, index=False)  # 'path' should be a string with the full file path
    
def main():
    csv_list = ["data/Tybee-Parcels-2023(in).csv", "data/Tybee-Parcels-2020.csv", "data/Tybee-Parcels-2021.csv", "data/Tybee-Parcels-2022.csv"]
    cleaned_df = PropertyDataLoader.clean_csv_list(csv_list)
    ordered_df = PropertyDataLoader.order_df(cleaned_df)
    PropertyDataLoader.load_data_to_csv(ordered_df, "preprocessed_parcels_data.csv")
    print(ordered_df.head())

if __name__ == "__main__":
    main()
