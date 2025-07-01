import pandas as pd
import logging
from typing import List

logging.basicConfig(
    filename="property_data_loader.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

class PropertyDataLoader:
    ADDRESS = [
        "PropAddres", "PropAddr_1", "PropAddr_2", "PropAddr_3", "PropAddr_5",
        "PropAddr_6", "PropAddr_7", "PropAddr_8", "PropAddr_9", "PropAddr10"
    ]

    REQUIRED_COLUMNS = {
        "Sale_Price": "Sale Price",
        "FairMarket": "Fair Market Value",
        "Acres": "Acres",
        "YearBuilt": "Year Built"
    }

    @classmethod
    def clean_csv_list(cls, csv_paths: List[str]) -> pd.DataFrame:
        all_csv_files = []
        for path in csv_paths:
            try:
                dataframe = pd.read_csv(path)
            except Exception as e:
                logging.error(f"Error: {e}")
                continue
            address = []
            for idx in range(len(dataframe)):
                parts = []
                for col in cls.ADDRESS:
                    value = dataframe.at[idx, col]
                    if pd.notnull(value):
                        parts.append(str(value).strip())
                full_address = ' '.join(parts)
                address.append(full_address)
            dataframe["Property Address"] = address
            required_cols = list(cls.REQUIRED_COLUMNS.keys()) + ["Property Address"]
            renamed_cols = cls.REQUIRED_COLUMNS.copy()
            renamed_cols["Property Address"] = "Property Address"
            df_subset = dataframe[required_cols].copy()
            df_subset.rename(columns=renamed_cols, inplace=True)
            for idx, row in df_subset.iterrows():
                if row.isnull().any():
                    missing_fields = row.index[row.isnull()].tolist()
                    logging.info(f"{path} - Row {idx} - Missing Fields: {missing_fields} — Row Data: {row.to_dict()}")
            cleaned = df_subset.dropna()
            all_csv_files.append(cleaned)
        combined_dataframe = pd.concat(all_csv_files, ignore_index=True)
        final_dataframe = combined_dataframe.drop_duplicates().reset_index(drop=True)
        logging.info(f"Final Cleaned Dataset: {len(final_dataframe)} Rows")
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
    csv_path = ["Tybee-Parcels-2023(in).csv"]
    cleaned_df = PropertyDataLoader.clean_csv_list(csv_path)
    ordered_df = PropertyDataLoader.order_df(cleaned_df)
    PropertyDataLoader.load_data_to_csv(ordered_df, "preprocessed_parcels_data.csv")
    print(ordered_df.head())

if __name__ == "__main__":
    main()
