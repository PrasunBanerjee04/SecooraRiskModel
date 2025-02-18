import pandas as pd
import requests as r
from datetime import *        

class DataLoader:

    def __init__(self, url):
        self.api_url = url
    
    @classmethod
    def extract_data(numOfDays):
        original_link = "https://api.sealevelsensors.org/v1.0/Datastreams(262)/Observations?$skip=0&$orderby=%40iot.id+desc&"
        now = datetime.now(timezone.utc)
        yest = (now - timedelta(days = numOfDays)).strftime('%Y-%m-%d')
        formatted_time = now.strftime("T%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        now = now.strftime('%Y-%m-%d')
        link = original_link + "$filter=phenomenonTime%20ge%20" + yest + formatted_time + "%20and%20phenomenonTime%20le%20" + now + formatted_time
        metadata = r.get(link)
        data = metadata.json()
        adict = {"resultTime":[], "reading":[]}
        while "@iot.nextLink" in data:
            for dictionary in data["value"]:
                rTime = dictionary["resultTime"]
                result = dictionary["result"]
                adict["resultTime"].append(rTime)
                adict["reading"].append(result)
            link = data["@iot.nextLink"]
            metadata = r.get(link)
            data = metadata.json()
        df = pd.DataFrame(adict)
        return df
    
    @classmethod
    def transform_data(data):
        #take two column data and integrate advanced features

        
        return data
    
    @classmethod
    def load_data_to_csv(data):
        #take pandas dataframe --> csv file 

        
        return data
    
    @classmethod
    def load_data_to_parquet(data):
        #take pandas dataframe --> parquet file 

        
        return data
