import pickle
import pandas as pd
from statistics import mean
from datetime import datetime

categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

if __name__ == "__main__":

    df = read_data(r'dataset\fhv_tripdata_2021-03.parquet')
   
    df1 = df.copy(deep=True)

    today = datetime.today()
    year = today.year
    month = today.month     

    df1['ride_id'] = f'{year:04d}/{month:02d}_' + df1.index.astype('str')

    # print(f'The dataframe column is {df1["ride_id"]}')
    model_path = 'models_pickle/model.bin'

    with open(model_path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    
    df2 = pd.DataFrame()

    df2['pred'] = y_pred.tolist()
    # print(df2)

    df_result = pd.concat([df1.ride_id, df2], axis=1, join='inner')
    
    df_result.to_parquet(
   'output_file',
    engine='pyarrow',
    compression=None,
    index=False
)

    # df3 = df2.append(df1.ride_id, ignore_index=True)

    # data_to_append = {}
    # for i in range(len(df1.ride_id)):
    #     data_to_append[df1.ride_id] = y_pred[i]
    # df2 = df2.append(data_to_append, ignore_index = True)

    # print(f'New Dataframe is{df_result}')

    # print(type(y_pred))
    # mean_duration = mean(y_pred)
    mean_duration = y_pred.mean()
    print(f'Mean predicted duration {mean_duration}')