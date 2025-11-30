import numpy as np
import pandas as pd

def data_dict_to_df(data_dict):
    """
    Ex of data_dict: 
    {
    'channel_names': ['AUX7', 'AUX8', 'AUX9', 'AUX10', 'AUX11', 'AUX12'], 
    'time_series': array([[  6846.55 ,  -4181.685, -13241.481,   4200.757,  -2022.079,  3539.197],
       [  6824.796,  -4189.582, -13240.736,   4201.8  ,  -2011.351,  3535.919],
       [  6805.277,  -4197.032, -13242.226,   4196.436,  -2007.179,  3537.707],
       ...,
       [  4408.314,  -6320.878, -14723.137,   3769.849,    195.637,  4496.522],
       [  4404.589,  -6321.474, -14724.478,   3730.364,    182.674,  4497.565],
       [  4406.526,  -6318.196, -14722.541,   3704.736,    172.84 ,  4501.886]]), 
    'time_stamps': array([10144.28630628, 10144.28730628, 10144.28830628, ...,   11938.29793946, 11938.29893946, 11938.29993946]), 
    'excluded_trigger': 'TRIGGER'
    }

    Ex of data_df: 
    Time,s,"Noraxon Desk Receiver.EMG 1,uV","Noraxon Desk Receiver.EMG 2,uV","Noraxon Desk Receiver.EMG 3,uV","Noraxon Desk Receiver.EMG 6,uV","Noraxon Desk Receiver.EMG 7,uV","Noraxon Desk Receiver.EMG 8,uV","Noraxon Desk Receiver.Sync,On"
    0.0600   -3.720640 -1.335040  0.866479  4.70090  -2.428430  -0.433885                                                  0                                                                                                                                                                                           
    0.0607   -2.805150  2.934970  4.232420  10.20110 -0.285218  -0.121488                                                  0                                                                                                                                                                                           
    0.0613   -0.058685  0.190720  2.999350  2.56932  -0.896399   2.013230                                                  0                                                                                                                                                                                           
    0.0620   -0.974173 -0.423822  0.866479 -3.53995  -1.507580  -2.863640                                                  0                                                                                                                                                                                           
    0.0627   -1.279340  1.409210  0.866479  1.65578  -1.817240  -4.095880                                                  0                                                                                                                                                                                           
    ...                                                                                                                  ...                                                                                                                                                                                           
    699.1040  9.401360 -1.642310 -2.799390 -23.67580  2.762540   3.835550                                                  0                                                                                                                                                                                           
    699.1047  3.298100 -3.782610 -2.799390 -21.84870  12.834800  1.093390                                                  0                                                                                                                                                                                           
    699.1053  3.603270 -6.219590 -1.566330 -25.80740  12.223600  5.970260                                                  0                                                                                                                                                                                           
    699.1060  9.401360 -3.168070 -0.666523 -26.72090  3.064050   2.933060                                                  0                                                                                                                                                                                           
    699.1067  6.654890  0.190720 -4.632330 -23.06670  0.627479  -5.918200                                                  0 
    """
    channel_names = data_dict['channel_names']
    time_series = data_dict['time_series']
    time_stamps = data_dict['time_stamps']

    data_df = pd.DataFrame(time_series, columns=channel_names)
    data_df.insert(0, 'Time,s', time_stamps)

    col_names = list(data_df.columns)
    ch_count = 0
    for i in range(len(col_names)):
        col = col_names[i]
        if 'time' in col.lower():
            col_names[i] = 'Time'
        if any(x in col.lower() for x in ['ch', 'emg', 'aux']):
            col_names[i] = f"Channel_{ch_count}"
            ch_count+=1
        if any(x in col.lower() for x in ['trigger', 'sync']):
            col_names[i] = 'Trigger'

    print(col_names)

    data_df.columns=col_names

    return data_df

def clean_data_df(data_df):
    df = data_df.copy()
  
    col_names = list(df.columns)
    ch_count = 0
    for i in range(len(col_names)):
        col = col_names[i]
        if 'time' in col.lower():
            col_names[i] = 'Time'
        if any(x in col.lower() for x in ['ch', 'emg']):
            col_names[i] = f"Channel_{ch_count}"
            ch_count+=1
        if any(x in col.lower() for x in ['trigger', 'sync']):
            col_names[i] = 'Trigger'

    df.columns=col_names

    return df