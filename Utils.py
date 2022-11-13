import os
join = os.path.join

def preprocess_normal(dfn):
    # Collect deleted sensors
    deleted_sensors = []

    # Remove all cols with unique < 4 
    for col in dfn.columns:
        if len(dfn[col].unique()) < 4:
            dfn = dfn.drop(col, axis=1)
            deleted_sensors.append(col)

    # Store deleted sensors
    with open(join(os.getcwd(), 'WADI', 'deleted_sensors.txt'), 'w') as f:
        for sensor in deleted_sensors:
            f.write(sensor + '\n')

    # Impute missing values with mean of before and after for each sensor
    for col in dfn.columns:
        dfn[col] = dfn[col].interpolate(method='linear', limit_direction='both')

    print(dfn.shape)
    return dfn, deleted_sensors