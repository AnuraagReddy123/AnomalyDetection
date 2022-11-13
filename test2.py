import os
import numpy as np
join = os.path.join
# Open files in sensor

sensors = os.listdir('WADI/Sensors')
for sensor in sensors:
    arr = np.load(join('WADI', 'Sensors', sensor), allow_pickle=True)
    print(arr.shape)
