N_STATES = 2
WINDOW_SIZE = 50
N_COEFF = 15

ALPHA_SPIKE = 0.1
ALPHA_PMS = 0.1
ALPHA_PSD = 0.5
ALPHA_LN = 1

# Alpha values
alpha_value_spike = [0.02, 0.1, 0.1, 0.1, 0.01, 0.02, 0.02, 0.02, 0.01, 0.005, 0.005, 0.002, 0.003, 0.1, 0.02, 0.03,
                     0.1]
alpha_value_ln = [1, 2, 5, 1, 1, 1, 1, 5, 5, 5, 1, 1, 1, 1, 1, 1, 5]
alpha_value_psd = [0.5, 1, 0.5, 0.1, 0.02, 0.02, 10,
                   1, 0.05, 0.2, 0.2, 0.1, 0.1, 0.07, 0.2, 0.02, 5]

list_of_frequencies = [100, 100, 100, 100, 100,
                       100, 100, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1]
list_of_kernel_size = [5, 3, 3, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
list_of_moving_window_size = [151, 151, 151, 151,
                              151, 151, 151, 21, 21, 3, 3, 3, 3, 3, 5, 5, 3]
list_of_filter = ['median', 'median', 'savgol', 'median', 'median', 'median', 'median', 'median', 'median',
                  'median', 'median', 'savgol', 'savgol', 'savgol', 'savgol', 'savgol', 'median']
# freq_dict = dict(zip(sensor_names, list_of_frequencies))
# dict_kernel_size = dict(zip(sensor_names, list_of_kernel_size))
# dict_moving_window_size = dict(zip(sensor_names, list_of_moving_window_size))
# dict_list_of_filter = dict(zip(sensor_names, list_of_filter))