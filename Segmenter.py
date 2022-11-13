def hmm_segmentation(data, window_size=1000, n_states=2):
    
    '''
    Version 3.0:
    
    Function to segment a signal using HMM.
    
    Input: 
    data: Pandas Dataframe: Pandas dataframe of a signal
    window_size: int; default 1000: Size of sliding window applied during moving average
    n_states: int: default 2: Number of states in HMM
    
    Output:
    decode_array: array: A signal of 0's and 1's, where the region between 1's is a segment.
    segments: list of tuples: A list of tuples with the starting and ending index of each segment
    motifs: list of pandas dataframes: A list of segments made of pandas dataframes
    '''    
    
    from hmmlearn import hmm
    import numpy as np
    
    print("Window size set to: ", window_size)
            
    # Find the moving average of the data
    moving_average = data.rolling(window_size, min_periods=1).mean()
    
    # Initialize an HMM
    HMM = hmm.GMMHMM(n_components = n_states, random_state=42, n_iter=100, init_params="mcs")
    
    # Fit an HMM for the moving average
    HMM.fit(np.atleast_2d(moving_average).T)
    
    # Extract the "decode signal"
    decoded = HMM.decode(np.atleast_2d(moving_average).T)
    
    # Return the segment
    decode_array = decoded[1]
    
    # Extract the indices of the segments
    segment_indices = []
    
    for i in range(len(decode_array)):
        try:
            if decode_array[i] == 0 and decode_array[i+1] == 1:
                segment_indices.append(i)
                
            elif decode_array[i] == 1 and decode_array[i+1] == 0:
                segment_indices.append(i)
        except:
            break
            
    # Creating the segments list
    segments = []
    
    for i in range(len(segment_indices)):
        try:
            segments.append((segment_indices[i],segment_indices[i+1]))
        except:
            break
            
    # Extracting the motifs
    motifs = []
    for segment in segments:
        sample = data[segment[0]:segment[1]]
        motifs.append(sample)
    
    return decode_array, segments, motifs,HMM