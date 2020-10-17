#utils for preprocessing

import numpy as np
import pandas as pd
import time 

#Function to standardize X based on X_train:
def standardize(data=None, variable_stop_index=34): #, val=None, test=None,
    variables = list(data)[:variable_stop_index]
    
    #initialize header info:
    data_z = data.copy(deep=True)
    
    ##get train stats (check if non-nan!)
    #mean = train[variables].mean(axis=0)
    #std = (train[variables]-mean).std(axis=0)
    #if (mean.isnull().sum() or std.isnull().sum()) > 0:
    #    print('Nan in statistics found.. increase na_thres of droping columns')
    
    #Sepsis Challenge provided data statistics (for time series variables)
    mean = np.array([
        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
        38.9974, 10.5585,  286.5404, 198.6777])
    std = np.array([
        17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
        14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
        6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
        19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
        1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
        0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
        29.8928, 7.0606,  137.3886, 96.8997])

    #standardize by train statistics:
    data_z[variables] = (data[variables]-mean)/std

    stats = {'mean': np.array(mean).tolist(), 'std': np.array(std).tolist()}

    return data_z, stats   


#Function to transform data from sparse df format to compact format as to feed to mgp-tcn script.
#Here the function takes a single patient file as data
def compact_transform(data, variable_stop_index=34):
    start_time = time.time()
    #initialize outputs:
    values = [] # values[i][j] stores lab/vital value of patient i as jth record (all lab,vital variable types in same array!) 
    times = [] # times[i][:] stores all distinct record times of pat i (hours since icu-intime) (sorted)
    ind_lvs = [] # ind_lvs[i][j] stores index (actually id: e.g. 0-9) to determine the lab/vital type of values[i][j]. 
    ind_times = [] # ind_times[i][j] stores index for times[i][index], to get observation time of values[i][j].
    labels = [] # binary label if case/control
    num_tcn_grid_times = [] # array with numb of grid_times for each patient
    tcn_grid_times = [] # array with explicit grid points: np.arange of num_tcn_grid_times
    num_obs_times = [] #number of times observed for each encounter; how long each row of T (times) really is
    num_obs_values = [] #number of lab values observed per encounter; how long each row of Y (values) really is
    #onset_hour = [] # hour, when sepsis (or control onset) occurs (practical for horizon analysis)
    pat_ids = [] # unique identifier for a icustay    

    #If data represents only one patient:
    #pat = data
    
    #Process all patients: 'data' is actually a list of pd dfs (per patient) -- not a large df anymore, commented below with ##
    ##ids = data['pat_id'].unique()
    ##for patid in ids:
    for pat in data:
        ##pat = data.query( "pat_id == @patid" ) 
        ##select only those rows where icustay_id matches current icuid iteration

        num_tcn_grid_time = pat['ICULOS'].values[-1]
        
        num_tcn_grid_times.append(num_tcn_grid_time)
        tcn_grid_times.append(np.arange(num_tcn_grid_time))

        # Get measurement observation times:
        pat_times = pat['ICULOS'].values # in sepsis challenge NOT irregularly observed, but already binned..
        times.append(pat_times)
        num_obs_times.append(len(pat_times))

        # Write label to labels
        labels.append(pat['SepsisLabel'].values)
        
        # Loop over variables to get values:
        variables = np.array(list(pat.iloc[:,:variable_stop_index])) #get variable names

        pat_values = [] #values list of current patient
        pat_ind_lvs = [] #ind_lvs list of curr patient
        pat_ind_times = [] #ind_times list of curr patient
        
        for v_id, var in enumerate(variables): #loop over variable ids and names
            vals = pat[[var, 'ICULOS']].dropna().values # values and corresponding time which are non-nan
            for val, chart_time in vals:
                pat_values.append(val)
                pat_ind_lvs.append(v_id)
                time_index = np.where(pat_times == chart_time)[0][0] # get index of pat_times where time matches
                pat_ind_times.append(time_index) # append index with which pat_times[index] return chart time of current value
        values.append(np.array(pat_values)) #append current patients' values to the overall values list
        num_obs_values.append(len(pat_values)) #append current patients' number of measurements
        ind_lvs.append(np.array(pat_ind_lvs)) #append current patients' indices of labvital ids to overall ind_lvs list
        ind_times.append(np.array(pat_ind_times)) #append current patients' indices of times to overall ind_times list
        
        #Append onset hour of current patient:
        #onset_hour.append(onset_hours.loc[onset_hours['icustay_id']==icuid]['onset_hour'].values[0])
        pat_ids.append(patid)

    results = {}
    items = ['values', 'times' , 'ind_lvs', 'ind_times' , 'labels', 'num_tcn_grid_times', 
                'tcn_grid_times', 'num_obs_times', 'num_obs_values',  'pat_ids']
    result_list = [values,times,ind_lvs,ind_times,labels, num_tcn_grid_times, 
                tcn_grid_times,num_obs_times,num_obs_values,  pat_ids]
    for value, key in zip(result_list, items):
        results[key] = np.array(value)
    print('Reformatting to compact format took {} seconds'.format(time.time() - start_time)) 
    return results




