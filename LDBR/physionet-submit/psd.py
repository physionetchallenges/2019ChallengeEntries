# psd.py - patient sepsis details
# function that extract and setup features , target details for processing

import pandas as pd
import numpy as np
import tensorflow as tf
import math

# from input, for vitals exract mean of last "x" readings, flag out-of-range, not-avail, in-range 
def hello():
    print( "hello from psd")
# HR: 80, Temp: 36.65, Resp:14, MAP: 90
col_mean_dict = {"HR": 70, "O2Sat": 97.5,"Temp": 36.80,"SBP": 100, "MAP": 75, "DBP": 70, "Resp": 18, 
                 "EtCO2" : 40, 'BaseExcess': 0, 'HCO3': 26, 'FiO2': 0.21, 'pH': 7.40, 'PaCO2': 40, 'SaO2': 97, 
                 'AST': 25, 'BUN': 13.5, 'Alkalinephos': 95 , 'Calcium': 9.5, 'Chloride': 102, 
                 'Creatinine': 0.85, 'Bilirubin_direct': 0.65,
                  'Glucose': 100, 'Lactate': 6.75, 'Magnesium': 2.0, 'Phosphate': 3.5, 'Potassium': 4.25,
                'Bilirubin_total': 0.65, 'TroponinI': 0.2, 'Hct': 45.5, 'Hgb': 14.6, 'PTT': 30, 'WBC': 7550,
                'Fibrinogen': 275, 'Platelets': 27.5
                }
c_header = ["HR", "O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN","Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct","Glucose","Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets","Age","Gender","Unit1","Unit2","HospAdmTime","ICULOS"]
MIN = {"HR": 80, "O2Sat": 20, "Temp": 35, "SBP": 30, "MAP": 20, "DBP": 20, "Resp": 15}
MAX = {"HR": 95, "O2Sat": 100, "Temp": 38, "SBP": 250, "MAP": 300, "DBP": 200, "Resp": 21}


def normal_colvalue( colname ):
  #print( colname, col_mean_dict[colname] )
  return col_mean_dict[colname]

def bin_age( age ):
    bage = np.floor(age/10)
    return bage/10.
    #return age

def bin_iculos( los ):
    blos = np.floor(los/5.0)    # was5
    return blos
    #return los

# will calculate mean of Gender, when fitting the model..if rows are from diff genders, mean will help
def compute_column_stats( cols_wecare, indf, timewindow ):
    print( "compute col")
    features_data = pd.DataFrame()
    #print( indf.head(30) )
    
    for col_name in cols_wecare:         # special case, indx 0
        print( 'processing ',col_name)
        j = 0
        nosof_rows = indf.shape[0]
        print( 'nos of rows', nosof_rows)
        while (j+timewindow) < nosof_rows :
            col_str = col_name + '-mean'
            v1 = indf.at[j,col_name]
            meanv = indf[[col_name]].iloc[j:(j+1+timewindow)].mean(axis=0)
            meanv = round(meanv[0], 2)
            # print( "j, j+tm, mean:  vals: ", j, (j+timewindow), meanv, indf[[col_name]].iloc[j:(j+1+timewindow)] )
            if( math.isnan(meanv) ):
                #print( "Mean value is NaN *********")
                meanv = normal_colvalue( col_name )
            #print( "j {}, j+tm {}, mean {} vals: {}".format( j, (j+timewindow), meanv, indf[[col_name]].iloc[j:(j+1+timewindow)]) )
            features_data.at[j, col_str] = meanv

            #print("label vals", j, indf.at[j+timewindow, 'SepsisLabel'], indf.at[j+timewindow, 'patient'] )
            features_data.at[j, 'SepsisLabel'] = indf.at[j+timewindow, 'SepsisLabel']
            #features_data.at[j, 'patient'] = indf.at[j+timewindow, 'patient']
            if( not bool(j%1000)):
                print( ".",end='' )
            j = j+1

    return features_data

def diff_from_normal( cols_wecare, indf ):
    features_data = pd.DataFrame()
    
    for col_name in cols_wecare:         # special case, indx 0
        #print( 'processing ',col_name)
        j = 0
        nosof_rows = indf.shape[0]
        #print( 'nos of rows', nosof_rows)
        while j < nosof_rows :
            col_str = col_name + '-pdiff'    # t1 - t0
            if( col_name == 'Gender' or col_name == 'Age' ) :
                features_data.at[j, col_name] = indf.at[j, col_name]
            else:
                #v1 = indf.loc[indf.index[j],col_name]
                v1 = indf.at[j,col_name]
                if( math.isnan(v1) ):
                    #v1 = 0
                    v1 = normal_colvalue(col_name)
                #diff = normal_colvalue(col_name) - v1
                diff = v1 - normal_colvalue(col_name)
                features_data.at[j, col_str] = diff

            features_data.at[j, 'SepsisLabel'] = indf.at[j, 'SepsisLabel']
            j = j+1
            #print(" row {} col {} , diff {} sepsis {}".format( j, v1, diff, features_data.at[j,'SepsisLabel']) )
           
        # print( "done with {} rows: {}".format(col_name,nosof_rows) )

    return features_data

#this must be called before other functions that use cols_wecare
def lactate_ph_risk( indf ):
    lrisk=0; prisk=0

    max_lactate = indf['Lactate'].max()
    max_ph = indf['pH'].max()
    sepsis = indf['SepsisLabel'].max()
    if( math.isnan(max_lactate) or max_lactate==0 ):
        lrisk = 0
    elif( max_lactate > (normal_colvalue('Lactate')*1.1) ):
        lrisk = 10
    if( math.isnan(max_ph) or max_ph==0 ):
        prisk = 0
    elif( max_ph > (normal_colvalue('pH')*1.1) ):
        prisk = 10
    if( lrisk==1 or prisk==1 ):
        print( 'lactate ph risk flagged, Sepsis is {}'.format(sepsis) )
    indf['risk'] = lrisk+prisk
    return indf

# for selected columns return col_reading/normalvalue 
def factor_ofnormal( cols_wecare, indf ):
    features_data = pd.DataFrame()
    prev_values = pd.DataFrame()

    if( 'risk' in indf.columns ):
        features_data['risk'] = indf['risk']
    
    for col_name in cols_wecare:         # special case, indx 0
        #print( 'processing ',col_name)
        prev_values.at[0,col_name] = np.nan
        j = 0
        nosof_rows = indf.shape[0]
        #print( 'nos of rows', nosof_rows)
        while j < nosof_rows :
            #col_str = col_name + '-pdiff'    
            col_str = col_name   
            if( col_name == 'Gender' ) :
                features_data.at[j, col_name] = indf.at[j, col_name]
            elif( col_name == 'Age' ):
                features_data.at[j, col_name] = bin_age( indf.at[j,col_name] )
            elif( col_name == 'ICULOS' ):
                #features_data.at[j, col_name] = (indf.at[j, col_name] - iculos_mean)/iculos_std
                features_data.at[j, col_name] = bin_iculos(indf.at[j,col_name])
            # elif( 'risk' in ):
            #     features_data.at[j, col_name] = indf.at[j, col_name]
            else:
                #v1 = indf.loc[indf.index[j],col_name]
                v1 = indf.at[j,col_name]
                if( math.isnan(v1) ):
                    if( math.isnan(prev_values.at[0,col_name]) ):
                        prev_values.at[0,col_name] = 0
                        v1 = 0
                    else:
                        #print( 'gave prev value {} {}'.format(col_name, j))
                        v1 = prev_values.at[0,col_name]
                    #v1 = 0
                    #v1 = normal_colvalue(col_name)
                diff = v1 - normal_colvalue(col_name)
                f = v1 / normal_colvalue(col_name)       # normalizze the value
                features_data.at[j, col_str] = f

            features_data.at[j, 'SepsisLabel'] = indf.at[j, 'SepsisLabel']
            j = j+1
            #print(" row {} col {} , diff {} sepsis {}".format( j, v1, diff, features_data.at[j,'SepsisLabel']) )           
        #print( "done with {} rows: {}".format(col_name,nosof_rows) )

    return features_data


def min_max(col_name, v):
    min_v = MIN[col_name]
    max_v = MAX[col_name]
    v_n = (v - min_v)/(max_v - min_v)
    return v_n


def normalize_data(cols_wecare, midx):
    # engg_fl1 = psd.factor_ofnormal(cols_wecare, midx)
    engg_fl1 = minmax_normalization(cols_wecare, midx)
    # print("minmax_normalization done..")
    engg_fl1 = create_crossfeatures(engg_fl1)
    # print("create_crossfeatures done..")
    return engg_fl1


def minmax_normalization(cols_wecare, indf, predict=False):
    features_data = pd.DataFrame()
    prev_values = pd.DataFrame()

    for col_name in cols_wecare:  # special case, indx 0
        # print( 'processing ',col_name)
        j = 0
        nosof_rows = indf.shape[0]
        prev_values.at[0, col_name] = np.nan
        # print( 'nos of rows', nosof_rows)
        while j < nosof_rows:
            col_str = col_name  # t1 - t0
            if (col_name == 'Gender'):
                features_data.at[j, col_name] = indf.at[j, col_name]
            elif (col_name == 'Age'):
                features_data.at[j, col_name] = bin_age(indf.at[j, col_name])
            elif (col_name == 'ICULOS'):
                # features_data.at[j, col_name] = (indf.at[j, col_name] - iculos_mean)/iculos_std
                features_data.at[j, col_name] = bin_iculos(indf.at[j, col_name])
            else:
                v1 = indf.at[j, col_name]
                if (math.isnan(v1)):
                    if (math.isnan(prev_values.at[0, col_name])):
                        prev_values.at[0, col_name] = 0
                        v1 = 0
                    else:
                        v1 = prev_values.at[0, col_name]
                f = min_max(col_name, v1)
                features_data.at[j, col_str] = f

            if not predict:
                features_data.at[j, 'SepsisLabel'] = indf.at[j, 'SepsisLabel']
            j = j + 1
            # print(" row {} col {} , diff {} sepsis {}".format( j, v1, diff, features_data.at[j,'SepsisLabel']) )
        # print( "done with {} rows: {}".format(col_name,nosof_rows) )

    return features_data

def factor_ofnormal_labs( cols_wecare, indf ):
    features_data = pd.DataFrame()
    
    for col_name in cols_wecare:         # special case, indx 0
        #print( 'processing ',col_name)
        j = 0
        nosof_rows = indf.shape[0]
        #print( 'nos of rows', nosof_rows)
        while j < nosof_rows :
            col_str = col_name 
            if( col_name == 'Gender' ) :
                features_data.at[j, col_name] = indf.at[j, col_name]
            elif( col_name == 'Age'):
                ag = indf.at[j,col_name]
                if(ag > 60):    # > 60, more prone to sepsis
                    ag = ag*2
                    features_data.at[j, col_name] = ag
                else:
                    features_data.at[j, col_name] = ag
            else:
                #v1 = indf.loc[indf.index[j],col_name]
                v1 = indf.at[j,col_name]
                if( math.isnan(v1) ):
                    v1 = 0
                    #v1 = normal_colvalue(col_name)
                diff = v1 - normal_colvalue(col_name)
                f = v1 / normal_colvalue(col_name)       # normalizze the value
                features_data.at[j, col_str] = f

            features_data.at[j, 'SepsisLabel'] = indf.at[j, 'SepsisLabel']
            j = j+1
            #print(" row {} col {} , diff {} sepsis {}".format( j, v1, diff, features_data.at[j,'SepsisLabel']) )           
        # print( "done with {} rows: {}".format(col_name,nosof_rows) )

    return features_data

def create_crossfeatures( indf, cols_to_x ):
    #print( 'new feature that is cross of {}'.format(cols_to_x) )
    for index, row in indf.iterrows():
        xfeature = 1
        # for col_name in cols_to_x:
        #     #print( 'col {} colvalue {} x {}'.format(col_name, indf.at[index,col_name], xfeature))
        #     xfeature = xfeature * indf.at[index, col_name]
        # xfeature = xfeature * 0.0005
        # indf.at[index,'xf'] = xfeature
        hr2 = indf.at[index,'HR-pdiff'] * indf.at[index,'HR-pdiff']  # +ve impact
        map2 = indf.at[index,'MAP-pdiff'] * indf.at[index,'MAP-pdiff']
        o2sat2 = indf.at[index,'O2Sat-pdiff'] * indf.at[index,'O2Sat-pdiff']
        resp2 = indf.at[index,'Resp-pdiff'] * indf.at[index,'Resp-pdiff']
        sbp2 = indf.at[index,'SBP-pdiff'] * indf.at[index,'SBP-pdiff']
        indf.at[index,'HR2'] = hr2 
        indf.at[index,'MAP2'] = map2
        indf.at[index,'O2Sat2'] = o2sat2
        indf.at[index,'Resp2'] = resp2
        indf.at[index,'SBP2'] = sbp2
    return indf

def create_crossfeaturesR2( indf, cols_to_x ):
    print( 'new feature that is cross of {}'.format(cols_to_x) )
    for index, row in indf.iterrows():
        hr2 = indf.at[index,'HR']
        map2 = indf.at[index,'MAP'] 
        resp2 = indf.at[index,'Resp']
        indf.at[index, 'xSP'] = hr2*map2*resp2
        # if( hr2 >= 1.2 and map2 >= 1.2 and resp2 >= 1.2 ):
        #     indf.at[index, 'xSP'] = 1    
    return indf

# for selected columns return (col_value - mean)/std dev
def normalize_by_mean_std( cols_wecare, indf ):
    features_data = pd.DataFrame()
    
    for col_name in cols_wecare:         # special case, indx 0
        #print( 'processing ',col_name)
        j = 0
        nosof_rows = indf.shape[0]
        col_mean = indf[col_name].mean()
        col_std = indf[col_name].std()
        print( "{} mean: {}  std: {}".format(col_name, col_mean, col_std))
        while j < nosof_rows :
            col_str = col_name + '-ms'    # t1 - t0
            if( col_name == 'Gender' or col_name == 'Age' ) :
                features_data.at[j, col_name] = indf.at[j, col_name]
            else:
                v1 = indf.at[j,col_name]
                if( math.isnan(v1) ):
                    v1 = 0
                    #v1 = normal_colvalue(col_name)
                diff = v1 - col_mean
                f = v1 / col_std      # normalizze the value
                features_data.at[j, col_str] = f

            features_data.at[j, 'SepsisLabel'] = indf.at[j, 'SepsisLabel']
            j = j+1
            #print(" row {} col {} , diff {} sepsis {}".format( j, v1, diff, features_data.at[j,'SepsisLabel']) )           
        # print( "done with {} rows: {}".format(col_name,nosof_rows) )

    return features_data


def features_only(df):
    # drop col (axis=1 )
    df2 = df.drop('SepsisLabel', axis=1 )
    return df2


def preprocess_targets(l_dataframe):
  
    output_targets = pd.DataFrame()
    output_targets["SepsisLabel"] = l_dataframe['SepsisLabel']
    
    return output_targets

def remove_features( indf, cols_toremove ):
    df2 = indf
    print( 'feature to remove {}'.format(cols_toremove))
    for col_name in cols_toremove:
        df2.drop(col_name, axis=1, inplace=True)
    return df2


# save complete model
def model_save( m, filename ):
    #model_json = m.to_json()
    #with open(filename, "w") as json_file:
        #json_file.write(model_json)
    # serialize weights to HDF5
    m.save(filename)
    print("Saved model to file {}".format(filename))

# load json and create model
def load_model( filename ):
    from keras.models import load_model

    model = load_model(filename)
    # summarize model.
    model.summary()
    return model

#Creates a tf feature spec from the dataframe and columns specified.
def create_feature_spec(df, columns=None):
    feature_spec = {}
    if columns == None:
        columns = df.columns.values.tolist()
    for f in columns:
        if df[f].dtype is np.dtype(np.int64) or df[f].dtype is np.dtype(np.int32):
            feature_spec[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        elif df[f].dtype is np.dtype(np.float64):
            feature_spec[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.float32)
        else:
            feature_spec[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)
    return feature_spec

 #Converts a dataframe into a list of tf.Example protos.
def df_to_examples(df, columns=None):
    examples = []
    if columns == None:
        columns = df.columns.values.tolist()
    for index, row in df.iterrows():
        example = tf.train.Example()
        for col in columns:
            if df[col].dtype is np.dtype(np.int64) or df[col].dtype is np.dtype(np.int32):
                example.features.feature[col].int64_list.value.append(int(row[col]))
            elif df[col].dtype is np.dtype(np.float64):
                example.features.feature[col].float_list.value.append(row[col])
            elif row[col] == row[col]:
                example.features.feature[col].bytes_list.value.append(row[col].encode('utf-8'))
        examples.append(example)
    return examples


def df_to_timestep_predict( indf, time_step ):
    data = []
    labels= []
    datalength = indf.shape[0]
    npdata = indf.to_numpy()
    # print( npdata.shape )
    if datalength - time_step == 0:
        return npdata[np.newaxis, ...]
    for i in range(0, datalength-time_step):
        indices = range(i,i+time_step,1)
        data.append(npdata[indices])

    return np.array(data)

# time step data for LSTM
def df_to_timestep( indf, time_step ):
    data = []
    labels= []
    datalength = indf.shape[0]
    d = features_only(indf)
    #print( d)
    l = preprocess_targets(indf)
    npdata = d.to_numpy()
    nplabel = l.to_numpy()
    #print( npdata.shape )
    for i in range(0, datalength-time_step):
        indices = range(i,i+time_step,1)
        data.append(npdata[indices])

        labels.append(nplabel[i+time_step])

    return np.array(data), np.array(labels)

# modelf.history.keys() --> dict_keys(['loss', 'auc', 'val_loss', 'val_auc', 'lr'])
def plot_train_history(modelf, title):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    loss = modelf.history['loss']
    val_loss = modelf.history['val_loss']
    auc = modelf.history['auc']
    val_auc = modelf.history['val_auc']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, c='b', label='Training loss')
    plt.plot(epochs, val_loss, c='r', label='Validation loss')

    plt.plot(epochs, auc, c='y', label='Training auc')
    plt.plot(epochs, val_auc, c='g', label='Validation auc')
    plt.title(title)
    plt.legend()

    plt.show()

def plot_predict( ntp, ntn ):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    plt.scatter( range(ntn.shape[0]), ntn, c='r', label="True -ve")
    plt.scatter( range(ntp.shape[0]), ntp, c='g', label="True +ve")
    plt.legend()
    plt.show()



def df_to_timestep_submit( indf, time_step ):
    data = []
    datalength = indf.shape[0]
    #d = features_only(indf)
    npdata = indf.to_numpy()
    # print( npdata.shape )
    for i in range(0, datalength-time_step):
        indices = range(i,i+time_step,1)
        data.append(npdata[indices])

    return np.array(data)


# for selected columns return col_reading/normalvalue 
def factor_ofnormal_submit( cols_wecare, indf ):
    features_data = pd.DataFrame()
    
    for col_name in cols_wecare:         # special case, indx 0
        #print( 'processing ',col_name)
        j = 0
        nosof_rows = indf.shape[0]
        #print( 'nos of rows', nosof_rows)
        while j < nosof_rows :
            col_str = col_name + '-pdiff'    # t1 - t0
            if( col_name == 'Gender' or col_name == 'Age' ) :
                features_data.at[j, col_name] = indf.at[j, col_name]
            else:
                #v1 = indf.loc[indf.index[j],col_name]
                v1 = indf.at[j,col_name]
                if( math.isnan(v1) ):
                    v1 = 0
                    #v1 = normal_colvalue(col_name)
                diff = v1 - normal_colvalue(col_name)
                f = v1 / normal_colvalue(col_name)       # normalizze the value
                features_data.at[j, col_str] = f

            j = j+1
            #print(" row {} col {} , diff {} sepsis {}".format( j, v1, diff, features_data.at[j,'SepsisLabel']) )           
        # print( "done with {} rows: {}".format(col_name,nosof_rows) )

    return features_data

def read_patient_file( dir, filename ) :  

    path_replace = dir + '/'
    #print( 'filename is {}'.format(filename) )
    dff = pd.read_csv(filename,sep = "|", header=0)
    #pname = filename.replace( './try/','')
    pname = filename.replace( path_replace,'')
    pname = pname.replace( '.psv','')
    dff['patient'] = pname
   # df = df.append(dff, ignore_index=True)
    #print( 'file: {}patient: {} shape: {}'.format(filename, pname, dff.shape) )
    return dff
	
