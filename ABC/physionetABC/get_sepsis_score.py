import pandas as pd
import xgboost as xgb
import copy
import numpy as np
'''
生成特征1的方法，index为选取的6个指标在原数据集中的位置
'''
def feature1(index,current_data):#HR:0;O2sat：1；Temp:2;SBP:3;DBP:5;Resp:6
    col_HR=current_data[:,index]
    # print(data.shape)
    # print(col_HR)
    flag=np.isnan(col_HR)#判断是否为nan
    # print(flag)
    try:
        not_null=np.argwhere(flag==False)#找出第一个不是nan的位置
        base_value=col_HR[not_null[0]]#基线值
    #     print(base_value)
        HR_f1=(col_HR-base_value)/base_value#计算HR的第一个特征：相对于base的变化率
        #HR_f1=np.round(HR_f1,4)#限制长度为4位小数，耗时较大
        HR_f1=HR_f1.reshape((current_data.shape[0],1))#reshape
#         print(HR_f1)
    except:#判断全为nan的情况
        HR_f1=np.full(col_HR.shape,np.nan)#填充np.nan
        HR_f1=HR_f1.reshape((current_data.shape[0],1))
#         print(HR_f1)
#     d1=np.concatenate((current_data,HR_f1),axis=1)#将新生成的一列拼上去
    return HR_f1
'''
生成特征2的方法，index为选取的6个指标在原数据集中的位置
'''
def feature2(index,current_data):
    col_HR=current_data[:,index]
    forward_data=[]
    for i in range(len(col_HR)):
        if i==0:
            forward=col_HR[0]
        else:
            cur=col_HR[:i]
    #         print(cur)
            flag=np.isnan(cur)#判断是否为nan
    #         print(cur)
    #         print(flag)
            try:
                not_nan=np.argwhere(flag==False)#不是nan的位置
                forward=cur[not_nan[-1]][0]#最近的一个不是nan的数
            except:
                forward=np.nan#全部为nan时，填成nan
        forward_data.append(forward)
    # forward_data=np.array(forward_data)
    change_rate=(col_HR-forward_data)/forward_data
    change_rate=change_rate.reshape((current_data.shape[0],1))
    return change_rate
#     d2=np.concatenate((current_data,change_rate),axis=1)#将新生成的一列拼上去
'''
paper特征离散化
五个特征：HR[60-80]、SBP[110-140]、DBP[70-90]、Resp[7-14]、Temp[36-38] 对应index【0；3；5；6；2】#注意该Index和读入数据index保持一致
取值-->区间内：Normal；小于区间：Low；大于区间：High【Normal:0,Low:1,High:2】
单位均一致
'''
def feature3(index,current_data):
    region_dict={0:[60,80],3:[110,140],4:[70,90],5:[7,14],2:[36,38]}#特征区间字典
    region=region_dict[index]#当前特征的区间
#     print(region)
    col_cur=current_data[:,index]
    new=np.full(col_cur.shape,np.nan)#新开辟一个空间，填充为np.nan
    new[col_cur>=region[0] & (col_cur<=region[1])]=0
    new[col_cur<region[0]]=1
    new[col_cur>region[1]]=2
    new=new.reshape((col_cur.shape[0],1))
    return new
'''
paper特征梯度
五个特征：HR[10]、SBP[10]、DBP[10]、Resp[3]、Temp[0.5] 对应index【0；3；5；6；2】#注意该Index和读入数据index保持一致
取值-->稳定：变化值绝对值小于等于value；增加：变化值大于value；减小：变化值为负，且绝对值大于value【稳定：0；增加：1；减小：2】
'''
def feature4(index,current_data):
    threshold_dict={0:10,3:10,4:10,5:3,2:0.5}#临界值字典
    threshold=threshold_dict[index]#当前特征的临界值
    col_HR=current_data[:,index]
    forward_data=[]
    for i in range(len(col_HR)):
        if i==0:
            forward=col_HR[0]
        else:
            cur=col_HR[:i]
    #         print(cur)
            flag=np.isnan(cur)#判断是否为nan
    #         print(cur)
    #         print(flag)
            try:
                not_nan=np.argwhere(flag==False)#不是nan的位置
                forward=cur[not_nan[-1]][0]#最近的一个不是nan的数
            except:
                forward=np.nan#全部为nan时，填成nan
        forward_data.append(forward)
    # forward_data=np.array(forward_data)
    change_volume=col_HR-forward_data
    new=np.full(change_volume.shape,np.nan)#新开辟一个空间，填充为np.nan
    new[np.fabs(change_volume)<threshold]=0
    new[change_volume>threshold]=1
    new[-change_volume<-threshold]=2
    new=new.reshape((col_HR.shape[0],1))
    return new
'''
window_size范围内特征的最大值
'''
def window_feature_1(size,index,current_data):
    col_cur=current_data[:,index]
    new=np.full(col_cur.shape,np.nan)#注意，前size-1个数据是nan
#     print(new[3])
    for i in range(len(col_cur)-size+1):
        cur_window=col_cur[i:size+i]#当前窗口内的数据
        new[i+size-1]=np.nanmax(cur_window)#当前窗口内的数据最大值填充
    new=new.reshape((col_cur.shape[0],1))
#     print(new)
    return new
'''
window_size范围内特征的最小值
'''
def window_feature_2(size,index,current_data):
    col_cur=current_data[:,index]
    new=np.full(col_cur.shape,np.nan)#注意，前size-1个数据是nan
#     print(new[3])
    for i in range(len(col_cur)-size+1):
        cur_window=col_cur[i:size+i]#当前窗口内的数据
        new[i+size-1]=np.nanmin(cur_window)#当前窗口内的数据最大值填充
    new=new.reshape((col_cur.shape[0],1))
#     print(new)
    return new
'''
window_size范围内特征的平均值
'''
def window_feature_3(size,index,current_data):
    col_cur=current_data[:,index]
    new=np.full(col_cur.shape,np.nan)#注意，前size-1个数据是nan
#     print(new[3])
    for i in range(len(col_cur)-size+1):
        cur_window=col_cur[i:size+i]#当前窗口内的数据
        new[i+size-1]=np.nanmean(cur_window)#当前窗口内的数据最大值填充
    new=new.reshape((col_cur.shape[0],1))
#     print(new)
    return new
'''
window_size范围内特征的标准差
'''
def window_feature_4(size,index,current_data):
    col_cur=current_data[:,index]
    new=np.full(col_cur.shape,np.nan)#注意，前size-1个数据是nan
#     print(new[3])
    for i in range(len(col_cur)-size+1):
        cur_window=col_cur[i:size+i]#当前窗口内的数据
        new[i+size-1]=np.nanstd(cur_window)#当前窗口内的数据最大值填充
    new=new.reshape((col_cur.shape[0],1))
#     print(new)
    return new
'''
读取numpy格式的数据
'''
#注意：本地测的时候连label一起读进来
def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    # if column_names[-1] == 'SepsisLabel':
    #     column_names = column_names[:-1]
    #     data = data[:, :-1]

    return data
'''
原始7个特征+70个新特征【7个特征+12+10+5窗口特征+3窗口特征】
'''
def feature_v13(data):
    d1=copy.deepcopy(data)
    for i in [0,1,2,3,4,5]:#注意：Glucose的index是21
        cur_f1=feature1(i,data)
        cur_f2=feature2(i,data)
        d1=np.concatenate((d1,cur_f1),axis=1)
        d1=np.concatenate((d1,cur_f2),axis=1)
        
    for i in [0,2,3,4,5]:
        cur_f3=feature3(i,data)
        cur_f4=feature4(i,data)
        d1=np.concatenate((d1,cur_f3),axis=1)
        d1=np.concatenate((d1,cur_f4),axis=1)
    for i in [0,1,2,3,4,5]:
        cur_f5=window_feature_1(5,i,data)
        cur_f6=window_feature_2(5,i,data)
        cur_f7=window_feature_3(5,i,data)
        cur_f8=window_feature_4(5,i,data)
        d1=np.concatenate((d1,cur_f5),axis=1)
        d1=np.concatenate((d1,cur_f6),axis=1)
        d1=np.concatenate((d1,cur_f7),axis=1)
        d1=np.concatenate((d1,cur_f8),axis=1)
    for i in [0,1,2,3,4,5]:
        cur_f9=window_feature_1(3,i,data)
        cur_f10=window_feature_2(3,i,data)
        cur_f11=window_feature_3(3,i,data)
        cur_f12=window_feature_4(3,i,data)
        d1=np.concatenate((d1,cur_f9),axis=1)
        d1=np.concatenate((d1,cur_f10),axis=1)
        d1=np.concatenate((d1,cur_f11),axis=1)
        d1=np.concatenate((d1,cur_f12),axis=1)
#     fake_pattern=feature_pattern_fake(63,57)#造fake_pattern特征；注意SBP_f3的index是57；Glucose_f3的index是63
#     d1=np.concatenate((d1,fake_pattern),axis=1)
#     qsofa=feature_qsofa(3,6)#造qsofa特征，SBP的index为3，Resp的index为6
#     d1=np.concatenate((d1,qsofa),axis=1)
    # print('耗时:%s'%(time.time()-start))
    sel_columns=['HR', 'O2Sat','Temp','SBP', 'DBP', 'Resp', 'ICULOS','HR_f1','HR_f2','O2sat_f1','O2sat_f2'
                 ,'Temp_f1','Temp_f2','SBP_f1','SBP_f2','DBP_f1','DBP_f2','Resp_f1','Resp_f2',
                 'HR_f3','HR_f4','Temp_f3','Temp_f4','SBP_f3','SBP_f4','DBP_f3','DBP_f4','Resp_f3','Resp_f4',
                 'HR_f5','HR_f6','HR_f7','HR_f8','O2sat_f5','O2sat_f6','O2sat_f7','O2sat_f8',
                 'Temp_f5','Temp_f6','Temp_f7','Temp_f8','SBP_f5','SBP_f6','SBP_f7','SBP_f8',
                 'DBP_f5','DBP_f6','DBP_f7','DBP_f8','Resp_f5','Resp_f6','Resp_f7','Resp_f8','HR_f9','HR_f10','HR_f11','HR_f12',
             'O2sat_f9','O2sat_f10','O2sat_f11','O2sat_f12','Temp_f9','Temp_f10','Temp_f11','Temp_f12',
             'SBP_f9','SBP_f10','SBP_f11','SBP_f12','DBP_f9','DBP_f10','DBP_f11','DBP_f12',
             'Resp_f9','Resp_f10','Resp_f11','Resp_f12']
    data1=xgb.DMatrix(d1,feature_names=sel_columns)
    return data1
    
'''
data:每个人对应的文件路径
model:训练好的xgb模型的路径
'''
def get_sepsis_score(data,model):
#     loaded_model = joblib.load(model)#load xgb model
    #loaded_model=load_sepsis_model()
    #raw_cur_test = load_challenge_data(data)#读取当前人的数据
#     print(raw_cur_test)
    org_feature = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
               'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
               'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
               'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
               'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
               'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
               'HospAdmTime', 'ICULOS']
    index=[0,1,2,3,5,6,39]#返回需要的7列数据
    cur_test=feature_v13(data[:,index])#丢弃指定列，以便和xgboost保持一致,并且新增特征工程列
#     print(cur_test)
    probs=model.predict(cur_test)#预测的概率
    y_pred = (probs >= 0.5)*1#预测的label
    #这里指定threshold
    return probs[-1],y_pred[-1]
def load_sepsis_model():
    loaded_model = xgb.Booster(model_file='xgboost_feature7_70_sample_utility_2000rounds.model')#load xgb model
    return loaded_model
def getZip(index):
    '''
    读取pkl文件  包含测试集路径信息
    i表示测试集编号【i=1,2,3,4】
    model_path:训练好的模型路径
    ！！！注意：每次运行之前要将labels和predictions目录下文件清空
    '''
    with open('test%s.pkl'%index,'rb') as f:
        data=pickle.load(f)
    MODEL=load_sepsis_model()
    '''
    对每个人的数据进行处理，生成txt
    '''
    for i in data:
        # print(i)
        name = i[10:17]
        tmp_data = load_challenge_data(i)
        single_org = tmp_data[:,-1]
        single_score, single_label = get_sepsis_score(tmp_data, MODEL)
        # save results
        with open('labels/%s.txt'%name, 'w') as f:
            f.write('SepsisLabel\n')
            if len(single_org) != 0:
                for l in list(single_org):
                    #print(l, list(single_org))
                    f.write('%d\n' % l)
                f.close()

        with open('predictions/%s.txt'%name, 'w') as f:
            f.write('PredictedProbability|PredictedLabel\n')
            if len(single_score) != 0:
                for (s, l) in zip(list(single_score), list(single_label)):
                    f.write('%g|%d\n' % (s, l))
            f.close()
    '''
    生成zip文件
    '''
    with zipfile.ZipFile('labels_%s.zip'%index, 'w') as z:
        for i in data:
            name = i[10:17]
            z.write('labels/%s.txt'%name)
    with zipfile.ZipFile('predictions_%s.zip'%index, 'w') as z:
        for i in data:
            name = i[10:17]
            z.write('predictions/%s.txt'%name)

    