import numpy as np
import glob
import copy
import os
import sys

os.system('rm -rf utility_figure_data_binary')
os.system('mkdir utility_figure_data_binary')

result_all=glob.glob('gs/*')
for the_file in result_all:
    name=copy.copy(the_file)
    name=name.replace('gs','result')
    RESULT=open(name,'r')
    result=RESULT.readline()
    GS=open(the_file,'r')
    gsline=GS.readline()
    gs=[]
    for gsline in GS:
        gsline=gsline.rstrip()
        gstable=gsline.split('|')
        gs.append(gstable[-1])
    GS.close()
    pos_start=1000000
    try:
        the_index=gs.index('1')
        pos_start=the_index-6
    except:
        pass

    ### create scoring cost
    scoring_cost=[]
    i=0
    while ((i<pos_start-7) and (i<len(gs))):
        val=-0.05
        scoring_cost.append(val)
        i=i+1
    while ((i<(pos_start)) and (i<len(gs))):
        val=(1/7.0*(7-pos_start+i))
        scoring_cost.append(val)
        i=i+1
    while ((i<(pos_start+9)) and (i<len(gs))):
        val=(1.0-1/9.0*(i-pos_start)+2.0/9.0*(i-pos_start))
        scoring_cost.append(val)
        i=i+1
    while (i<len(gs)):
        if (pos_start==1000000):
            val=-0.05
            scoring_cost.append(val)
        else:
            val=2
            scoring_cost.append(val)
        i=i+1

    pred=[]
    for line in RESULT:
        line=line.strip()
        resulttable=line.split('|')
        print(resulttable[0])
        pred.append(float(resulttable[0]))
    RESULT.close()
    GS.close()

    nametable=name.split('/')
    NEW=open(('utility_figure_data_binary/'+nametable[-1]),'w')
    i=0
    while (i<len(gs)):
        print(pred[i],scoring_cost[i])
        if (pred[i]-0.5)>0:
            k=1
        else:
            k=-1
        val=k*scoring_cost[i]
        NEW.write('%d\t%d\t%.4f\t%.4f\n' % (i,int(gs[i]),pred[i],val))

        i=i+1
    NEW.close()




    
