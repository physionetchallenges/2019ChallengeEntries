function [score, label] = get_sepsis_score(data, model)
[s1,s2]=size(data);
% outs=0;

cEns1=model{1,1};
cEns2=model{2,1};
cEns3=model{3,1};
cEns6=model{4,1};
cEns9=model{5,1};
cEns12=model{6,1};
cEns16=model{7,1};
cEns20=model{8,1};
cEns24=model{9,1};
cEns30=model{10,1};
cEns36=model{11,1};
cEns43=model{12,1};
cEns51=model{13,1};
cEns61=model{14,1};
cEns101=model{15,1};

cEns1_2=model{1,2};
cEns2_2=model{2,2};
cEns3_2=model{3,2};
cEns6_2=model{4,2};
cEns9_2=model{5,2};
cEns12_2=model{6,2};
cEns16_2=model{7,2};
cEns20_2=model{8,2};
cEns24_2=model{9,2};
cEns30_2=model{10,2};
cEns36_2=model{11,2};
cEns43_2=model{12,2};
cEns51_2=model{13,2};
cEns61_2=model{14,2};
cEns101_2=model{15,2};

cEns1_3=model{1,3};
cEns2_3=model{2,3};
cEns3_3=model{3,3};
cEns6_3=model{4,3};
cEns9_3=model{5,3};
cEns12_3=model{6,3};
cEns16_3=model{7,3};
cEns20_3=model{8,3};
cEns24_3=model{9,3};
cEns30_3=model{10,3};
cEns36_3=model{11,3};
cEns43_3=model{12,3};
cEns51_3=model{13,3};
cEns61_3=model{14,3};
cEns101_3=model{15,3};

cEns1_4=model{1,4};
cEns2_4=model{2,4};
cEns3_4=model{3,4};
cEns6_4=model{4,4};
cEns9_4=model{5,4};
cEns12_4=model{6,4};
cEns16_4=model{7,4};
cEns20_4=model{8,4};
cEns24_4=model{9,4};
cEns30_4=model{10,4};
cEns36_4=model{11,4};
cEns43_4=model{12,4};
cEns51_4=model{13,4};
cEns61_4=model{14,4};
cEns101_4=model{15,4};

xx1=model{1,5};
xx2=model{2,5};
xx3=model{3,5};
xx6=model{4,5};
xx9=model{5,5};
xx12=model{6,5};
xx16=model{7,5};
xx20=model{8,5};
xx24=model{9,5};
xx30=model{10,5};
xx36=model{11,5};
xx43=model{12,5};
xx51=model{13,5};
xx61=model{14,5};
xx101=model{15,5};

fxx1_1=1:length(xx1);fxx1_1(xx1==0)=[];
fxx2_2=1:length(xx2);fxx2_2(xx2==0)=[];
fxx3_5=1:length(xx3);fxx3_5(xx3==0)=[];
fxx6_8=1:length(xx6);fxx6_8(xx6==0)=[];
fxx9_11=1:length(xx9);fxx9_11(xx9==0)=[];
fxx24_29=1:length(xx24);fxx24_29(xx24==0)=[];
fxx36_42=1:length(xx36);fxx36_42(xx36==0)=[];
fxx43_50=1:length(xx43);fxx43_50(xx43==0)=[];
fxx51_60=1:length(xx51);fxx51_60(xx51==0)=[];
fxx61_100=1:length(xx61);fxx61_100(xx61==0)=[];
fxx12_15=1:length(xx12);fxx12_15(xx12==0)=[];
fxx16_19=1:length(xx16);fxx16_19(xx16==0)=[];
fxx20_23=1:length(xx20);fxx20_23(xx20==0)=[];
fxx30_35=1:length(xx30);fxx30_35(xx30==0)=[];
fxx101_1000=1:length(xx101);fxx101_1000(xx101==0)=[];

cdata=createCorrespondingFeaturesNewBasicExtended_Submission1(data);
%13,5,8,11,7,6
% if selectCase==13

dataOut16=zeros(size(data));
dataOut16(isnan(data))=1;
cdata=[cdata,dataOut16];

cdata2=createPreviousHour(cdata(:,1:38),1);
cdata=[cdata,cdata2];
dataOut5=createAllFeaturesC(data);
cdata=[cdata,dataOut5];

dataOut9=findDiffHistory(cdata(:,1:39));
dataOut12=findMinHistory(dataOut9);
cdata=[cdata,dataOut12];


cdata(isnan(cdata))=0;
kii=1;
outs1=0;outs2=0;outs3=0;outs4=0;
if s1==1
    cdata1=cdata;%cdata1=cdata1;
    outs1=cEns1.predict(cdata1);outs2=cEns1_2.predict(cdata1);
    outs3=cEns1_3.predict(cdata1(fxx1_1));outs4=cEns1_4.predict(cdata1(fxx1_1));
    %     outs=cEns1.predict(cdata1);+cEns1_2.predict(cdata1);
end
if s1==2
    kii=2;
    cdata2=cdata;cdata2=cdata2(s1,:);
    %     outs=cEns2.predict(cdata2)+cEns2_2.predict(cdata2);
    outs1=cEns2.predict(cdata2);outs2=cEns2_2.predict(cdata2);
    outs3=cEns2_3.predict(cdata2(fxx2_2));outs4=cEns2_4.predict(cdata2(fxx2_2));
    
end
if s1>=3 && s1<=5
    kii=3;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns3.predict(cdata3);outs2=cEns3_2.predict(cdata3);
    outs3=cEns3_3.predict(cdata3(fxx3_5));outs4=cEns3_4.predict(cdata3(fxx3_5));
    
end
if s1>=6 && s1<=8
    kii=4;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns6.predict(cdata3);outs2=cEns6_2.predict(cdata3);
    outs3=cEns6_3.predict(cdata3(fxx6_8));outs4=cEns6_4.predict(cdata3(fxx6_8));
    
end
if s1>=9 && s1<=11
    kii=5;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns9.predict(cdata3);outs2=cEns9_2.predict(cdata3);
    outs3=cEns9_3.predict(cdata3(fxx9_11));outs4=cEns9_4.predict(cdata3(fxx9_11));
    
end
if s1>=12 && s1<=15
    kii=6;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns12.predict(cdata3);outs2=cEns12_2.predict(cdata3);
    outs3=cEns12_3.predict(cdata3(fxx12_15));outs4=cEns12_4.predict(cdata3(fxx12_15));
    
end
if s1>=16 && s1<=19
    kii=7;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns16.predict(cdata3);outs2=cEns16_2.predict(cdata3);
    outs3=cEns16_3.predict(cdata3(fxx16_19));outs4=cEns16_4.predict(cdata3(fxx16_19));
    
end
if s1>=20 && s1<=23
    kii=8;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns20.predict(cdata3);outs2=cEns20_2.predict(cdata3);
    outs3=cEns20_3.predict(cdata3(fxx20_23));outs4=cEns20_4.predict(cdata3(fxx20_23));
    
end
if s1>=24 && s1<=29
    kii=9;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns24.predict(cdata3);outs2=cEns24_2.predict(cdata3);
    outs3=cEns24_3.predict(cdata3(fxx24_29));outs4=cEns24_4.predict(cdata3(fxx24_29));
    
end
if s1>=30 && s1<=35
    kii=10;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns30.predict(cdata3);outs2=cEns30_2.predict(cdata3);
    outs3=cEns30_3.predict(cdata3(fxx30_35));outs4=cEns30_4.predict(cdata3(fxx30_35));
    
end
if s1>=36 && s1<=42
    kii=11;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns36.predict(cdata3);outs2=cEns36_2.predict(cdata3);
    outs3=cEns36_3.predict(cdata3(fxx36_42));outs4=cEns36_4.predict(cdata3(fxx36_42));
    
end
if s1>=43 && s1<=50
    kii=12;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns43.predict(cdata3);outs2=cEns43_2.predict(cdata3);
    outs3=cEns43_3.predict(cdata3(fxx43_50));outs4=cEns43_4.predict(cdata3(fxx43_50));
    
end
if s1>=51 && s1<=60
    kii=13;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns51.predict(cdata3);outs2=cEns51_2.predict(cdata3);
    outs3=cEns51_3.predict(cdata3(fxx51_60));outs4=cEns51_4.predict(cdata3(fxx51_60));
    
end
if s1>=61 && s1<=100
    kii=14;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns61.predict(cdata3);outs2=cEns61_2.predict(cdata3);
    outs3=cEns61_3.predict(cdata3(fxx61_100));outs4=cEns61_4.predict(cdata3(fxx61_100));
    
end
if s1>=101
    kii=15;
    cdata3=cdata;cdata3=cdata3(s1,:);
    outs1=cEns101.predict(cdata3);outs2=cEns101_2.predict(cdata3);
    outs3=cEns101_3.predict(cdata3(fxx101_1000));outs4=cEns101_4.predict(cdata3(fxx101_1000));
    
end
outs=5*outs1+4*outs2+6*outs4+6*outs3;
outs=outs/21;

if outs<0
    outs=0;
end
if outs>0.999
    outs=0.999;
end
outsP=outs;
% threshs=[0.170000000000000,0.225000000000000,0.112500000000000,0.260000000000000,0.267500000000000,0.290000000000000,0.232500000000000,0.212500000000000,0.240000000000000,0.0950000000000000,0.222500000000000,0.222500000000000,0.187500000000000,0.157500000000000,0.110000000000000,0.100000000000000,0.100000000000000];
% threshs=[0.100000000000000,0.362500000000000,0.290000000000000,0.540000000000000,0.355000000000000,0.420000000000000,0.455000000000000,0.262500000000000,0.240000000000000,0.432500000000000,0.400000000000000,0.310000000000000,0.325000000000000,0.197500000000000,0.215000000000000,0.0100000000000000,0.205000000000000];
threshs=[0.375000000000000,0.460000000000000,0.350000000000000,0.385000000000000,0.400000000000000,0.325000000000000,0.330000000000000,0.305000000000000,0.305000000000000,0.400000000000000,0.340000000000000,0.345000000000000,0.330000000000000,0.405000000000000,0.340000000000000,0.350000000000000,0.330000000000000,0.335000000000000,0.305000000000000,0.315000000000000,0.295000000000000,0.325000000000000,0.270000000000000,0.305000000000000,0.350000000000000,0.315000000000000,0.320000000000000,0.300000000000000,0.320000000000000,0.375000000000000,0.350000000000000,0.335000000000000,0.315000000000000,0.340000000000000,0.350000000000000,0.380000000000000,0.335000000000000,0.335000000000000,0.375000000000000,0.350000000000000,0.300000000000000,0.400000000000000,0.410000000000000,0.430000000000000,0.380000000000000,0.395000000000000,0.325000000000000,0.325000000000000,0.335000000000000,0.355000000000000,0.250000000000000,0.250000000000000,0.330000000000000,0.220000000000000,0.215000000000000,0.260000000000000,0.285000000000000,0.270000000000000,0.190000000000000,0.285000000000000,0.0700000000000000,0.110000000000000,0.0450000000000000,0.0600000000000000,0.0700000000000000,0.0400000000000000,0.125000000000000,0.180000000000000,0.0800000000000000,0.00500000000000000,0.0800000000000000,0.175000000000000,0.0850000000000000,0.0600000000000000,0.200000000000000,0.100000000000000,0.120000000000000,0.165000000000000,0.160000000000000,0.250000000000000,0.0100000000000000,0.125000000000000,0.305000000000000,0.480000000000000];
outs=applyThresholds(outs,threshs,s1);

% % score=outsP/threshsD(kii);
% % score=outsP;
if outs>0.99
    score=0.5+outsP;
    if score>0.9
        score=0.9;
    end
else
    score=outsP;
    if score>0.5
        score=0.49;
    end
end
label=outs;

end

function outsP=applyThresholds(outs,threshs,s1)
% global ffx2_2 ffx3_5  ffx6_8 ffx9_11 ffx12_15 ffx16_19 ffx20_23 ffx24_29 ffx30_35 ffx36_42 ffx43_50 ffx51_60 ffx61_100 ffx101_1000
outsP=zeros(length(outs),1);
if s1<=80
    if outs>threshs(s1)
        outsP=1;
    end
end

if s1>80 && s1<=90
    if outs>threshs(81)
        outsP=1;
    end    
end


if s1>=91 && s1<=100
    if outs>threshs(82)
        outsP=1;
    end    
end


if s1>=101 && s1<=200
    if outs>threshs(83)
        outsP=1;
    end    
end

if s1>=201
    if outs>threshs(84)
        outsP=1;
    end    
end
end

function outsP=applyThresholdsPrevious(outs,threshs,s1)
% global ffx2_2 ffx3_5  ffx6_8 ffx9_11 ffx12_15 ffx16_19 ffx20_23 ffx24_29 ffx30_35 ffx36_42 ffx43_50 ffx51_60 ffx61_100 ffx101_1000
s1a=length(outs);
outsP=0;
if s1==1
    if outs(1)>threshs(1)
        outsP(1)=1;
    end
end
if s1==2
    if outs>threshs(2)
        outsP=1;
    end
end
if s1>=3 && s1<=5
    if outs>threshs(3)
        outsP=1;
    end
    
end

if s1>=6 && s1<=8
    
    if outs>threshs(4)
        outsP=1;
    end
    
end

if s1>=9 && s1<=11
    
    if outs>threshs(5)
        outsP=1;
    end
    
end

if s1>=12 && s1<=15
    
    if outs>threshs(6)
        outsP=1;
    end
    
end

if s1>=16 && s1<=19
    
    if outs>threshs(7)
        outsP=1;
    end
    
end

if s1>=20 && s1<=23
    
    if outs>threshs(8)
        outsP=1;
    end
    
end

if s1>=24 && s1<=29
    
    if outs>threshs(9)
        outsP=1;
    end
    
end

if s1>=30 && s1<=35
    
    if outs>threshs(10)
        outsP=1;
    end
    
end

if s1>=36 && s1<=42
    
    if outs>threshs(11)
        outsP=1;
    end
    
end

if s1>=43 && s1<=50
    
    if outs>threshs(12)
        outsP=1;
    end
    
end

if s1>=51 && s1<=60
    
    if outs>threshs(13)
        outsP=1;
    end
    
end

if s1>=61 && s1<=80
    
    if outs>threshs(14)
        outsP=1;
    end
    
end

if s1>=81 && s1<=100
    
    if outs>threshs(15)
        outsP=1;
    end
    
end

if s1>=101 && s1<=200
    
    if outs>threshs(16)
        outsP=1;
    end
    
end

if s1>=201
    
    if outs>threshs(17)
        outsP=1;
    end
    
end
end


function dataExtended=createCorrespondingFeaturesNewBasicExtended_Submission1(data)
minV=[20,20,20.9000000000000,20,20,20,1,10,-32,0,-50,6.62000000000000,10,23,3,1,7,1,26,0.100000000000000,0.0100000000000000,10,0.200000000000000,0.200000000000000,0.200000000000000,1,0.100000000000000,0.0100000000000000,5.50000000000000,2.20000000000000,12.5000000000000,0.100000000000000,34,1,14,0,0,0,-5366.86000000000,1];
maxV=[260,80,29.1000000000000,280,280,280,99,90,132,55,4050,1.31000000000000,90,77,9958,267,3826,26.9000000000000,119,46.5000000000000,37.4900000000000,978,30.8000000000000,9.60000000000000,18.6000000000000,26.5000000000000,49.5000000000000,439.990000000000,66.2000000000000,29.8000000000000,237.500000000000,439.900000000000,1726,2321,86,1,1,1,5390.85000000000,335];
[s1,s2]=size(data);
%0:0, 1:median,2:min
approaches=[1,1,1,1,1,2,1,0,1,1,0,1,1,2,0,1,0,0,1,2,0,0,0,0,1,2,2,2,0,0,0,2,1,0,1,0,1,1,1,2,1];
% approaches=3*ones(1,40);
approaches=[1;3;1;1;3;1;1;4;2;3;4;0;0;0;0;1;0;3;3;2;0;1;4;2;2;4;0;3;3;3;0;2;2;3;0;0;4;0;0;0;0]';
approaches(approaches==4)=3;
medVals=[0.417159763313609,0.980769230769231,0.648955855141422,0.505494505494506,0.395604395604396,0.348901098901099,0.362859362859363,0.427350427350427,0.411421911421912,0.566433566433566,0.240360873694207,0.671168526130358,0.495726495726496,0.975024975024975,0.233163903780493,0.276865456640738,0.244038763118742,0.436659994280812,0.754363283775049,0.244003308519438,0.236719535465868,0.326726443290861,0.278221778221778,0.366987179487179,0.367245657568238,0.320754716981132,0.241647241647242,0.230891611172981,0.530560074366721,0.450180691791430,0.293927125506073,0.249130047038663,0.315892682057224,0.294733702316641,0.670035778175313,0.999999999999999,0.230769230769231,0.999999999999999,0.995713534543218,0.239954075774971,1.23281490055539];
minVals=[0.239644970414201,0.230769230769231,0.230769230769231,0.236263736263736,0.230769230769231,0.230769230769231,0.230769230769231,0.230769230769231,0.248251748251748,0.230769230769231,0.240265906932574,0.236641221374046,0.230769230769231,0.240759240759241,0.230923725802215,0.230769230769231,0.231573444851019,0.230769230769231,0.489334195216548,0.230769230769231,0.230769230769231,0.233915368884694,0.233266733266733,0.230769230769231,0.230769230769231,0.242380261248186,0.230769230769231,0.230769230769231,0.272600511271206,0.233350542075374,0.230769230769231,0.230769230769231,0.231214903289063,0.231100652901601,0.230769230769231,0.230769230769231,0.230769230769231,0.230769230769231,0.230769230769231,0.230769230769231,0.233065442020666];
data=data-repmat(minV,[s1 1]);
data=data./repmat(maxV+0.000000000000001,[s1 1]);
data=data+.3;
% data(isnan(data)==1)=0;
% for i=1:s1
%     data(1:i,:) = fillmissing(data(1:i,:),'previous');
% end
% data = fillmissing(data,'previous');
data=data/1.3;
for i=1:s2
    ccol=data(:,i);
    if approaches(i)==0
        ccol(isnan(ccol))=0;
    end
    if approaches(i)==1
        ccol(isnan(ccol))=medVals(i);
    end
    if approaches(i)==2
        ccol(isnan(ccol))=minVals(i);
    end
    if approaches(i)==3
        ccol(isnan(ccol))=2*medVals(i);
    end
    if approaches(i)==4
        ccol(isnan(ccol))=4*medVals(i);
    end
    data(:,i)=ccol;
end
data(isnan(data)==1)=0;

data(:,end+1)=data(:,end)+data(:,end-1);

dataExtended=zeros([s1 s2*5+5]);
for i=1:s1
    cdata=data(1:i,:);
    if i==1
        cdata=[cdata;cdata];
    end
    mins=min(cdata);maxs=max(cdata);stds=std(cdata);means=mean(cdata);
    dataExtended(i,:)=[cdata(i,:),mins,maxs,means,stds];
end
dataExtended=dataExtended(:,:);
end




function dataOut=createPreviousHour(cdata,hr)
[s1,s2]=size(cdata);
dataOut=cdata;
cdata(isnan(cdata))=0;
for i=hr+1:s1
    dataOut(i,:)=cdata(i-hr,:);
end
end


function dataOut=findMeanHistory(cdata)
[s1,s2]=size(cdata);
dataOut=cdata;
for i=2:s1
    dataOut(i,:)=nanmean(cdata(1:i,:),1);
end
end

function dataOut=findStdHistory(cdata)
[s1,s2]=size(cdata);
dataOut=cdata;
for i=2:s1
    dataOut(i,:)=nanstd(cdata(1:i,:),1);
end
end

function dataOut=findMaxHistory(cdata)
[s1,s2]=size(cdata);
dataOut=cdata;
for i=2:s1
    dataOut(i,:)=nanmax(cdata(1:i,:),[],1);
end
end

function dataOut=findMedianHistory(cdata)
[s1,s2]=size(cdata);
dataOut=cdata;
for i=2:s1
    dataOut(i,:)=nanmedian(cdata(1:i,:),1);
end
end

function dataOut=findMinHistory(cdata)
[s1,s2]=size(cdata);
dataOut=cdata;
for i=2:s1
    dataOut(i,:)=nanmin(cdata(1:i,:),[],1);
end
end

function dataOut=findMomentHistory(cdata,deg)
[s1,s2]=size(cdata);
dataOut=cdata;
cdata(isnan(cdata))=0;
for i=2:s1
    dataOut(i,:)=moment(cdata(1:i,:),deg);
end
end

function dataOut=findDiffHistory(cdata)
[s1,s2]=size(cdata);
dataOut=cdata;
for i=2:s1
    zz=diff(cdata(1:i,:),1);
    dataOut(i,:)=zz(end,:);
end
end

function cdataA=createAllFeaturesC(cdata)
[s1,~]=size(cdata);
cdataA=zeros(s1,36);
cdataA(:,1)=featureHR(cdata(:,1));
cdataA(:,2)=featureO2(cdata(:,2));
cdataA(:,3)=featureTemp(cdata(:,3));
cdataA(:,4)=featureAge(cdata(:,35));
cdataA(:,5)=featureRR(cdata(:,7));
cdataA(:,6)=featureBP(cdata(:,4),cdata(:,6));
cdataA(:,7)=feature_4_SBP(cdata(:,4));
cdataA(:,8)=feature_5_MAP(cdata(:,5));
cdataA(:,9)=feature_6_DBP(cdata(:,6));
cdataA(:,10)=feature_8_ETCO2(cdata(:,8));
cdataA(:,11)=feature_9_BaseExcess(cdata(:,9));
cdataA(:,12)=feature_10_HCO3(cdata(:,10));
cdataA(:,13)=feature_11_FiO2(cdata(:,11));
cdataA(:,14)=feature_12_pH(cdata(:,12));
cdataA(:,15)=feature_13_PaCO2(cdata(:,13));
cdataA(:,16)=feature_14_SaO2(cdata(:,14));
cdataA(:,17)=feature_15_AST(cdata(:,15));
cdataA(:,18)=feature_16_BUN(cdata(:,16));
cdataA(:,19)=feature_17_Alkalinephos(cdata(:,17));
cdataA(:,20)=feature_18_Calcium(cdata(:,18));
cdataA(:,21)=feature_19_Chloride(cdata(:,19));
cdataA(:,22)=feature_20_Creatinine(cdata(:,20));
cdataA(:,23)=feature_21_Bilirubin_direct(cdata(:,21));
cdataA(:,24)=feature_22_Glucose(cdata(:,22));
cdataA(:,25)=feature_23_Lactate(cdata(:,23));
cdataA(:,26)=feature_24_Magnesium(cdata(:,24));
cdataA(:,27)=feature_25_Phosphate(cdata(:,25));
cdataA(:,28)=feature_26_Potassium(cdata(:,26));
cdataA(:,29)=feature_27_Bilirubin_total(cdata(:,27));
cdataA(:,30)=feature_28_TroponinI(cdata(:,28));
cdataA(:,31)=feature_29_Hematocrit(cdata(:,29));
cdataA(:,32)=feature_30_Hemoglobin(cdata(:,30));
cdataA(:,33)=feature_31_PTT(cdata(:,31));
cdataA(:,34)=feature_32_WBC(cdata(:,32));
cdataA(:,35)=feature_33_Fibrinogen(cdata(:,33));
cdataA(:,36)=feature_34_Platelets(cdata(:,34));
end


% The normal MAP range is between 70 and 100 mmHg
function newFeature=feature_5_MAP(FE)
newFeature=-1*ones(length(FE),1);
minv=70;maxv=110;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end

% ETCO2 35-45 mm Hg i
function newFeature=feature_8_ETCO2(FE)
newFeature=-1*ones(length(FE),1);
minv=10;maxv=70;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end

% The normal MAP range is between 70 and 100 mmHg
function newFeature=feature_4_SBP(FE)
newFeature=-1*ones(length(FE),1);
minv=100;maxv=180;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end

% The normal MAP range is between 70 and 100 mmHg
function newFeature=feature_6_DBP(FE)
newFeature=-1*ones(length(FE),1);
minv=60;maxv=120;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end

function newFeature=featureAge(FE)%35
newFeature=-1*ones(length(FE),1);
minv=40;maxv=60;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end


%Bilirubin_direct	Bilirubin direct (mg/dL) 0 to 0.4
function newFeature=feature_21_Bilirubin_direct(FE)
newFeature=-1*ones(length(FE),1);
minv=0.;maxv=20;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%Fibrinogen	(mg/dL) 150-400 
function newFeature=feature_33_Fibrinogen(FE)
newFeature=-1*ones(length(FE),1);
minv=300;maxv=400;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%WBC	Leukocyte count (count*10^3/µL) 4.00-11.0
function newFeature=feature_32_WBC(FE)
newFeature=-1*ones(length(FE),1);
minv=2;maxv=11;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%Platelets	(count*10^3/µL) platelets in the blood is 150 to 400 
function newFeature=feature_34_Platelets(FE)
newFeature=-1*ones(length(FE),1);
% minv=100/10;maxv=550/10;
minv=200;maxv=400;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%PTT	partial thromboplastin time (seconds)  60-70 seconds
function newFeature=feature_31_PTT(FE)
newFeature=-1*ones(length(FE),1);
minv=59;maxv=71;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% Hgb	Hemoglobin (g/dL) Male: 13.8 to 17.2 grams per deciliter (g/dL) Female: 12.1 to 15.1 g/dL
function newFeature=feature_30_Hemoglobin(FE)
newFeature=-1*ones(length(FE),1);
minv=12;maxv=17;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% Hct	Hematocrit (%) 45% to 52% for men and 37% to 48% for women
function newFeature=feature_29_Hematocrit(FE)
newFeature=-1*ones(length(FE),1);
minv=40;maxv=50;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% TroponinI	Troponin I (ng/mL) 0.00 – 0.40.
function newFeature=feature_28_TroponinI(FE)
newFeature=-1*ones(length(FE),1);
minv=1;maxv=4;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%Bilirubin_total	Total bilirubin (mg/dL) 0.2 to 1.2 mg/dL
function newFeature=feature_27_Bilirubin_total(FE)
newFeature=-1*ones(length(FE),1);
minv=0.5;maxv=2;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%Potassium	(mmol/L)  normally 3.6 to 5.2 millimoles per liter (mmol/L)
function newFeature=feature_26_Potassium(FE)
newFeature=-1*ones(length(FE),1);
minv=3.4;maxv=5.5;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%Phosphate	(mg/dL) 2.5 to 4.5 mg/dL
function newFeature=feature_25_Phosphate(FE)
newFeature=-1*ones(length(FE),1);
minv=2;maxv=6;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%Lactate	Lactic acid (mg/dL) 0.5-2 mmol/L =9-36 | conversion from mg/dl to mmol/L is to divide by 18. To convert mmol/L to mg/dl, multiply by 18. 
function newFeature=feature_23_Lactate(FE)
newFeature=-1*ones(length(FE),1);
minv=0.5;maxv=35;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%Glucose	Serum glucose (mg/dL)  between 3.9 and 7.1 mmol/L (70 to 130 mg/dL).
function newFeature=feature_22_Glucose(FE)
newFeature=-1*ones(length(FE),1);
minv=70.;maxv=130;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% Magnesium	(mmol/dL)  0.6-1.1 mmol/L /10 => 0.06 -0.11
function newFeature=feature_24_Magnesium(FE)
newFeature=-1*ones(length(FE),1);
minv=0.6;maxv=2;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end

% Chloride	(mmol/L)  chloride is as follows: Normal range: 98-106 mmol/L. Critical values: < 70 or >120 mmol/L.
function newFeature=feature_19_Chloride(FE)
newFeature=-1*ones(length(FE),1);
minv=98;maxv=106;  minv2=70;maxv2=120;
newFeature(FE>=minv & FE<=maxv)=1;
newFeature(FE>minv2 & FE<minv & FE>maxv & FE<maxv2)=0.3;
newFeature(FE<=minv2 | FE>=maxv2)=0;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
% newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%Creatinine	(mg/dL) 0.6 to 1.2 milligrams 
function newFeature=feature_20_Creatinine(FE)
newFeature=-1*ones(length(FE),1);
minv=0.6;maxv=1.2;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% SaO2	Oxygen saturation from arterial blood (%) SaO2:95-100
function newFeature=feature_14_SaO2(FE)
newFeature=-1*ones(length(FE),1);
minv=95;maxv=100;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%PaCO2	Partial pressure of carbon dioxide from arterial blood (mm Hg) 38 to 42 mm Hg
function newFeature=feature_13_PaCO2(FE)
newFeature=-1*ones(length(FE),1);
minv=38;maxv=42;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%pH	N/A 7.38 to 7.42
function newFeature=feature_12_pH(FE)
newFeature=-1*ones(length(FE),1);
minv=7.38;maxv=7.42;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% Calcium	(mg/dL) 8.5 to 10.5 mg/dl
function newFeature=feature_18_Calcium(FE)
newFeature=-1*ones(length(FE),1);
minv=8.5;maxv=10.5;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
%FiO2	Fraction of inspired oxygen (%) 0.24 – 0.44
function newFeature=feature_11_FiO2(FE)
newFeature=-1*ones(length(FE),1);
minv=0.24;maxv=0.44;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% HCO3	Bicarbonate (mmol/L)   22 to 28
function newFeature=feature_10_HCO3(FE)
newFeature=-1*ones(length(FE),1);
minv=20;maxv=30;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% BaseExcess	Measure of excess bicarbonate (mmol/L) (range –3 mmol/L to +3 mmol/L)
function newFeature=feature_9_BaseExcess(FE)
newFeature=-1*ones(length(FE),1);
newFeature(FE>=-3 & FE<=3)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>4 | FE<-4)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% AST	Aspartate transaminase (IU/L)  Males: 6-34 IU/L. Females: 8-40 IU/L.
function newFeature=feature_15_AST(FE)
newFeature=-1*ones(length(FE),1);
minv=7;maxv=37;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% BUN	Blood urea nitrogen (mg/dL) 7 to 20 mg/dL
function newFeature=feature_16_BUN(FE)
newFeature=-1*ones(length(FE),1);
minv=6;maxv=21;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end
% Alkalinephos	Alkaline phosphatase (IU/L) is 44 to 147 international units per liter (IU/L) 
function newFeature=feature_17_Alkalinephos(FE)
newFeature=-1*ones(length(FE),1);
minv=40;maxv=150;
newFeature(FE>=minv & FE<=maxv)=1;
% newFeature(FE<-3 & FE>=-4)=0.8;
% newFeature(FE<=4 & FE>3)=0.8;
newFeature(FE>maxv | FE<minv)=0;
newFeature(isnan(FE) | FE==0 | newFeature==-1)=0.5;
end

function newFeature=featureO2(O2)%2
newFeature=-1*ones(length(O2),1);
newFeature(O2>=90 & O2<100)=1;
newFeature(O2<90)=0;
newFeature(isnan(O2) | O2==0)=0.5;
end
function newFeature=featureTemp(temp)%3
newFeature=-1*ones(length(temp),1);
newFeature(temp>38 | temp<36)=0;
newFeature(temp>37.6 | temp<36.4 & newFeature==-1)=0.3;
newFeature(temp<=37.6 & temp>=36.4)=1;
newFeature(isnan(temp) | temp==0)=0.6;
end
function newFeature=featureRR(RR)%7
newFeature=-1*ones(length(RR),1);
newFeature(RR>=12 & RR<=20)=1;
newFeature(RR<12 | RR>20)=0;
newFeature(RR==0 | isnan(RR))=0.5;
end
function newFeature=featureBP(SBP,DBP)%4,5,6 %4,6
newFeature=0.5*ones(length(SBP),1);
newFeature(SBP<90 & DBP<60)=1;
newFeature(SBP>=90 & SBP<=120 & DBP>=60 & DBP<=80)=0.3;
newFeature(SBP>=120 & SBP<=140 & DBP>=80& DBP<=90)=0.6;
newFeature(SBP>140 | DBP>90)=1;
newFeature(isnan(SBP) | isnan(DBP) | SBP==0 | DBP==0)=0.5;
end
function newFeature=featureHR(HR)%1%35
newFeature=0.5*ones(length(HR),1);
newFeature( HR>=100 | HR<60)=0;
newFeature( HR<100 & HR>=60)=1;
% newFeature( (HR>=70 & HR<190) & AGE<10)=1;
% newFeature( (HR<70 | HR>=190) & AGE<10)=0;
newFeature(isnan(HR) | HR==0)=0.5;
end

