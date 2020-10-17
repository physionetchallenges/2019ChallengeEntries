clc;clear all;close all force; 

addpath('../../train_test_data2');
addpath('../utils');

load('data.mat');
load('train_test_ind.mat');



drawnow;


XTrain=data(r_train);
YTrain=labels(r_train);
is_septic=contain_sepsis(r_train);


XTest=data(r_test);
YTest=labels(r_test);

w=numel(cat(1,YTrain{:}))/sum(cat(1,YTrain{:}));




maxv= max(cat(1,XTrain{:}),[],1);
minv= min(cat(1,XTrain{:}),[],1);
XTrain=cellfun(@(x)  normalize015(x,minv,maxv),XTrain,'UniformOutput',false);
XTest=cellfun(@(x) normalize015(x,minv,maxv),XTest,'UniformOutput',false);

XTrain=cellfun(@(x) x(:,1:7) ,XTrain,'UniformOutput',false);
XTest=cellfun(@(x) x(:,1:7) ,XTest,'UniformOutput',false);


tic
[XTrain_peaces,YTrain_peaces]=divide_to_6_h_peaces(XTrain,YTrain);
toc




random_sample=randperm(length(YTrain_peaces),800000);


XTrain_peaces=XTrain_peaces(random_sample,:);
YTrain_peaces=YTrain_peaces(random_sample,:);


Mdl = fitcensemble(XTrain_peaces, YTrain_peaces, 'Method','Bag','Learners','tree','NumLearningCycles', 50);

save('model.mat','Mdl')

[XTest_peaces,YTest_peaces]=divide_to_6_h_peaces(XTest,YTest);


[~,tmp]=predict(Mdl,XTest_peaces);
vys=join_6_h_peaces(XTest,YTest,tmp);


x0=[0.5];
A=[];
b=[];
Aeq=[];
beq=[];
lb=[0];
ub=[1];
nonlcon=[];
x = ga(@(x) pred(x,YTest,vys),1,A,b,Aeq,beq,lb,ub,nonlcon,optimoptions('ga','Display','iter','MaxGenerations',25));


normalized_observed_utility=-pred(x,YTest,vys)


save('x.mat','x')


