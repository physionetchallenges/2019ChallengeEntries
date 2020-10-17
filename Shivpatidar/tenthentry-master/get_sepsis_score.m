function [scores, labels]  = get_sepsis_score(rawdata,model)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here



data=rawdata(:,[1:36 39:40]);

xxx=find(data(:,4)<data(:,6));

for ii=1:length(xxx)
    uu=data(xxx(ii),4);
    vv=data(xxx(ii),6);
    data(xxx(ii),6)=uu;
    data(xxx(ii),4)=vv;
end

[data] = fillthenans1(data);
data(isnan(data))=0;
feat=propfeat(data);
data1=[data feat];
x=[5,4,4,6,-2,2,-1,4,8,0,8,4,0,1,2,7,6,0,-2,-3,-1,0,2,5,4,2,5,3,3,-1,3,7,4,2,7,1,-2,-1,4,4,3,2,1,0,1,0,1,1,0,1,0,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,0,383];
power = x(1:42);
data2=data1.^power;

[labels,scores1] = predict(model,[data1 data1(:,1)./(data1(:,4).*data1(:,35)) data2(:,logical(x(42+1:end-1))) ]);
 
scores=scores1(:,2);





 
 
end




function [outputdata] = fillthenans1(inputdata)
 outputdataf=[];
 if(size(inputdata,1)==1)
     outputdataf=inputdata;
 else
        
        for uu=1:38
        datata=fillmissing(inputdata(:,uu)','linear',2,'EndValues','nearest');
        outputdataf(:,uu)=datata';
        end
        
 end
outputdata=outputdataf(end,:);

end

function feat=propfeat(data)

feat11=data(:,1)./(data(:,35).^2);%correct
feat12=data(:,6)./(data(:,36).^2);
feat13=data(:,8)./(data(:,31).^2);
feat14=data(:,35)./(data(:,36).^2);




feat=[feat11 feat12 feat13 feat14  ];
feat(isinf(feat))=100;



end

