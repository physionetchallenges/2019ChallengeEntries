function [score, label] = get_sepsis_score(x,model)

xname={'HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3',...
    'FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride',...
    'Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate',...
    'Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen',...
    'Platelets','Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS'}';    

t=x(:,end);
if t(end)>=60
    label=1;
    score=1.8363;
    return
end
nx=length(xname);

[x,xmeas]=lastsample(x,t);

%Demographic data
for c=35:39
    if isnan(x(c))
        x(c)=0;
    end
    xmeas(c)=t(end)-1;
end

vmeas=model.vmeas;
for i=1:length(vmeas)    
    if xmeas(i)>=vmeas(i)
        x(i)=NaN;
    end
end
res=model.res;
for i=1:length(res)
    if isnan(x(i)),continue,end    
    if isnan(res(i)),continue,end
    x(i)=round(res(i)*x(i))/res(i);
end

gc=model.gc;
v1=model.v1;
v2=model.v2;
gz=model.gz;
z=zeros(1,nx);
for i=1:nx    
    xx=x(i);
    if isnan(xx)
        j=find(gc==i&isnan(v1));
    else
        j=find(gc==i&~isnan(v1));
        x1=v1(j);        
        k=find(xx>=x1,1,'last');
        if length(k)==1
            j=j(k);
        end
    end
    if length(j)==1
        z(i)=gz(j);
    end
end

mod=model.mod;
b=model.b;

X=[1 z(mod)];
X(isnan(X))=0;
xb=X*b;
label=xb>0;
score=xb;
    
end

function [x,xmeas]=lastsample(data,t)

maxtime=28*24;
[nr,nx]=size(data);
t2=t(nr);
x=data(nr,:);
xmeas=maxtime*ones(1,nx);
for i=1:nx
    if ~isnan(x(i))
        xmeas(i)=0;
        continue
    end
    xx=data(:,i);
    k=find(~isnan(xx),1,'last');
    if isempty(k),continue,end
    t1=t(k);
    x(i)=xx(k);
    xmeas(i)=t2-t1;
end

end
