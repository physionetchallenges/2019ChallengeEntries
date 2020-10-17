function [score, label] = get_sepsis_score(data, model)

%data : not contain label and feature names
ensemble = model;
%ensemble

%% remove some features
labelusenum = ensemble.labelusenum;
data_use = zeros(1,size(labelusenum,2));

for i=1:size(labelusenum,2) 
    data_use(:,i) =  data(end,labelusenum(i)); %data_use:last row
end

%% Input
data_knn = ensemble.InsertOrg;

N=5;
for i=1:size(data_use,1)
    data_new = [data_use(i,:);data_knn];
    data_new1 = [];

    for j = 1:size(data_new,2) 
        if ~isnan(data_new(1,j))
            data_new1 = [data_new1,data_new(:,j)];
        end
    end

    distance = sqrt(data_new1(1,:).^2*ones(size(data_new1(2:end,:)'))+ones(size(data_new1(1,:)))*(data_new1(2:end,:)').^2-2*data_new1(1,:)*data_new1(2:end,:)');
    [distance_sort,index] = sort(distance);

    dist_sum = 0;
    for j=1:N
        dist_sum = dist_sum + distance_sort(j);
    end   
    dist_alpha = distance_sort(1:N)/dist_sum; 
   
    data_alpha = [];
    for j=1:N
        data_alpha = [data_alpha;data_knn(index(j),:)];  
    end
        
    data_avg = dist_alpha*data_alpha;
    
    for j = 1:size(data_use,2)
        if isnan(data_use(i,j))
           data_use(i,j) = data_avg(1,j);
        end
    end
end

%% evalueate 
%zscore normalized
%data_use
data_zscore = zscore([data_use;data_knn]);
data_use = data_zscore(1:size(data_use,1),:);
re = EvaluateValue(data_use,ensemble); 

%% return score & label

    if re <= ensemble.thresh 
       score = re/(2*ensemble.thresh);
    else if re >= ensemble.max
       score = 1;
    else
       score = 0.5 + (re - ensemble.max)/(2*(max(re) - ensemble.max));
        end
    end

%   score = 1 - exp(-l_exp_bx); % a single risk score 
    label = double(score > 0.5); % a single binary
%prediction
end
