function [score, label] = get_sepsis_score(data, param)
% Copyright 2019, TATA Consultancy Services. All rights reserved.

% parameter setting
win = 36;
k = 249;
re_normal = [60 90 36.1 100 65 12; 100 100 37.2 120 110 25];

t = size(data,1);

raw_data = data;
re_data = data(:,[1 2 3 4 5 7]);

% resampling vitals
for ch = 1:size(re_data,2)
    
    t_nnan = find(~isnan(re_data(:,ch)));
    if isempty(t_nnan)
        rng(1);
        re_data(:,ch) = re_normal(1,ch) + (re_normal(2,ch) - re_normal(1,ch))*rand(size(re_data,1),1);
    elseif length(t_nnan) == 1
        re_data(:,ch) = re_data(t_nnan,ch);
    else
        re_data(:,ch) = spline(t_nnan, re_data(t_nnan,ch), 1:t);
    end
    
end

if t < win
    raw_data_seg = raw_data;
    re_data_seg = re_data;
else
    raw_data_seg = raw_data(t-win+1:t,:);
    re_data_seg = re_data(t-win+1:t,:);
end


F = [feature_extract(raw_data_seg, re_data_seg, param.normal_range)...
      feature_extract_VS(raw_data_seg(end,:), param.normal_range_vs)];


% predict by RF
[pred_label1, pred_prob] = predict(param.model1, F(:,param.ranking(1:k)));
pred_label1 = str2num(cell2mat(pred_label1));
                
% predict by Adaboost
pred_label2 = predict(param.model2,F(:,param.ranking(1:k)));
                
% Fusion
pred_label = double(pred_label1 & pred_label2);
pred_prob = pred_label;


score = pred_prob(:,1);
label = pred_label;

end
