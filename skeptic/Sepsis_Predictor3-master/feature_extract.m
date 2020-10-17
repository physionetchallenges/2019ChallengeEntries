function F = feature_extract(raw_data, re_data, normal_range)
% Copyright 2019, TATA Consultancy Services. All rights reserved.

% Total 99 features

l = size(raw_data,1);

% 1 icu stay duration
icuhrs = raw_data(end,40);

% 1 Unit information
unit_info = raw_data(end,37)+2*raw_data(end,38);
unit_info(isnan(unit_info)) = 0;

% 1 Hospital admission time
hosp_adm = raw_data(end,39);

% 34 no. of measurements
non_nan = ~isnan(raw_data(:,1:34));
norm_non_nan_count = sum(double(non_nan),1)./l;
norm_non_nan_count(:,[10 16 20 29 30 32]) = []; 

% SB Feat1 34 NO.s
non_nan_data = raw_data(:,1:34);
if l > 1
   x1 = non_nan_data(end,:)- non_nan_data(end-1,:);
else
   x1 = zeros(1,34);
end
x1(isnan(x1)) = 0;

% SB Feat1 34 NO.s
% if l > 1
%    x2 = non_nan_data(end,:)- nanmean(non_nan_data(1:end-1,:));
% else
%    x2 = zeros(1,34);
% end
% x2(isnan(x2)) = 0;

% 6 shannon entropy
% hr = re_data(:,1);
% o2sat = re_data(:,2);
% temp = re_data(:,3);
% sbp = re_data(:,4);
% map = re_data(:,5);
% resp = re_data(:,6);
% ent = [wentropy(hr,'shannon') wentropy(o2sat,'shannon') wentropy(sbp,'shannon') wentropy(temp,'shannon') ... 
%     wentropy(map,'shannon') wentropy(resp,'shannon')];


% out of range features
lower_lim = repmat(normal_range(1,:),l,1);
upper_lim = repmat(normal_range(2,:),l,1);

lower_than_normal = raw_data(:,1:34) < lower_lim;
lower_than_normal_count = sum(double(lower_than_normal),1);

higher_than_normal = raw_data(:,1:34) > upper_lim;
higher_than_normal_count = sum(double(higher_than_normal),1);

% 34 out of range
out_of_range_norm = (lower_than_normal_count + higher_than_normal_count)./l;
% out_of_range_norm(:,[11 28 33]) = [];

% 34 SB features

% x1 = zeros(size(raw_data,1),34);
% non_nan_count = sum(double(non_nan),1);
% non_nan_greater_one = (non_nan_count > 1);
% x1(non_nan_greater_one) = non_nan(non_nan_count) - non_nan(non_nan_count-1);

% % 34 how much less than lower limit of normal range
% less_low = zeros(1,34);
% for i=1:34
%     if sum(double(lower_than_normal(:,i))) > 0
%         less_low(i) = max(normal_range(1,i)-raw_data(:,i));
%     end
% end

% % 7 sd while out of range
% out_of_range = lower_than_normal | higher_than_normal;
% oor_std = [std(raw_data(out_of_range(:,1),1)) ...
%     std(raw_data(out_of_range(:,2),2)) ...
%     std(raw_data(out_of_range(:,3),3)) ...
%     std(raw_data(out_of_range(:,4),4)) ...
%     std(raw_data(out_of_range(:,5),5)) ...
%     std(raw_data(out_of_range(:,6),6)) ...
%     std(raw_data(out_of_range(:,7),7))];
% 
% % 15 correlations among 6 vitals
% if size(re_data,1) < 2
%     correlation = zeros(1, 15);
% else
%     r = corrcoef(re_data);
%     correlation = [];
%     for i=1:size(re_data,2)-1
%         correlation = [correlation r(i,i+1:end)];
%     end
% end

% 34 max change in raw data features
change_feat = zeros(1,34);
for ch = 1:34
    ch_data = raw_data(:,ch);
    ch_data(isnan(ch_data)) = [];
    change_ch_data = diff(ch_data);
    if isempty(change_ch_data)
        change_feat(:,ch) = 0;
    else
        change_feat(:,ch) = max(change_ch_data);
    end
end
F = [icuhrs unit_info hosp_adm norm_non_nan_count out_of_range_norm x1];

end