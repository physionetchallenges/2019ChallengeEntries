%% feature extraction

% if p.fm_ex
%     Significant features found using tree training and ANOVA
featureLabels = [1 3 4 6 7 8 18 23 29];
nanC_labels = [11 12 13 14 18 22 23 26];

current_data = X(:,featureLabels);
current_data_nan = X(:,nanC_labels);
% average values of whole duration
f_mean = mean(current_data,1,'omitnan');
f_mean(isnan(f_mean)) = 0;

for j = 1:numel(featureLabels)
    %     entropy (feature 4), nans replace to '0'.
    entropy_data = current_data(:,j);
    entropy_data(isnan(entropy_data)) = 0;
    f_entr(j) = wentropy(entropy_data,'shannon');
end
% kurtosis (feature 5)
f_kurt = kurtosis(current_data,0,1);
% standard error (feature 6)
f_se = std(current_data,1,'omitnan') / sqrt(size(current_data,1));
%         count how many non vital measurements
f_nC =  mean(sum(~isnan(current_data_nan))/Xd1);

% statistical features | various features based on last 7 hours
f_stats = getStatisticalFeatures_n(current_data, 7);

f1 = f_mean;
f2 = f_entr;
f3 = f_kurt;
f3(isnan(f3)) = 0; % Nan to zero
f4 = f_se;
f4(isnan(f4)) = 0; % Nan to zero
f5 = X(1,35); %age
f6 = X(1,39); % hospitalization time, before ICU
f6(isnan(f6)) = 0; % nan to zero
f7 = f_nC;
f8 = f_stats;
f8(isnan(f8)) = 0; % Nan to zero

F = [f1 f2 f3 f4 f5 f6 f7 f8]; %8 sets with 111 features

%     normalize
f_N = (F - model{2,2}.normalization(1,:)) ./ model{2,2}.normalization(2,:);


featuresX = f_N;


