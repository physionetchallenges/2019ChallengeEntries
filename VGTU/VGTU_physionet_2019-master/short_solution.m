%% feature extraction
% to get features from 3 datasets, run 3 times with different p.config path
% or run "all" dataset
% save results manually

%     if num_rows<9
%         f_mean = zeros(num_rows, 34);
%         f_entr = zeros(num_rows, 7);
%     else
%         f_mean = zeros(9, 34);
%         f_entr = zeros(9, 7);
%     end
%     % same structure as mean
%     f_med = f_mean;
%     % same structure for 7 feature | kurtosis and stardard error
%     f_kurt = f_entr;
%     f_se = f_entr;
%
%     f_ans = [];
%
%     for t = 1:num_rows
current_data = X;

% write answer (feature 1)
%         f_ans(t,1) = answers{i}(t);

%     mean feature 2 | omit nans | if no samples then 0
f_mean = mean(current_data,1,'omitnan');
f_mean(isnan(f_mean)) = 0;
f_mean(35:end)=[];

%     median feature 3 | omit nans | if no samples then 0
f_med = median(current_data,1,'omitnan');
f_med(isnan(f_med)) = 0;
f_med(35:end)=[];

for j = 1:7
    %     entropy (feature 4), nans replace to '0'.
    entropy_data = current_data(:,j);
    entropy_data(isnan(entropy_data)) = 0;
    f_entr(j) = wentropy(entropy_data,'shannon');
end
%     from here use only vitals
current_data = current_data(:,1:7);

% kurtosis (feature 5)
f_kurt = kurtosis(current_data,0,1);

% standard error (feature 6), first sample is '0' error
if Xd1==1
    f_se = zeros(1,7);
else
    f_se = std(current_data,1,'omitnan') / sqrt(size(current_data,1));
end

% for the short solution break loop
%         if t == 9
%             break;
%         end
%
%     end

% Save results to cell
%     fe_s{1, i} = f_ans;
%     fe_s{2, i} = f_mean;
%     fe_s{3, i} = f_med;
%     fe_s{4, i} = f_entr;
%     fe_s{5, i} = f_kurt;
%     fe_s{6, i} = f_se;
%     fe_s{7,i} = data(1,35); %age
%     fe_s{8,i} = data(1,39); %hosp time

%% dataset grouping
% only for local run
% this version works for cells only

%
%     path2f = 'D:\Datasets\physionet\2019\features\short';
%     fname = 'features_0821';
%     features_load = load(fullfile(path2f, fname));
%     features = struct2array(features_load);
%
%     fes_ALL=[];
%     num_files = length(features);
%     for i = 1:num_files
%         if iscell(features)
%     find false sepsis index
%             f_idx =  find(features{1,i}==0);
%             %     find true sepsis index
%             t_idx = find(features{1,i}==1);
%             %             make feature vector
%             f1 = features{1,i};
f2 = f_mean;
f3 = f_med(1:7); %for median only 7 (other values same as mean)
f4 = f_entr;
f5 = f_kurt;
f5(isnan(f5)) = 0; % Nan to zero
f6 = f_se;
f6(isnan(f6)) = 0; % Nan to zero
f7 = X(1,35); %age
%             f7 = ones(size(f2,1),1) * f7; % copy for all answers
f8 = X(1,39); % hospitalization time, before ICU
f8(isnan(f8)) = 0; % nan to zero
%             f8 = ones(size(f1,1),1) * f8; % copy for all answers
F = [f2 f3 f4 f5 f6 f7 f8];

%         end
%         fes_ALL = [fes_ALL;F];

%     end


%% normalization 


    %     normalize all
% f_N = normalize(F,1);
f_N = (F - model{2,1}.normalization(1,:)) ./ model{2,1}.normalization(2,:);

featuresX = f_N;
%     fes_ALL=[];
%     adasyn_features =
%     adasyn_labels =
%     [f_syn, l_syn] = ADASYN(f_N(:,2:end), f_N(:,1));
%     f_balan = [l_syn f_syn; f_N];

%     num_i = size(features,2);
%     for i = 1:num_i
%     end






