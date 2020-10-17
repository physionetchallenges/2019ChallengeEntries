function [score, label] = get_sepsis_score(current_data, model)
load('median_values.mat')
N_features = 40;
N_rows = 3;
th = 0.07;

gender_column = 36;
label_gender = current_data(1, gender_column);

switch label_gender
    case 0
        median_values = median_values_females;
    case 1
        median_values = median_values_males;
end

current_data = current_data(:, 1:N_features);

for k = 1 : N_features
    if isnan(current_data(1, k))
       i1 = find(~isnan(current_data(:, k)), 1);
       if isempty(i1) 
          current_data(1, k) =  median_values(k);
       else 
           current_data(1, k) =  current_data(i1, k);              
       end           
    end       
end
        
data_filled = fillmissing(current_data, 'previous', 1);

feature_vector = MakeFeatureVector(data_filled, N_features, N_rows);
[label_vector, scores_vector] = predict(model.model, feature_vector);
label_rf = str2double(label_vector);
score_rf = scores_vector(end);

if (score_rf > th)
    label = 1;
else 
    label = 0;
end

score = score_rf;
end
