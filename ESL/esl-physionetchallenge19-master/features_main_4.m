function feature_vector = features_main_4(signal, N_features)
feature_vector = [];
N_hours = 3;

for ind = 1:7
    ind_range = ind:N_features:(N_hours-1)*N_features+ind;
    s1 = signal(1, ind_range);
    feature_vector = [feature_vector s1];
end

for ind = 8:34
    ind_range = ind:N_features:(N_hours-1)*N_features+ind;
    s1 = signal(1, ind_range);
    feature_vector = [feature_vector mean(s1)];
end

feature_vector = [feature_vector signal(1, 35:40)];

end
