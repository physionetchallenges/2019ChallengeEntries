function feature_vector = MakeFeatureVector(data, N_features, N_rows)

X = [];
x_current = repmat(data(1, :), 1, N_rows);

for j = 1: size(data, 1)
    x_current(1:N_features) = data(j, :);
    X = [X; x_current];
    x_current = circshift(x_current,[0, N_features]);
end

signal = X(end, :);

feature_vector = features_main_4(signal, N_features);
end

