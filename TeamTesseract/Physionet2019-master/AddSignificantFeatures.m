function X = AddSignificantFeatures(X, keySet)
% lactate level & Mean Arterial Pressure are important biomarker for sepsis
% Increase its weight by making it squared.
  index = keySet('Lactate');
  X(:, index) = X(:, index).^2;
  index = keySet('MAP');
  X(:, index) = X(:, index).^2;
  index = keySet('HR');
  X(:, index) = X(:, index).^2;
  index = keySet('Resp');
  X(:, index) = X(:, index).^2;
end