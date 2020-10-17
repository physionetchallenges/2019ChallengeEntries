% This utility function imputes important features from feature vector. It
% is assumed that these features will have at least 20% valid entries. 
% The important features are HR, MAP, Resp, Lactate
function X = ImputeFeatures(X, keySet)
% X: Features of a patient
% Look at each feature and interpolate its values so that there is no NaN
% at the end.
%  numFeatures = size(X, 2);
  
  t = 1: size(X,1);
  t = t';
  indicesImpFeature = [keySet('HR'), keySet('MAP'), keySet('Resp'), keySet('Lactate')];
  for i = 1:length(indicesImpFeature) % features past 34 are either meta-data or categorical features.
      f = X(:, indicesImpFeature(i));
      ts = t(~isnan(f));
      if (size(ts, 1) > 1)
          fs = f(~isnan(f));
          fs_i = interp1(ts, fs, t, 'linear');
          % Plot the original values and interpolated values.
          %figure(13);
          %subplot(2, 1, 1);
          %plot(t, f);
          %grid;
          %subplot(2, 1, 2);
          %plot(t, fs_i);
          %grid;
          X(:, i) = fs_i; % replace original values with new values.
      end
  end
end