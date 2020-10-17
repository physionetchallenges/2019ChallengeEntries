function [score, label] = get_sepsis_score(data, model)
%include all available data
selvital = [1:40];
%imputation
X = data(:,selvital); 
X = fillmissing(X,'previous');
X = fillmissing(X,'constant',0);

padding = 8;
X = history(X, padding);

scores = predict(model.data, X);

score = scores(end);
label = score;
end
