function model = load_sepsis_model()

maxFeatures = 15;

load('classificationEnsemble','classificationEnsemble');
load('stats.mat','medianData','meanData','stdData');
load('importantFeatures','importantFeatures');
                 
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);

model.medianData = medianData;
model.meanData = meanData;
model.stdData = stdData;
model.vec = sort(importantFeatures(1:maxFeatures));
model.ensemblePredictFcn = ensemblePredictFcn;

end

