% 
% Copyright (C) 2019
% Shailesh Nirgudkar
% Shreyasi Datta
% Tianyu Ding

% 
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

datadir='C:\Shailesh\Learn\Research\PhysionetChallenge_2019\training\setAB';

% processRecords = true;
% [XTotal, YTotal, keyMap] = CreateTrainingData(datadir, processRecords);
% 
% for i=1:10
%     XTrain = [];
%     YTrain = [];
%     XValidate = [];
%     YValidate = [];
%     for j=1:10
%         if (j==i)
%             XValidate = [XValidate XTotal{j}];
%             YValidate = [YValidate YTotal{j}];
%         else
%             XTrain = [XTrain; XTotal{j}];
%             YTrain = [YTrain; YTotal{j}];
%         end
%     end
%     
%     % Store the training data and validation data
%     trainingdata = ['trainingdata' num2str(i) '.mat'];
%     validationdata = ['validationdata' num2str(i) '.mat'];
%     save(trainingdata, 'XTrain', 'YTrain');
%     save(validationdata, 'XValidate', 'YValidate');
% end % for loop
% 
% for i = 1:10
%     trainingdata = ['trainingdata' num2str(i) '.mat'];
%     validationdata = ['validationdata' num2str(i) '.mat'];
%     load(trainingdata); % XTrain, YTrain
%     load(validationdata); % XValidate, YValidate
%     [Xd1,Xd2] = size(XTrain);
%     fprintf('Size of train X = [%d, %d]\n', Xd1, Xd2);
% 
%     numFeatures = Xd2;
%     numSamples = Xd1;
% 
%     modifiedKeySet = keyMap;
% 
%     rng(0,'twister') % For reproducibility
%     % last prop name-value pair means use all predictor variables.
%     t = templateTree('surrogate','all', 'Reproducible', true, 'NumVariablesToSample','all');
% 
%     classNames = {'Sepsis', 'Healthy'};
%     cost.ClassNames = classNames;
%     cost.ClassificationCosts = [0 5; 1 0];
%     Ensemble = fitcensemble(XTrain, YTrain, 'Method', 'RUSBoost',...
%                             'Learners', t,...
%                             'PredictorNames', keyMap.keys, ...
%                             'OptimizeHyperparameters', {'NumLearningCycles','LearnRate','MaxNumSplits'}, ...
%                             'Cost', cost.ClassificationCosts);
%                             
%                         
%    % 'NumLearningCycles',150
%    % 'LearnRate',0.1, ...
%    %'OptimizeHyperparameters', {'NumLearningCycles','LearnRate','MaxNumSplits'}
% 
%     save(['computedEnsemble' num2str(i)], 'Ensemble');
%                     
%     [YFit, scores] = predict(Ensemble, XValidate);
% 
%     confusionchart(YValidate, YFit);
% 
% end % for

% Now read entire training data without any modifications, use Ensemble
% giving minimum false negative and predict the score over entire
% population.
processRecords = false;
[XValidate, YValidate, keyMap] = CreateTrainingData(datadir, processRecords);
XValidate = cell2mat(XValidate);
save('wholeTrainingData.mat', 'XValidate', 'YValidate');
load(['computedEnsemble' num2str(7)], 'Ensemble');
[YFit, scores] = predict(Ensemble, XValidate);
YValidate = cell2mat(YValidate);
confusionchart(YValidate, YFit);
