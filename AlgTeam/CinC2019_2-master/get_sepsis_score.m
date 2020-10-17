function [score, label] = get_sepsis_score(data, model)

testData = plausFilter(data);

testData(testData(:,35)>=90,35) = 90; % Feature 35: age
testData(:,37) = 0.5*(testData(:,37)-testData(:,38)+1); % Unit1(MICU)=1, Unit2(SICU)=0
testData(:,38) = [];
model.meanData(38) = [];
model.medianData(38) = [];
model.stdData(38) = [];
    
indData = [1:35,38:39];
testData(:,indData) = imputeData(testData(:,indData),model.meanData(indData));
testData(:,indData) = (testData(:,indData)-model.medianData(indData))./model.stdData(indData);

testData(isnan(testData(:,36)),36) = 1; % gender (male)
testData(isnan(testData(:,37)),37) = 1; % MICU

testData = testData(end,:);
testData = testData(:,model.vec);

predFcn = model.ensemblePredictFcn;
[label, scores] = predFcn(testData);
score = scores(:,2);

end

function data = imputeData(data,medianData)

    isFeatNan = find(isnan(data(1,:)));
    data(1,isFeatNan) = medianData(isFeatNan);

    for ind = 2:size(data,1)
        isFeatNan = find(isnan(data(ind,:)));
        data(ind,isFeatNan) = data(ind-1,isFeatNan);
    end

end

function dataOut = plausFilter(dataIn)

plausibility = [
10 300          % HR
60 100          % SpO2
32 42.2         % Temp 
40 280          % SBP
0 300           % MAP 
20 130          % DBP
5 60            % Resp 
0 150           % EtCO2
-20 20          % BaseExcess 
0 50            % HCO3 
0 1             % FiO2 
6 8             % pH
0 200           % PaCO2
0 100           % SaO2
0 400           % AST
0 500           % BUN
0 250           % Alkalinephos
0 20            % Calcium 
75 145          % Chloride 
0 10            % Creatinine 
0 50            % Bilirubin_direct
0 1000          % Glucose
0 100           % Lactate
0 10            % Magnesium 
0 12            % Phosphate 
1 10            % Potassium
0 50            % Bilirubin_total 
0 200           % TroponinI
10 70           % Hct
2 22            % Hgb
0 250           % PTT
0 50            % WBC
0 800           % Fibrinogen
5 1500          % Platelets
0 150           % Age
0 1             % Gender
0 1             % Unit1
0 1             % Unit2
-inf inf	% HospAdmTime
1 inf       % ICULOS
];

dataOut = dataIn;
dataOut(dataOut < plausibility(:,1)') = nan;
dataOut(dataOut > plausibility(:,2)') = nan;

end