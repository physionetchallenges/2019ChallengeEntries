function [score, label] = get_sepsis_score(data, model)
    
means0 = cell2mat(model(1));
stds0 = cell2mat(model(2));
meansA = cell2mat(model(3));
stdsA = cell2mat(model(4));
meansB = cell2mat(model(5));
stdsB = cell2mat(model(6));
mults0 = cell2mat(model(7));
biases0 = cell2mat(model(8));
multsA = cell2mat(model(9));
biasesA = cell2mat(model(10));
multsB = cell2mat(model(11));
biasesB = cell2mat(model(12));
boundary0 = cell2mat(model(13));
boundaryA = cell2mat(model(14));
boundaryB = cell2mat(model(15));
memristorModel = cell2mat(model(16));

edgeUp = memristorModel(2);
edgeDown = memristorModel(3);

dataOnly = data(:,1:39);

%Normalization of Data Set0
normData0 = dataOnly - means0;
normData0 = normData0./stds0;

%Normalization of Data SetA
normDataA = dataOnly - meansA;
normDataA = normDataA./stdsA;

%Normalization of Data SetB
normDataB = dataOnly - meansB;
normDataB = normDataB./stdsB;


%model individual Scaling and biasing
%set0
res0 = resistanceModelIndScIndBias(normData0, mults0, biases0, memristorModel);
pr0 = probabilityEstimation(res0, boundary0);

%setA
resA = resistanceModelIndScIndBias(normDataA, multsA, biasesA, memristorModel);
prA = probabilityEstimation(resA, boundaryA);

%setB
resB = resistanceModelIndScIndBias(normDataB, multsB, biasesB, memristorModel);
prB = probabilityEstimation(resB, boundaryB);


score = ((1.0/9.0) * pr0) + ((4.0/9.0) * prA) + ((4.0/9.0) * prB);

if(score > 0.5)
    label = 1;
else
    label = 0;
end

end
