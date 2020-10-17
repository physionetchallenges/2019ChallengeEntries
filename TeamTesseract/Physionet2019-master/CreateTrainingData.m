function [XTotal, YTotal, keyMap] = CreateTrainingData(inputDir, processRecords)
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

    trainingdata = dir([inputDir '/*.psv']);
    

    fileID = fopen('SepsisRecords.txt', 'w');
    % ICULOS : 40, HosAdTime : 39, Unit1 : 37, Unit2 : 38
    heading = 'filename|t|HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel';
    fprintf(fileID, '%s\n', heading);

    % Mean Arterial Pressure (MAP) & Lactate level seems important for sepsis
    % shock
    % for quickSofa : Resp (8th entry) or SBP(5th entry) seems important
    X1 = [];
    Y1 = [];
    X2 = [];
    Y2 = [];
    X3 = [];
    Y3 = [];
    X4 = [];
    Y4 = [];
    X5 = [];
    Y5 = [];
    X6 = [];
    Y6 = [];
    X7 = [];
    Y7 = [];
    X8 = [];
    Y8 = [];
    X9 = [];
    Y9 = [];
    X10 = [];
    Y10 = [];
    XSepsis = [];
    YSepsis = [];
    XHealthy = [];
    YHealthy = [];

    [totalRecords, ~] = size(trainingdata);
    
    keyMap = containers.Map({'HR','O2Sat','Temp','SBP','MAP','DBP','Resp',...
              'EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2',...
              'AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct',...
              'Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI',...
              'Hct','Hgb','PTT','WBC','Fibrinogen','Platelets','Age','Gender','Unit1',... 
              'Unit2','HospAdmTime', 'ICULOS'}, ...
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40});

    for i=1:totalRecords
        
        fName = trainingdata(i).name;
        
        [X, Y, isSepsis] = AccumulateRecords([inputDir '/' fName], fileID);
        if (processRecords)
            [X] = AddSignificantFeatures(X, keyMap);
            [X] = ImputeFeatures(X);
        end
        
        if (isSepsis)
            XSepsis = [XSepsis; X];
            YSepsis = [YSepsis; Y];
        else
            XHealthy = [XHealthy; X];
            YHealthy = [YHealthy; Y];
        end
        
        if (i <= 0.1 * totalRecords)
            X1 = [X1; X];
            Y1 = [Y1; Y];
        elseif ((0.1 * totalRecords < i) && (i <= 0.2 * totalRecords))
            X2 = [X2; X];
            Y2 = [Y2; Y];
        elseif ((0.2 * totalRecords < i) && (i <= 0.3 * totalRecords))
            X3 = [X3; X];
            Y3 = [Y3; Y];
        elseif ((0.3 * totalRecords < i) && ( i <= 0.4 * totalRecords))
            X4 = [X4; X];
            Y4 = [Y4; Y];
        elseif ((0.4 * totalRecords < i) && (i <= 0.5 * totalRecords))
            X5 = [X5; X];
            Y5 = [Y5; Y];
        elseif ((0.5 * totalRecords < i) && (i <= 0.6 * totalRecords))
            X6 = [X6; X];
            Y6 = [Y6; Y];
        elseif ((0.6 * totalRecords < i) && (i <= 0.7 * totalRecords))
            X7 = [X7; X];
            Y7 = [Y7; Y];
        elseif ((0.7 * totalRecords < i) && (i <= 0.8 * totalRecords))
            X8 = [X8; X];
            Y8 = [Y8; Y];
        elseif ((0.8 * totalRecords < i) && (i <= 0.9 * totalRecords))
            X9 = [X9; X];
            Y9 = [Y9; Y];
        elseif ((0.9 * totalRecords < i) && (i <= totalRecords))
            X10 = [X10; X];
            Y10 = [Y10; Y];
        end
        
    end % for

    XTotal = {X1; X2; X3; X4; X5; X6; X7; X8; X9; X10};
    YTotal = {Y1; Y2; Y3; Y4; Y5; Y6; Y7; Y8; Y9; Y10};
    [XTotald1, XTotald2] = size(XTotal);
    
    % Plot missing data %
    figure(1);
    barh(sum(isnan(cell2mat(XTotal)),1)/size(XTotal,1));
    h = gca;
    h.YTick = size(XTotal, 2);
    %h.YTickLabel = keyMap.keys();
    ylabel('Predictor');
    xlabel('Fraction of missing values');
    
    % First 10% values are in validation set.
    %XValidate = XTotal(1:size(X1, 1), 1:size(X1, 2));
    %YValidate = YTotal(1:size(Y1, 1), 1:size(Y1, 2));
    %XTrain = XTotal(size(X1, 1)+1:size(XTotal, 1), 1:size(XTotal, 2));
    %YTrain = YTotal(size(Y1, 1)+1:size(YTotal, 1), 1:size(YTotal, 2));

    %figure(1);
    %plot(XSepsis(:,39), XSepsis(:,8), 'g', XHealthy(:,39), XHealthy(:,8), 'm');
    
    %figure(2);
    %h1 = histogram(XSepsis(:, mapFeatureName('Resp')), 5);
    %hold on;
    
    %figure(3);
    %h2 = histogram(XHealthy(:, mapFeatureName('Resp')), 5);
    %hold on;

    fclose(fileID);
end

