% 
% Copyright (C) 2019
% Shailesh Nirgudkar
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
function [score, label] = get_sepsis_score(data, model)

	XTest = data(:, 1:40);
	keyMap = containers.Map({'HR','O2Sat','Temp','SBP','MAP','DBP','Resp',...
              'EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2',...
              'AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct',...
              'Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI',...
              'Hct','Hgb','PTT','WBC','Fibrinogen','Platelets','Age','Gender','Unit1',... 
              'Unit2','HospAdmTime', 'ICULOS'}, ...
              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40});
        
    [XTest] = ImputeFeatures(XTest, keyMap);
    [XTest] = AddSignificantFeatures(XTest, keyMap);
	
    %m = size(data, 1);
    % Take last row only for prediction.
	[tmplabel, tmpscore] = predict(model.Ensemble, XTest);
    if (sum(tmpscore(end,:)) > 1.0)
        tmpscore(end,1) = tmpscore(end,1)/(tmpscore(end,1) + tmpscore(end,2));
        tmpscore(end,2) = tmpscore(end,2)/(tmpscore(end,1) + tmpscore(end,2));
    end
    isSepsis = (model.Ensemble.ClassNames==1);
    tmpscore_issepsis = tmpscore(:, isSepsis);
    tmplabel_issepsis = (tmplabel==1);
    
%     [~, probROC, thre] = perfcurve([0 1], tmpscore_issepsis, true);
%     figure
%     hold
%     scatter(thre, probROC, 'filled')
%     xlabel('Score')
%     ylabel('Probability')

   if ((tmpscore(end,1)/tmpscore(end,2) < 0.6))
       label = 1;
       score = tmpscore(end, 2);
   else
       label = 0;
       score = tmpscore(end, 1);
   end
    

%     label = tmplabel(end);
%     if (label == 0)
%         score = tmpscore(end, 1);
%     elseif (label == 1)
%         score = tmpscore(end, 2);
%     end
	
	% tmpscores has size (N, 2). The first column is probability that
    % patient is non-sepsis. Second probability is patient is sepsis.
    % These probabilities do not add to 1. So to determine if patient is
    % sepsis, either second probability has to be > 0.5 or ratio of first
    % prob/second prob must be < 0.25 so that second probability clearly
    % wins and their sum should at least be 0.45.
%     for i=1:size(tmplabel,1)
%         if (tmpscore(i,2) >= 0.5 || ...
%             (sum(tmpscore(i,:)) > 0.45 && tmpscore(i,1)/tmpscore(i,2) < 0.25))
%             label = 1;
%             score = tmpscore(i,2);
%         else
%             label = 0;
%             score = tmpscore(i,1);
%         end
%     end
end
