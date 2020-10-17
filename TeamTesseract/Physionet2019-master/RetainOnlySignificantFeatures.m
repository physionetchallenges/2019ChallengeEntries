% The utility function is used for dropping features which have mostly NaNs
function [X, keySet, valid] = RetainOnlySignificantFeatures(X)
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


     %Y = X(:,[1:36, 39:40]);
    
    keySet = {'HR','O2Sat','Temp','SBP','MAP','DBP','Resp',...
              'EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2',...
              'AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct',...
              'Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI',...
              'Hct','Hgb','PTT','WBC','Fibrinogen','Platelets','Age','Gender','Unit1',... 
              'Unit2','HospAdmTime',...
              'ICULOS'};
    
   
    %X = Y;
    valid = true;
end

