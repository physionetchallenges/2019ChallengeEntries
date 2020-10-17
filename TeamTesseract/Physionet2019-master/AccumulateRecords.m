function [X, Y, isSepsis] = AccumulateRecords(input_file, fileID)
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


% read the data
% true indicates retain SepsisLabel
Ta = ReadChallengeData(input_file, true);
% grab the features
X = Ta(:,1:40);       
Y = Ta(:,41);

isSepsis = false;

[~,fname,~] = fileparts(input_file);

if exist('fileID', 'var')
    once = false;
    if (sum(Y) ~= 0) % There are entries containing 1's
        isSepsis = true;
        for i=1:size(Y)
              if (once == false)
                %fprintf('%s\n', fname);
                once = true;
              end
              arrayInStr = sprintf('%f|', X(i,:), Y(i));
              allOneString = sprintf('%s|%d|%s', fname, i, arrayInStr);
              allOneString = allOneString(1:end-1);
              %fprintf(fileID, '%s\n', allOneString);
        end
    end
end % fileID

