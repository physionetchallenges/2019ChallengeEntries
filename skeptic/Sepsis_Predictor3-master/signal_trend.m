function [trend] = signal_trend(x,param)
% Copyright 2019, TATA Consultancy Services. All rights reserved.

% param = 1, if signal rises and stays high during sepsis
%        -1, if signal falls and stays low during sepsis

x(isnan(x)) = [];
x = x*param;

x_change = diff(x);
x_change(find(x_change>=0)) = 1;
x_change(find(x_change<0)) = 0;

if ~isempty(x_change)
    trend = mean(x_change);
else
    trend = 0;
end

end