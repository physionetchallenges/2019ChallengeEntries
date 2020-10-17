function [output] = history(patient, padding)
    [numRows, numVitals] = size(patient);
       
    out = cell(1, padding);
    for i = 0:padding - 1
        data = patient(1:max(0, numRows - i), :);
        zeroArray = zeros(padding, 40);
        pad = zeroArray(1:min(numRows, i), :);
        out{i + 1} = [pad; data];
    end
    output = cell2mat(out);
end