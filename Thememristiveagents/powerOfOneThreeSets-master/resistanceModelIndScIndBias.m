function resistance = resistanceModelIndScIndBias(normData, mults, biases, memristorModel)
transformedInp = normData.*mults;
transformedInp = transformedInp + biases;

dimensions = size(transformedInp);
rows = dimensions(1);
columns = dimensions(2);

resistance = 4.0;
for i = 1:(rows - 1)
    for j= 1:columns
        resistance = updateResistance(resistance, memristorModel, transformedInp(i,j));
    end
end

  i = rows;
  mean = 0;
for j= 1:columns
        resistance = updateResistance(resistance, memristorModel, transformedInp(i,j));
        mean = mean + resistance;
end
 mean = mean/columns; 
 resistance = mean;
