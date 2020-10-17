function [values, column_names] = ReadChallengeData(filename, retainSepsisLabel)
  f = fopen(filename, 'rt');
  try
    l = fgetl(f);
    column_names = strsplit(l, '|');
    values = dlmread(filename, '|', 1, 0);
  catch ex
    fclose(f);
    rethrow(ex);
  end
  fclose(f);

  if (~retainSepsisLabel)
    %% ignore SepsisLabel column if present
    if strcmp(column_names(end), 'SepsisLabel')
        column_names = column_names(1:end-1);
        values = values(:,1:end-1);
    end
  end
end
