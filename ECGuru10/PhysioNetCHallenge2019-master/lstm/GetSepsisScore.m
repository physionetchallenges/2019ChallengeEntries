function [scores, labels] = GetSepsisScore(input_file)
Ta = ReadChallengeData(input_file);

XTest={Ta(:,1:40)}; 
load('model_d2_b128_velka_sit.mat')
load('pom_variables.mat')
tresh=x;

XTest=cellfun(@(x) normalize015(x,minv,maxv),XTest,'UniformOutput',false);
XTest=cellfun(@(x) nany_na_nuly(x) ,XTest,'UniformOutput',false);
XTest=cellfun(@(x) x' ,XTest,'UniformOutput',false);



xx=XTest{1};
tmpp=randperm(length(lengths),bs);
tmp=lengths(tmpp);
tmp=max(tmp);
size0=size(xx,2);
if tmp>size0
    pad=tmp-size0;
    xx=padarray(xx,[0 pad],sp,'post');
end

vyss=predict(net,{xx},'MiniBatchSize',1,'SequencePaddingValue',sp);
vyss=vyss{1};
vyss=vyss(:,1:size0);
vyss=vyss';
scores=vyss(:,2);

labels = double([scores>tresh]);
end




function [values, column_names] = ReadChallengeData(filename)
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

  %% ignore SepsisLabel column if present
  if strcmp(column_names(end), 'SepsisLabel')
    column_names = column_names(1:end-1);
    values = values(:,1:end-1);
  end
end



