function [score, label] = get_sepsis_score(data, model)
    addpath('lstm')
    addpath('utils')
    net=model{1};
    x=model{2};
    minq=model{3};
    maxq=model{4};
    
    
    data=normalize015(data,minq,maxq);
    data=nany_na_nuly(data);
    
    sp=0;
    
    score=predict(net,data','MiniBatchSize',1,'SequencePaddingValue',sp);
    
    score=score';
    score=score(:,2);
    score=score(end);
    
    
    label = double(score > x);
end
