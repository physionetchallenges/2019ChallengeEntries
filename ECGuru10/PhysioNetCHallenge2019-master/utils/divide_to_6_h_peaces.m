function [X_peaces,Y_peaces]=divide_to_6_h_peaces(X,Y)


lengths=cellfun(@(x) size(x,1),X,'UniformOutput',true);

X_peaces=zeros(sum(lengths-5),size(X{1},2)*6);
Y_peaces=zeros(sum(lengths-5),1);



counter=0;
for k=1:length(X)
%     k
    x=X{k};
    y=Y{k};
    
    for kk=6:size(x,1)
        counter=counter+1;
        Y_peaces(counter,:)=y(kk);
        tmp=x(kk-5:kk,:);
        X_peaces(counter,:)=tmp(:);
        
    end
    
    
    
    
end



