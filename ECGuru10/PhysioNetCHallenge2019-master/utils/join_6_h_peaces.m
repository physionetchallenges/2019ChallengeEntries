function vys=join_6_h_peaces(X,Y,data)



vys=cell(size(X));
counter=0;
for k=1:length(X)
%     k
    x=X{k};
    y=zeros(size(x,1),1);
    
    for kk=6:size(x,1)
        counter=counter+1;
        y(kk)=data(counter);
    end
    
    vys{k}=y;
    
    
end



