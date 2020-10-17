function x=nany_na_nuly(x)
    x(isnan(x))=0;
end