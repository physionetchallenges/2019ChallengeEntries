function [score, label] = get_sepsis_score(data, model)
ICUL = data(end,40);
N=52;

if ICUL>N
    label = 1;
    score = ( ICUL*0.5)/N;
    if score > 1
       score =1;
    end
else
    label = 0;
    score = ( ICUL*0.5 )/N;
    if score >0.5
        score =0.5;
    end
end


end
