function normalized_observed_utility=pred(prah,YTest,vys)


    vys_bin=cellfun(@(x) x>prah,vys,'UniformOutput',false);



    [~, ~, ~, ~, normalized_observed_utility,~]=compute_scores_2019_cell(YTest,vys_bin,vys);


    normalized_observed_utility=-normalized_observed_utility;




end