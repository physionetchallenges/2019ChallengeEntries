function normalized_observed_utility=pred(prah,YTest,vys)


    vys_bin=cellfun(@(x) x>prah,vys,'UniformOutput',false);



    vys_bin_T=cellfun(@(x) x' ,vys_bin,'UniformOutput',false);
    vys_T=cellfun(@(x) x' ,vys,'UniformOutput',false);
    YTest_T=cellfun(@(x) x' ,YTest,'UniformOutput',false);



    [~, ~, ~, ~, normalized_observed_utility,~]=compute_scores_2019_cell(YTest_T,vys_bin_T,vys_T);


    normalized_observed_utility=-normalized_observed_utility;




end