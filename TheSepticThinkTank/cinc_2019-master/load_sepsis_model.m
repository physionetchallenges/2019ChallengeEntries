function model = load_sepsis_model()
    model = load('ultimate_pred_cut_optimized.mat');
    model.th = 0.52;
    model.t_delta = 4;
    
    tmp = load('TOP_FEAT_real_cost_time_mean_std_5_n_cv_10.mat','TOP_FEAT');
    model.top_feats = tmp.TOP_FEAT(1:100);
end
