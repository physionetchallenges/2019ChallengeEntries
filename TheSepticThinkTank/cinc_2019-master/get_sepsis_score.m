function [score, label] = get_sepsis_score(data, model)
% generate_advanced_complete_data_optimized
% reduce_tree_time_complete_data_optimized
% generate_mini_trees %NEEDS HACKED CompactTreeBagger!
    
    x0  = data(end,:);
    
    n_lines = size(data,1);
    n_0     = max(1,n_lines-model.t_delta);
    
    x1 = mean(data(n_0:n_lines,:),1,'omitnan');
    x2 = std(data(n_0:n_lines,:),0,1,'omitnan');
    
    feat_vec = [x0 x1 x2];
    
    feat_vec = feat_vec(model.top_feats);
    
    [~, prob] = predict(model.ultimate_pred_cut,feat_vec);
    score = prob(2);
    label = double(score>model.th);
    
end