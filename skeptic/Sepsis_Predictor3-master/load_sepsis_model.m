function [param] = load_sepsis_model()
% Copyright 2019, TATA Consultancy Services. All rights reserved.

    m1 = load('model1.mat');
    model1 = m1.model1;
    m2 = load('model2.mat');
    model2 = m2.model2;
    
    r = load('rank_mrmr.mat');
    ranking = r.ranking;
    
    nr = load('normal_range.mat');
    normal_range = nr.normal_range;
    nr_vs = load('normal_range_vs.mat');
    normal_range_vs = nr_vs.normal_range_vs;
    
    param.model1 = model1;
    param.model2 = model2;
    param.ranking = ranking;
    param.normal_range = normal_range;
    param.normal_range_vs = normal_range_vs;
end
