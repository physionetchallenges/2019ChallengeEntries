function model = load_sepsis_model()

    model{1,1} = load('m_short_0823');
    model{2,1} = load('norm_s');
    model{1,2} = load('m_medium_0823_2');
    model{2,2} = load('norm_m');
    model{1,3} = load('m_long_0824');
    model{2,3} = load('norm_l');
%     model = [];
end
