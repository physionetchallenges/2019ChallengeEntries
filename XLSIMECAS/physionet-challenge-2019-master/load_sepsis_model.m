function model = load_sepsis_model()
    load('ensemble.mat')
    model = ensemble;
end
