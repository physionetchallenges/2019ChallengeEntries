function model = load_sepsis_model()
    %import decision tree prediciton fct
    import classreg.learning.classif.CompactClassificationEnsemble.*;
    
    %load array of trees as random forest model
    loadedObj = load('model.mat');
    model = loadedObj.model;
    model = handler(model);
end
