function model = load_sepsis_model()
    load('FeatsComb.mat','FeatsComb');
    load('Coefs1.mat','Coefs1');
    load('Coefs2.mat','Coefs2');
    load('Coefs3.mat','Coefs3');
    
    model.FeatsComb=FeatsComb;
    model.Coefs=[Coefs1; Coefs2; Coefs3];
end
