function model = load_sepsis_model()
% loadpath= 'C:\Users\Noorzadeh\Google Drive\SepsisChallenge\training\matFiles';
% load ([loadpath '\alldata-Binterp.mat']);
% % Da=D; sepsa =seps;
% % load ([loadpath '\alldata-Binterp.mat']);
% % D=[Da ;D]; seps=[sepsa;seps];
% 
% ICU = D(:,40);
% ICUseps = ICU(seps==1);
% ICUnonseps = ICU(seps==0);
% 
% display(mean(ICU));
% display(std(ICU));
% 
% display(mean(ICUseps));
% display(std(ICUseps));
% 
% display(mean(ICUnonseps));
% display(std(ICUnonseps));
% 
% boxplot([ICUseps ; ICUnonseps],[ones(1,length(ICUseps)) 2*ones(1,length(ICUnonseps))] );
% xticklabels({'sepsis','nonsepsi'});
% title('ICU Length of stay');
model =[];
