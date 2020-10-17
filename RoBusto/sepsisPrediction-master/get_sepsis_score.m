function [score, label] = get_sepsis_score(data, model)


feats2retain=1:40;
feats2el=[37 38];
feats2retain(feats2el)=[];
data=data(:,feats2retain);

data=fillmissing(data,'previous',1);

nClassifiers=1;

FeatsComb=model.FeatsComb;
Coefs=model.Coefs;

NFeats=sum(FeatsComb,2);

sampleMTS=data(end,:);
availableFeats=double(~isnan(sampleMTS));
validMdls=FeatsComb * availableFeats';
validMdls=validMdls==NFeats;
validMdls=find(validMdls,nClassifiers); 

if ~isempty(validMdls)
    FeatCombs_h=FeatsComb(validMdls,:);
    FeatWeights_h=Coefs(validMdls,2:end);
    Intercepts_h=Coefs(validMdls,1);
    MultiModel_score_h=nan(numel(validMdls),1);
    for mdl_i=1:numel(validMdls)

        FeatComb_h_i=logical(FeatCombs_h(mdl_i,:));
        Intercept_h_i=Intercepts_h(mdl_i);
        FeatWeights_h_i=FeatWeights_h(mdl_i,FeatComb_h_i);
        logRegMdl=[Intercept_h_i FeatWeights_h_i]';
        validFeats_h_i=sampleMTS(FeatComb_h_i);



        MultiModel_score_h(mdl_i) = glmval(logRegMdl,validFeats_h_i,'logit'); 


    end

    weightedMeanScore=mean(MultiModel_score_h);
    score=weightedMeanScore;
    label=weightedMeanScore>=0.5;
else
    score=0;
    label=0;
end


end
