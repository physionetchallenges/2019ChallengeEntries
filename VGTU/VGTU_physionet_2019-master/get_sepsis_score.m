function [score, label] = get_sepsis_score(data, model)
% send only last value

Xd1= size(data,1);
score = 0;
label = 0;

X = data;
if Xd1<10
    try
        short_solution;
        score = model{1,1}.trainedModel.predictFcn(featuresX);
    catch
        disp('error 1')
        %         disp(Xd1)
        score = 0;
    end
    
    % %%% ---------------------------------------------------------------------
elseif Xd1<61
    % for moderate ICU time

    try
        medium_solution;
        if model{1,2}.trainedModel.predictFcn(featuresX)
            score = 1;
        else
            score = 0;
        end
    catch
        disp('error 2')
        disp(Xd1)
        score = 0;
    end
    
    
    % %%% ---------------------------------------------------------------------
else
    %      for long ICU time
    %     Significant features from tree training and ANOVA
    try
        long_solution
        if model{1,3}.trainedModel.predictFcn(featuresX)
            score = 1;
        end
    catch
        disp('error 3')
        disp(Xd1)
        score = 0;
    end
    
end

label = score;
