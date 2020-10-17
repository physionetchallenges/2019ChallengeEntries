function Y = getStatisticalFeatures_n(X, frameSize)

% X = Ta(:,7);

Xs1 = size(X,1);
% X = data;
Y=[];
remainder= mod(Xs1, frameSize);
    
% calculate for 9 features
for i = 1:9
    % calculate featuers for 10hour frames
    % frameSize = 10;


    % last frame can be empty or filled with nans
    % or last frame duration is 'frameSize'.... not tested
    % to improve normalize... not tested
    X_filled = X(:,i);
    if remainder~=0
        X_filled(end+1:end+(frameSize-remainder)) = nan;
    end
    X_windowd = reshape(X_filled, frameSize, ceil(Xs1/frameSize));

%     std of signal
    feat(1) = std(X(:,i),'omitnan');

    X_winE = mean(X_windowd,'omitnan');
    X_windiff = diff(X_winE);
    % basic statistics: std_fr, last2fr
    feat(2) = std(X_winE);
    feat(3) = mean(X_winE(end-1:end));
    % maximum found values with other values: max2first. max2min
    feat(4) = max(X_winE)-(X_winE(1));
    feat(5) = max(X_winE)-min(X_winE);
    % features with change every 'framesize' hours: lastdiff, diffstd, maxdiff
    feat(6) = X_windiff(end);
    feat(7) = std(X_windiff);
    feat(8) = max(abs(X_windiff));
    
    Y = [Y;feat];
    clear feat

end

Y = Y(:)';
