classdef nanregression_layer < nnet.layer.RegressionLayer
    % This layer implements the generalized dice loss function for training
    % semantic segmentation networks.
    
    
    methods
        
        function layer = nanregression_layer(name)
            % layer =  dicePixelClassificationLayer(name) creates a Dice
            % pixel classification layer with the specified name.
            
            % Set layer name.          
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'regression layer ignoring zero values';
        end
        
        
        function loss = forwardLoss(layer, Y, T)
            
           
            nonnans=T~=0;
            YY=Y.*nonnans;
            TT=T.*nonnans;
            YYY=(YY-TT).^2;
            loss=0.5*sum(YYY(:))/sum(nonnans(:));
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            
            
            nonnans=T~=0;
            YY=Y.*nonnans;
            TT=T.*nonnans;
           
            dLdY=(YY-TT)/sum(nonnans(:));
            
        end
    end
end