classdef LayerNorm < nnet.layer.Layer

    properties(Constant)
        % (Optional) Layer properties.
        Epsilon = 1e-8;
        % Layer properties go here.
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
        Gamma
        Beta
        
    end
    
    methods
        function layer = LayerNorm(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the layer.

            % Layer constructor function goes here.
            layer.Name = name;

            % Set layer description.
            layer.Description = "LayerNorm";
            
            layer.Gamma = rand([1]);
            layer.Beta  = rand([1]);
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            
            % Layer forward function for prediction goes here.
            shape=size(X);
            if length(shape)==2
                mu=mean(X,1);
                mu=reshape(mu,1,[]);
                mu=repmat(mu,[shape(1) 1]);
                sig2=sum((X-mu).^2,1);
                sig2=reshape(sig2,1,[]);
                sig2=repmat(sig2,[shape(1) 1]);
                Z=(X-mu)./sqrt(sig2+layer.Epsilon);
                
                g=layer.Gamma;
                b=layer.Beta;
                g=g(:,:,1);
                b=b(:,:,1);
                g=repmat(g,[shape(1) shape(2)]);
                b=repmat(b,[shape(1) shape(2)]);
                
                Z=Z.*g-b;
                
            else
                mu=mean(X,[1 3]);
                mu=reshape(mu,1,[],1);
                mu=repmat(mu,[shape(1) 1 shape(3)]);
                sig2=sum((X-mu).^2,[1 3]);
                sig2=reshape(sig2,1,[],1);
                sig2=repmat(sig2,[shape(1) 1  shape(3)]);
                Z=(X-mu)./sqrt(sig2+layer.Epsilon);
                
                g=layer.Gamma;
                b=layer.Beta;
                g=repmat(g,[shape(1) shape(2) shape(3)]);
                b=repmat(b,[shape(1) shape(2) shape(3)]);
                
                Z=Z.*g-b;
            end
            
        end

        function [Z,memory] = forward(layer, X)
            % (Optional) Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer  - Layer to forward propagate through
            %         X      - Input data
            % Outputs:
            %         Z      - Output of layer forward function
            %         memory - Memory value for backward propagation

            shape=size(X);
            if length(shape)==2
                mu=mean(X,1);
                mu=reshape(mu,1,[]);
                mu=repmat(mu,[shape(1) 1]);
                sig2=sum((X-mu).^2,1);
                sig2=reshape(sig2,1,[]);
                sig2=repmat(sig2,[shape(1) 1]);
                Z=(X-mu)./sqrt(sig2+layer.Epsilon);
                
                g=layer.Gamma;
                b=layer.Beta;
                g=g(:,:,1);
                b=b(:,:,1);
                g=repmat(g,[shape(1) shape(2)]);
                b=repmat(b,[shape(1) shape(2)]);
                
                Z=Z.*g-b;
                
            else
                mu=mean(X,[1 3]);
                mu=reshape(mu,1,[],1);
                mu=repmat(mu,[shape(1) 1 shape(3)]);
                sig2=sum((X-mu).^2,[1 3]);
                sig2=reshape(sig2,1,[],1);
                sig2=repmat(sig2,[shape(1) 1  shape(3)]);
                Z=(X-mu)./sqrt(sig2+layer.Epsilon);
                
                g=layer.Gamma;
                b=layer.Beta;
                g=repmat(g,[shape(1) shape(2) shape(3)]);
                b=repmat(b,[shape(1) shape(2) shape(3)]);
                
                Z=Z.*g-b;
            end
            
            
            % Layer forward function for training goes here.
%            
%             memory={mu,sig2,g,b};
            memory=[];
        end

        function [dLdX,dGamma,dBeta] = backward(layer, X, Z, dLdZ,memory)
            % Backward propagate the derivative of the loss function through 
            % the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Output of layer forward function            
            %         dLdZ              - Gradient propagated from the deeper layer
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX              - Derivative of the loss with respect to the
            %                             input data
            %         dLdW1, ..., dLdWn - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            % Layer backward function goes here.
            
            shape=size(X);
            if length(shape)==2
                mu=mean(X,1);
                mu=reshape(mu,1,[]);
                mu=repmat(mu,[shape(1) 1]);
                sig2=sum((X-mu).^2,1);
                sig2=reshape(sig2,1,[]);
                sig2=repmat(sig2,[shape(1) 1]);
%                 Z=(X-mu)./sqrt(sig2+layer.Epsilon);
                
                g=layer.Gamma;
                b=layer.Beta;
                g=g(:,:,1);
                b=b(:,:,1);
                g=repmat(g,[shape(1) shape(2)]);
                b=repmat(b,[shape(1) shape(2)]);
                
%                 Z=Z.*g-b;
                
            else
                mu=mean(X,[1 3]);
                mu=reshape(mu,1,[],1);
                mu=repmat(mu,[shape(1) 1 shape(3)]);
                sig2=sum((X-mu).^2,[1 3]);
                sig2=reshape(sig2,1,[],1);
                sig2=repmat(sig2,[shape(1) 1  shape(3)]);
%                 Z=(X-mu)./sqrt(sig2+layer.Epsilon);
                
                g=layer.Gamma;
                b=layer.Beta;
                g=repmat(g,[shape(1) shape(2) shape(3)]);
                b=repmat(b,[shape(1) shape(2) shape(3)]);
                
%                 Z=Z.*g-b;
            end
            
            dLdX=dLdZ.*g.*(1./sqrt(sig2+eps));
            dGamma=sum(dLdZ.*X,'all');
            dBeta=sum(-dLdZ,'all');
            
        end
    end
end