function newResistance = updateResistance(resistance, memrModel, voltageDiff)

        if(isnan(voltageDiff))
            newResistance = resistance;
        else
            upperR = memrModel(2);
            downR = memrModel(3);
            alpha = memrModel(4);
            beta = memrModel(5);
            vthres = memrModel(6);
            
            dt = 0.001282051282051282051282051282;
            dotR = (beta * voltageDiff) + 0.5 * (alpha - beta) * (abs(voltageDiff + vthres) - abs(voltageDiff - vthres));
            newResistance = resistance + (dt * dotR);
            
                if(newResistance > upperR)
                      newResistance = upperR;
                end
                if(newResistance < downR)
                       newResistance = downR;
                end        
        end
end