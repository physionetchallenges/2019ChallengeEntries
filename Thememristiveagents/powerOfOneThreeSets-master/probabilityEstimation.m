function probability = probabilityEstimation(resistance, boundary)

if(boundary > 4.0)
      if( resistance > boundary)
          if(7.0 == boundary)
              probability = 1.0;
          else
              probability = 0.5  + 0.5 * (resistance - boundary)/(7.0 - boundary);
          end 
          if(probability > 1.0)
             probability = 1.0;
          end
      else
          if(resistance < 4.0)
             probability = 0.0;
          else
             probability = (0.5/(boundary - 4.0)) * (resistance - 4.0);
          end
      end
else
    if( resistance < boundary)
               probability = 0.5 + 0.5 * (resistance - boundary)/(1.0 - boundary); 
           if(probability > 1.0)
              probability = 1.0; 
          end
    else
          if(resistance > 4.0)
              probability = 0.0;
          else
              probability = 0.5 * (resistance - 4.0)/(boundary - 4.0);
          end
          if(probability < 0.0)
              probability = 0.0;
          end
    end
end
