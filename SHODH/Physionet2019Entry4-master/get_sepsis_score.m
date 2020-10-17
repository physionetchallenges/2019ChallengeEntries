% This is Physionet Challenge 2019 Entry 4
% S-2 guidelines
% August 2019 by Aruna Deogire


function [scores, labels] = get_sepsis_score(data,model)

Ta = data;
% grab the features
X=Ta(:,1:34);
[Xd1,Xd2] = size(X);
c=Xd1;


% Making NaN values zero for 1 to 41 columns
  for k=1:Xd2, Ta(isnan(Ta(:,k)),k)=0;end
  
 % PARAMETER Extraction and assigning variables for identification
 % ************************************************************************  
    % SIR Parameters
        HR = Ta(c,1);Temp=Ta(c,3); Resp =Ta(c,7); WBC=Ta(c,32);
    % Stage 2 Parameters 
       SBP =Ta(c,4);Creatinine=Ta(c,20);
       Bilirubin_total =Ta(c,27); Platelets=Ta(c,34);
                
       SIR_PARA= [HR Resp Temp WBC];
       SOFA_PARA =[SBP Creatinine Bilirubin_total Platelets];
        
 % Initialization
 % ************************************************************************  
 
    hrcount=0; tempcount=0;  respcount=0; wbccount=0;
    sbpcount=0;crcount=0;bicount=0;ptcount=0;
 
    Phr=0;Ptemp=0;Psbp=0;Presp=0;
    Pmap=0;Pcr=0;Pbi=0;Ppt=0;Pmap=0;Pwbc=0;
 
%  Calculations
% *************************************************************************

% SIR Calculations

% HR
       if (HR >91)
           hrcount = 1;
           Phr=0.15;
       else
           hrcount= 0;
           Phr=0;
       end
% Temperature       
  
      if (Temp>38||(Temp<36 && Temp>0))
           tempcount = 1;
           Ptemp=0.15;
      else
          tempcount = 0;
          Ptemp=0;
      end
  
% Respiration
  
        if (Resp>20)
          respcount = 1;
          Presp=0.15;
        else
          respcount = 0;
          Presp=0;
        end      
% WBC
  
        if (WBC >12||(WBC <4 && WBC >0))
            wbccount = 1;
            Pwbc=0.15;
        else
            wbccount = 0;
            Pwbc=0;
        end       
  
 SIRcount= hrcount+respcount+tempcount+wbccount;
 PSIR= Phr + Ptemp + Presp + Pwbc  ;     
        
% SOFA Calculations
% ************************************************************************* 
      
%   SBP

      if SBP <90 && SBP >0 
        sbpcount=1;
        Psbp=0.1;
     else
        sbpcount=0;
        Psbp=0;
     end
 
%  Scr
   
       if Creatinine>2 
           crcount=1;
           Pcr=0.1;
       else
           crcount=0;
           Pcr=0;
       end
  
 %  Sbili  
   
        if Bilirubin_total>2 
           bicount=1;
           Pbi = 0.1;
        else
            bicount=0;
            Pbi=0;
        end
  
  
% Platelets
        
        if Platelets<100 &&   Platelets>0
           ptcount=1;
           Ppt = 0.1;
        else
            ptcount=0;
            Ppt =0;
        end
   
  SOFA= sbpcount + crcount + bicount + ptcount;
  PSOFA = Psbp + Pcr + Pbi + Ppt ;
  
% Total Probability
% *************************************************************************

 for i=1:Xd1
 if PSIR>=0.3
     PSIR=0.3;
 end
 if PSOFA>=0.1
     PSOFA=0.1;
 end
 end
    P = PSIR + PSOFA; P=P';
    
% Sepsis Detection
% *************************************************************************

       if P >=0.4
            L =1;
        else
            L=0;
        end

     scores=P;
     labels=L;
end


     

   
  