#!/usr/bin/Rscript

get_sepsis_score = function(data, model){
  library(dplyr)
  colnames(data) = names(model$medians)
  data = data.frame(data)
  for(i in 1:40){
    naInd = as.numeric(is.na(data[,i]))
    currentNames = names(data)
    data = cbind.data.frame(data, naInd)
    names(data) = c(currentNames, paste(currentNames[i],"_MI", sep = ""))
  }
  data = tidyr::fill(data,names(data))
  
  data$FiO2 = ifelse(is.na(data$FiO2), 0.21,data$FiO2)
  data$Bilirubin_total = ifelse(is.na(data$Bilirubin_total), 0.8,data$Bilirubin_total)
  data$Platelets = ifelse(is.na(data$Platelets),190,data$Platelets )
  data$MAP = ifelse(is.na(data$MAP),82,data$MAP )
  data$Creatinine = ifelse(is.na(data$Creatinine),0.8,data$Creatinine )
  data$SaO2 = ifelse(is.na(data$SaO2),97,data$SaO2 )
  
  
  data = data %>% 
    mutate(
      PlateletScore = ifelse(Platelets > 150, 0,
                             ifelse(Platelets > 100 & Platelets <=150, 1,
                                    ifelse(Platelets > 50 & Platelets <=100, 2,
                                           ifelse(Platelets > 20& Platelets <=50, 3, 4)))),
      BilirubinScore = ifelse(Bilirubin_total < 1.2, 0,
                              ifelse(Bilirubin_total < 1.9 & Bilirubin_total >=1.2, 1,
                                     ifelse(Bilirubin_total < 5.9 & Bilirubin_total >=1.9, 2,
                                            ifelse(Bilirubin_total < 12 & Bilirubin_total >=5.9, 3, 4)))),
      MAPScore = ifelse(MAP < 70, 1,0),
      CreatineScore = ifelse(Creatinine < 1.2, 0,
                             ifelse(Creatinine < 1.9 & Creatinine >= 1.2 , 1,
                                    ifelse(Creatinine < 3.4 & Creatinine >= 1.9, 2,
                                           ifelse(Creatinine < 4.9 & Creatinine >= 3.4, 3, 4)))),
      sfRatio = SaO2/FiO2,
      sfRatioScore = ifelse(sfRatio > 400, 0,
                            ifelse(sfRatio <=400 & sfRatio > 315 , 1,
                                   ifelse(sfRatio < 315& sfRatio > 235, 2,
                                          ifelse(sfRatio < 235 & sfRatio > 150, 3,4)))),
      sofaScore = PlateletScore + BilirubinScore +MAPScore + CreatineScore + sfRatioScore,
      sofaDiff1 = sofaScore - lag(sofaScore, n = 2),
      sofaDiff2 = sofaScore - lag(sofaScore, n = 4),
      sofaDiff3 = sofaScore - lag(sofaScore, n = 6),
      sofaDiff4 = sofaScore - lag(sofaScore, n = 8),
      
      sofaDiff1 = ifelse(is.na(sofaDiff1),0,sofaDiff1),
      sofaDiff2 = ifelse(is.na(sofaDiff2),0,sofaDiff2),
      sofaDiff3 = ifelse(is.na(sofaDiff3),0,sofaDiff3),
      sofaDiff4 = ifelse(is.na(sofaDiff4),0,sofaDiff4)
    )
  
  for(i in 1:length(model$medians)){
    data[is.na(data[,i]),i] = model$medians[i]
  }
  
  valPrior = model$valPrior
  getDeltaMI = model$getDeltaMI
  
  data = dplyr::mutate(data,
                       HR_1 = valPrior(HR),
                       HR_2 = valPrior(HR_1),
                       HR_3 = valPrior(HR_2),
                       HR_4 = valPrior(HR_3),
                       
                       ICULOS_2 = row_number(),
                       
                       
                       HR_Avg = (HR_1+HR_2+HR_3+HR_4)/4,
                       HR_Var = ((HR_Avg - HR)^2 + (HR_Avg-HR_1)^2 + (HR_Avg-HR_2)^2 + (HR_Avg-HR_3)^2 +(HR_Avg-HR_4)^2)/4,
                       HR_DA = HR - HR_Avg,
                       
                       
                       O2_1 = valPrior(O2Sat),
                       O2_2 = valPrior(O2_1),
                       O2_3 = valPrior(O2_2),
                       O2_4 = valPrior(O2_3),
                       
                       O2_Avg = (O2_1+O2_2+O2_3+O2_4)/4,
                       O2_Var = ((O2_Avg -O2Sat)^2 + (O2_Avg-O2_1)^2 + (O2_Avg-O2_2)^2 + (O2_Avg-O2_3)^2 +(O2_Avg-O2_4)^2)/4,
                       
                       O2_DA = O2Sat - O2_Avg,
                       
                       
                       
                       Temp_1 = valPrior(Temp),
                       Temp_2 = valPrior(Temp_1),
                       Temp_3 = valPrior(Temp_2),
                       Temp_4 = valPrior(Temp_3),
                       
                       Temp_Avg = (Temp_1+Temp_2+Temp_3+Temp_4)/4,
                       Temp_Var = ((Temp_Avg - Temp)^2 + (Temp_Avg-Temp_1)^2 + (Temp_Avg-Temp_2)^2 + (Temp_Avg-Temp_3)^2 +(Temp_Avg-Temp_4)^2)/4,
                       Temp_DA = Temp - Temp_Avg,
                       
                       SBP_1 = valPrior(SBP),
                       SBP_2 = valPrior(SBP_1),
                       SBP_3 = valPrior(SBP_2),
                       SBP_4 = valPrior(SBP_3),
                       
                       SBP_Avg = (SBP_1+SBP_2+SBP_3+SBP_4)/4,
                       SBP_Var = ((SBP_Avg -SBP)^2 + (SBP_Avg-SBP_1)^2 + (SBP_Avg-SBP_2)^2 + (SBP_Avg-SBP_3)^2 +(SBP_Avg-SBP_4)^2)/4,
                       
                       SBP_DA = SBP - SBP_Avg,
                       
                       MAP_1 = valPrior(MAP),
                       MAP_2 = valPrior(MAP_1),
                       MAP_3 = valPrior(MAP_2),
                       MAP_4 = valPrior(MAP_3),
                       
                       MAP_Avg = (MAP_1+MAP_2+MAP_3+MAP_4)/4,
                       MAP_Var = ((MAP_Avg -MAP)^2 + (MAP_Avg-MAP_1)^2 + (MAP_Avg-MAP_2)^2 + (MAP_Avg-MAP_3)^2 +(MAP_Avg-MAP_4)^2)/4,
                       
                       MAP_DA = MAP - MAP_Avg,
                       
                       DBP_1 = valPrior(DBP),
                       DBP_2 = valPrior(DBP_1),
                       DBP_3 = valPrior(DBP_2),
                       DBP_4 = valPrior(DBP_3),
                       
                       DBP_Avg = (DBP_1+DBP_2+DBP_3+DBP_4)/4,
                       DBP_Var = ((DBP_Avg -DBP)^2 + (DBP_Avg-DBP_1)^2 + (DBP_Avg-DBP_2)^2 + (DBP_Avg-DBP_3)^2 +(DBP_Avg-DBP_4)^2)/4,
                       
                       DBP_DA = DBP - DBP_Avg,
                       
                       
                       Resp_1 = valPrior(Resp),
                       Resp_2 = valPrior(Resp_1),
                       Resp_3 = valPrior(Resp_2),
                       Resp_4 = valPrior(Resp_3),
                       
                       Resp_Avg = (Resp_1+Resp_2+Resp_3+Resp_4)/4,
                       Resp_Var = ((Resp_Avg -Resp)^2 + (Resp_Avg-Resp_1)^2 + (Resp_Avg-Resp_2)^2 + (Resp_Avg-Resp_3)^2 +(Resp_Avg-Resp_4)^2)/4,
                       Resp_DA = Resp - Resp_Avg,
                       
                       
                       FiO2_1 = valPrior(FiO2),
                       FiO2_2 = valPrior(FiO2_1),
                       FiO2_3 = valPrior(FiO2_2),
                       FiO2_4 = valPrior(FiO2_3),
                       
                       FiO2D = FiO2 - FiO2_1,
                       FiO2_Avg = (FiO2_1+FiO2_2+FiO2_3+FiO2_4)/4,
                       FiO2_Var = ((FiO2_Avg -FiO2)^2 + (FiO2_Avg-FiO2_1)^2 + (FiO2_Avg-FiO2_2)^2 + (FiO2_Avg-FiO2_3)^2 +(FiO2_Avg-FiO2_4)^2)/4,
                       FiO2_D = FiO2 - FiO2_Avg,
                       
                       pH_1 = valPrior(pH),
                       pH_2 = valPrior(pH_1),
                       pH_3 = valPrior(pH_2),
                       pH_4 = valPrior(pH_3),
                       
                       
                       SaO2_1 = valPrior(SaO2),
                       SaO2_2 = valPrior(SaO2_1),
                       SaO2_3 = valPrior(SaO2_2),
                       SaO2_4 = valPrior(SaO2_3),
                       
                       
                       
                       WBC_1 = valPrior(WBC),
                       WBC_2 = valPrior(WBC_1),
                       WBC_3 = valPrior(WBC_2),
                       WBC_4 = valPrior(WBC_3),
                       
                       Lactate_1 = valPrior(Lactate),
                       Lactate_2 = valPrior(Lactate_1),
                       Lactate_3 = valPrior(Lactate_2),
                       Lactate_4 = valPrior(Lactate_3),
                       
                       CreatineCount = cumsum(!Creatinine_MI),
                       PlateletCount = cumsum(!Platelets_MI),
                       FiO2Count = cumsum(!FiO2_MI),
                       pHCount = cumsum(!pH_MI),
                       PaCO2Count = cumsum(!PaCO2_MI),
                       SaO2Count = cumsum(!SaO2_MI),
                       PlateletCount = cumsum(!Platelets_MI),
                       ASTCount = cumsum(!AST_MI),
                       BUNCount = cumsum(!BUN_MI),
                       PlateletsCount = cumsum(!Platelets_MI),
                       AlkalinephosCount = cumsum(!Alkalinephos_MI),
                       CalciumCount = cumsum(!Calcium_MI),
                       Bilirubin_totalCount = cumsum(!Bilirubin_total_MI),
                       Bilirubin_directCount = cumsum(!Bilirubin_direct_MI),
                       GlucoseCount = cumsum(!Glucose_MI),
                       LactateCount = cumsum(!Lactate_MI),
                       HctCount = cumsum(!Hct_MI),
                       HgbCount = cumsum(!Hgb_MI),
                       PTTCount = cumsum(!PTT_MI),
                       WBCCount = cumsum(!WBC_MI),
                       FibrinogenCount = cumsum(!Fibrinogen_MI),
                       HRCount = cumsum(!HR_MI),
                       O2SatCount = cumsum(!O2Sat_MI),
                       TempCount = cumsum(!Temp_MI),
                       SBPCount = cumsum(!SBP_MI),
                       MAPCount = cumsum(!MAP_MI),
                       DBPCount = cumsum(!DBP_MI),
                       RespCount = cumsum(!Resp_MI),
                       
                       
                       
                       HR_D = getDeltaMI(HR_MI),
                       O2Sat_D = getDeltaMI(O2Sat_MI),
                       Temp_D = getDeltaMI(Temp_MI),
                       SBP_D = getDeltaMI(SBP_MI),
                       MAP_D = getDeltaMI(MAP_MI),
                       DBP_D = getDeltaMI(DBP_MI),
                       Resp_D = getDeltaMI(Resp_MI),
                       EtCO2_D = getDeltaMI(EtCO2_MI),
                       BaseExcess_D = getDeltaMI(BaseExcess_MI),
                       HCO3_D = getDeltaMI(HCO3_MI),
                       FiO2_D = getDeltaMI(FiO2_MI),
                       pH_D = getDeltaMI(pH_MI),
                       PaCO2_D = getDeltaMI(PaCO2_MI),
                       SaO2_D = getDeltaMI(SaO2_MI),
                       AST_D = getDeltaMI(AST_MI),
                       BUN_D = getDeltaMI(BUN_MI),
                       Alkalinephos_D = getDeltaMI(Alkalinephos_MI),
                       Calcium_D = getDeltaMI(Calcium_MI),
                       Chloride_D = getDeltaMI(Chloride_MI),
                       Creatinine_D = getDeltaMI(Creatinine_MI),
                       Glucose_D = getDeltaMI(Glucose_MI),
                       Lactate_D = getDeltaMI(Lactate_MI),
                       Magnesium_D = getDeltaMI(Magnesium_MI),
                       Phosphate_D = getDeltaMI(Phosphate_MI),
                       Potassium_D = getDeltaMI(Potassium_MI),
                       Bilirubin_direct_D = getDeltaMI(Bilirubin_direct_MI),
                       Bilirubin_total_D = getDeltaMI(Bilirubin_total_MI),
                       Hct_D = getDeltaMI(Hct_MI),
                       Hgb_D = getDeltaMI(Hgb_MI),
                       PTT_D = getDeltaMI(PTT_MI),
                       WBC_D = getDeltaMI(WBC_MI),
                       Platelets_D = getDeltaMI(Platelets_MI),
                       TroponinI_D = getDeltaMI(TroponinI_MI),
                       Fibrinogen_D = getDeltaMI(Fibrinogen_MI),
                       
                       Temp12 = sqrt(Temp),
                       Temp2 = Temp^2,
                       Temp3 = Temp^3,
                       LogTemp = log(Temp),
                       BUN12 = sqrt(BUN),
                       BUN2 = BUN^2,
                       BUN3 = BUN^3,
                       LogBUN = log(BUN),
                       HR12 = sqrt(HR),
                       HR2 = HR^2,
                       HR3 = HR^3,
                       LogHR = log(HR),
                       
                       TempLast12 = lag(!Temp_MI,1,default = 0) + lag(!Temp_MI,2,default = 0) +lag(!Temp_MI,3,default = 0) +lag(!Temp_MI,4,default = 0) +lag(!Temp_MI,5,default = 0) +
                         lag(!Temp_MI,6,default = 0) +lag(!Temp_MI,7,default = 0) +lag(!Temp_MI,8,default = 0) +lag(!Temp_MI,9,default = 0) +lag(!Temp_MI,10,default = 0) +lag(!Temp_MI,11,default = 0) +
                         lag(!Temp_MI,12,default = 0),
                       
                       TempLast6 = lag(!Temp_MI,1,default = 0) + lag(!Temp_MI,2,default = 0) +lag(!Temp_MI,3,default = 0) +lag(!Temp_MI,4,default = 0) +lag(!Temp_MI,5,default = 0) +
                         lag(!Temp_MI,6,default = 0),
                       
                       HRLast12 = lag(!HR_MI,1,default = 0) + lag(!HR_MI,2,default = 0) +lag(!HR_MI,3,default = 0) +lag(!HR_MI,4,default = 0) +lag(!HR_MI,5,default = 0) +
                         lag(!HR_MI,6,default = 0) +lag(!HR_MI,7,default = 0) +lag(!HR_MI,8,default = 0) +lag(!HR_MI,9,default = 0) +lag(!HR_MI,10,default = 0) +lag(!HR_MI,11,default = 0) +
                         lag(!HR_MI,12,default = 0),
                       
                       HRLast6 = lag(!HR_MI,1,default = 0) + lag(!HR_MI,2,default = 0) +lag(!HR_MI,3,default = 0) +lag(!HR_MI,4,default = 0) +lag(!HR_MI,5,default = 0) +
                         lag(!HR_MI,6,default = 0),
                       
                       FiO2Last12 = lag(!FiO2_MI,1,default = 0) + lag(!FiO2_MI,2,default = 0) +lag(!FiO2_MI,3,default = 0) +lag(!FiO2_MI,4,default = 0) +lag(!FiO2_MI,5,default = 0) +
                         lag(!FiO2_MI,6,default = 0) +lag(!FiO2_MI,7,default = 0) +lag(!FiO2_MI,8,default = 0) +lag(!FiO2_MI,9,default = 0) +lag(!FiO2_MI,10,default = 0) +lag(!FiO2_MI,11,default = 0) +
                         lag(!FiO2_MI,12,default = 0),
                       
                       FiO2Last6 = lag(!FiO2_MI,1,default = 0) + lag(!FiO2_MI,2,default = 0) +lag(!FiO2_MI,3,default = 0) +lag(!FiO2_MI,4,default = 0) +lag(!FiO2_MI,5,default = 0) +
                         lag(!FiO2_MI,6,default = 0),
                       
                       O2SatLast12 = lag(!O2Sat_MI,1,default = 0) + lag(!O2Sat_MI,2,default = 0) +lag(!O2Sat_MI,3,default = 0) +lag(!O2Sat_MI,4,default = 0) +lag(!O2Sat_MI,5,default = 0) +
                         lag(!O2Sat_MI,6,default = 0) +lag(!O2Sat_MI,7,default = 0) +lag(!O2Sat_MI,8,default = 0) +lag(!O2Sat_MI,9,default = 0) +lag(!O2Sat_MI,10,default = 0) +lag(!O2Sat_MI,11,default = 0) +
                         lag(!O2Sat_MI,12,default = 0),
                       
                       O2SatLast6 = lag(!O2Sat_MI,1,default = 0) + lag(!O2Sat_MI,2,default = 0) +lag(!O2Sat_MI,3,default = 0) +lag(!O2Sat_MI,4,default = 0) +lag(!O2Sat_MI,5,default = 0) +
                         lag(!O2Sat_MI,6,default = 0)
  )
  data = data[-c(41:80)]
  scores = predict(model$mod, as.matrix(data))
  labels = (scores > 0.57)
  results = cbind(scores[nrow(data)], labels[nrow(data)])
  return(results)
}

load_sepsis_model = function(){
  columnMedians = readRDS("columnMedians")
  xgbMod = readRDS("XGBoostMod")

  valPrior = function(x){
    if(length(x) == 1){
      return(-999)
    }else{
      return(c(-999,x[1:(length(x)-1)]))
    }
  }
  
  getDeltaMI = function(x){
    cs = cumsum(x)
    cm = cummax(cs*!x)
    return(cs-cm)
  }
  
  return(list(mod = xgbMod, medians = columnMedians, valPrior = valPrior, getDeltaMI = getDeltaMI))
}
