#!/usr/bin/Rscript

## And now continue as before.
get_sepsis_score = function(CINCdata, myModel){
  myModel <- load_sepsis_model()

  ## Add the column names back
  colnames(CINCdata) <- c("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", 
                          "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", 
                          "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", 
                          "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", 
                          "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", 
                          "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", 
                          "ICULOS")
  
  ## Forward fill missing values
  cols <- colnames(CINCdata)[1:35]
  for (col in cols){
    # With the line feeding of data we can now have only a single row...
    if (nrow(CINCdata)>=2){
      for (n in seq(2,nrow(CINCdata))){
        if (is.na(CINCdata[[n,col]])){
          CINCdata[[n,col]]<- CINCdata[[n-1,col]]
        }
      }
    }
  }
  # Calculate a few extra indices
  CINCdata <- cbind(CINCdata, "PulsePressure"=(CINCdata[,"SBP"]-CINCdata[,"DBP"]))
  CINCdata[CINCdata[,"PulsePressure"]<0,"PulsePressure"] <- NA
  CINCdata <- cbind (CINCdata,"CO"=CINCdata[,"PulsePressure"]*CINCdata[,"HR"]/1000)
  CINCdata <- cbind (CINCdata,"ShockIndex"= CINCdata[,"HR"]/CINCdata[,"SBP"])
  CINCdata <- cbind (CINCdata,"ModifiedShockIndex"=CINCdata[,"HR"]/CINCdata[,"MAP"])
  CINCdata <- cbind (CINCdata,"COvariation"=CINCdata[,"PulsePressure"]/CINCdata[,"MAP"])
  
  #CombinationVitals 
  CINCdata <- cbind(CINCdata, "HRTemp"=(CINCdata[,"HR"]/CINCdata[,"Temp"]))
  CINCdata <- cbind(CINCdata, "RRTemp"=(CINCdata[,"Resp"]/CINCdata[,"Temp"]))
  CINCdata <- cbind(CINCdata, "comb_RR"=(6.21 + 0.06*CINCdata[,"HR"]+0.20*CINCdata[,"Temp"]))
  CINCdata <- cbind(CINCdata, "RRdiff"=(CINCdata[,"Resp"]+0.20*CINCdata[,"comb_RR"]))
  
  # Make Combinations [Electrolytes]
  CINCdata <- cbind(CINCdata, "comb_PotassiumMagnesium"=(CINCdata[,"Potassium"]/CINCdata[,"Magnesium"]))
  CINCdata <- cbind(CINCdata, "comb_MagnesiumCalcium"=(CINCdata[,"Magnesium"]/CINCdata[,"Calcium"]))

  # SpO2 with virtual shunt
  CINCdata <- cbind(CINCdata, "VS"=(68.864 * log10(103.711 -CINCdata[,"SaO2"])- 52.109))
  
  # P/F ratios
  CINCdata[CINCdata[,"FiO2"]<0.20,"FiO2"] <- 0.20 
  CINCdata <- cbind(CINCdata, "PF"=(CINCdata[,"SaO2"]/CINCdata[,"FiO2"]))
  CINCdata <- cbind(CINCdata, "SF"=(CINCdata[,"O2Sat"]/CINCdata[,"FiO2"]))

  # Oxygen delivery
  CINCdata <- cbind(CINCdata, "DO2"=(((1.39*CINCdata[,"Hgb"]* CINCdata[,"O2Sat"]/100) + (0.003* CINCdata[,"SaO2"]))*CINCdata[,"CO"]))
  CINCdata <- cbind(CINCdata, "HbBO2MAP"=(CINCdata[,"Hgb"]+CINCdata[,"O2Sat"]+CINCdata[,"MAP"]))
  CINCdata[is.na(CINCdata[,"HbBO2MAP"]),"HbBO2MAP"] <- 10.4+CINCdata[is.na(CINCdata[,"HbBO2MAP"]),"O2Sat"]+CINCdata[is.na(CINCdata[,"HbBO2MAP"]),"MAP"]
  
  # Lab combinations
  CINCdata <- cbind(CINCdata, "UreaCreat"=(CINCdata[,"BUN"]/CINCdata[,"Creatinine"]))
  CINCdata <- cbind(CINCdata, "UreaCreatsum"=(CINCdata[,"BUN"]+CINCdata[,"Creatinine"]))
  CINCdata <- cbind(CINCdata, "comb_Hgb"=(15.50 - 0.063*CINCdata[,"O2Sat"]+ 0.015*CINCdata[,"MAP"]))
  CINCdata <- cbind(CINCdata, "comb_HCO3Lac"=(CINCdata[,"HCO3"]/CINCdata[,"Lactate"]))
  CINCdata <- cbind(CINCdata, "comb_HCO3Lacdiff"=(CINCdata[,"HCO3"]+CINCdata[,"Lactate"]))
  CINCdata <- cbind(CINCdata, "AnionGap"=(140+CINCdata[,"Potassium"]-CINCdata[,"Chloride"]-CINCdata[,"HCO3"]))
  CINCdata <- cbind(CINCdata, "comb_ClpH"=(CINCdata[,"Chloride"]/CINCdata[,"pH"]))
  
  # Add number of lab values
  if (nrow(CINCdata)>=2){
    CINCdata <- cbind (CINCdata,"numLabs"=rowSums(!is.na(CINCdata[,11:34])))
  }else{
    CINCdata <- cbind (CINCdata,"numLabs"=sum(!is.na(CINCdata[,11:34])))
  }

  # Identify abnormal values
  cols <- colnames(CINCdata)[c(1:34,41:63)]
  for (col in cols){
    colAbnormal<-paste0('abn_',col)
    CN <- colnames(CINCdata)
    CINCdata <- cbind (CINCdata,colAbnormal=!(CINCdata[,col] > myModel$normdf[,col][1] & CINCdata[,col] < myModel$normdf[,col][2]))
    # Naming is complicated
    colnames(CINCdata)<-c(CN,colAbnormal)
    # Missing is normal
    CINCdata[is.na(CINCdata[,colAbnormal]),colAbnormal] <- 0
  }
  
  # Make absolute z-scores [Also do age!]
  for (col in c("Age",cols)){
    colaZscore<-paste0('az_',col)
    CN <- colnames(CINCdata)
    CINCdata <- cbind (CINCdata,colaZscore=abs((CINCdata[,col] - myModel$meanCINCdata[col])/ myModel$sdCINCdata[col]))
    # Naming is complicated
    colnames(CINCdata)<-c(CN,colaZscore)
    # Missing is normal
    CINCdata[is.na(CINCdata[,colaZscore]),colaZscore] <- 0
  }
  
  # Add 1st derivative
  n <- nrow(CINCdata)
  for (col in cols){
    colDelta<-paste0('Delta_',col)
    CN <- colnames(CINCdata)
    if (n>=2){
      CINCdata <- cbind (CINCdata,colDelta=c(NA,CINCdata[2:n,col]-CINCdata[1:n-1,col]))
    }else{
      CINCdata <- cbind (CINCdata,colDelta=0)
    }
    # Naming is complicated
    colnames(CINCdata)<-c(CN,colDelta)
    # Missing is normal
    CINCdata[is.na(CINCdata[,colDelta]),colDelta] <- 0
  }
  
  # Replace a few missing values with normal values
  for (c in cols){
    CINCdata[is.na(CINCdata[,c]),c] <- mean(myModel$normdf[,c])
  }
  
  # Make the myModel manually 
  ## which(colnames(CINCdata) %in% names(myModel$coeffs))
   cn_CINCdata <- colnames(CINCdata)[c(35:36,64:86,88:93,95:100,102:106,108:109,111:125,128:139,142:143,145:156,159:162,163:165,167:171,
                                       173:179,181:182,188:189,195:197,201,203,205:206,209:211,213,223:224,229:230,232,235:236,
                                       40,2:4,7,42,43,47,52,16,20,22,26,29,30,32,34,39,57)];
   cn_Model <- names(myModel$coeffs[2:length(myModel$coeffs)])
   cbind(cn_CINCdata,cn_Model,cn_CINCdata==cn_Model)
  #
  scores=myModel$coeffs[1]+CINCdata[,c(35:36,64:86,88:93,95:100,102:106,108:109,111:125,128:139,142:143,145:156,159:162,163:165,167:171,
                                       173:179,181:182,188:189,195:197,201,203,205:206,209:211,213,223:224,229:230,232,235:236,
                                       40,2:4,7,42,43,47,52,16,20,22,26,29,30,32,34,39,57)] %*% myModel$coeffs[2:146]
  #Debug: #return(CINCdata)
  scores <- plogis(scores)
  
  # Round the score
  scores <- round(scores,digits=3)
  # Zero the scores we don't know
  scores[is.na(scores)] <- 0
  labels <- scores > myModel$cutoff
  
  # Make the score
  results <- cbind(scores=scores, labels=labels)
  # The new format wants only a single value pair returned...
  results <- tail(results,1)
  return(results)
}

load_sepsis_model = function(){
  myModel<-NULL
  
  myModel$cutoff <- 0.024
  myModel$meanCINCdata <-
    structure(c(NA, 23.3029598706288, 83.9013786848645, 97.132856887831, 
                36.8524520620334, 123.628926432469, 82.6852667657605, 64.2408385399842, 
                18.5685149897509, 33.6377679385079, -0.269208651065925, 24.3155597166449, 
                0.490816311247089, 7.38834399377586, 40.6489923625897, 93.3661199916155, 
                138.777731125784, 22.4903402064341, 96.3410038538089, 8.04424197898551, 
                105.645915505339, 1.44279104523792, 1.13643110250603, 131.704734712249, 
                1.99958431567192, 2.03496467994132, 3.53991040560614, 4.09907387918333, 
                1.48745323062004, 5.74663085084951, 31.816220947285, 10.6502898235763, 
                37.058585286139, 11.1524420714392, 295.863585505111, 206.006290729936, 
                61.9907195016482, 0.555748401370719, 0.492498745699756, 0.507501254300244, 
                -53.0579266936389, 0, 59.8285537107514, 4.97801532537259, 0.703748669639609, 
                1.0511124416104, 0.72818279549807, 2.27594941992041, 0.50399374513594, 
                18.6155703408557, -0.041763563166269, 2.06544047949225, 0.295388967101313, 
                1.88379257877611, 2.4267954864037, 213.206527747929, 222.024205089219, 
                72.5665989299371, 190.415292598047, 18.821152941968, 23.9714702360733, 
                10.6208244863089, 16.5562713747581, 25.5489749093331, 14.1959984160646, 
                14.3820218357295), .Names = c("patient", "ICULOS", "HR", "O2Sat", 
                      "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", 
                      "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", 
                      "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", 
                      "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", 
                      "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", 
                      "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "SepsisLabel", 
                      "PulsePressure", "CO", "ShockIndex", "ModifiedShockIndex", "COvariation", 
                      "HRTemp", "RRTemp", "comb_RR", "RRdiff", "comb_PotassiumMagnesium", 
                      "comb_MagnesiumCalcium", "VS", "VSgap", "PF", "SF", "DO2", "HbBO2MAP", 
                      "UreaCreat", "UreaCreatsum", "comb_Hgb", "comb_HCO3Lac", "comb_HCO3Lacdiff", 
                      "AnionGap", "comb_ClpH"))
  
  myModel$sdCINCdata <- structure(c(NA, 19.7113781193856, 17.1412404132297, 3.08681722635008, 
                                    0.699052520108699, 23.0558486242605, 16.379442150324, 14.1497363788998, 
                                    4.99091472002804, 10.7611228130795, 3.98593913652066, 4.13573627070768, 
                                    0.182056970588601, 0.0649654528661226, 8.79724434667236, 10.0908941776371, 
                                    558.566340470898, 18.801076996622, 106.644309410576, 1.83135988647285, 
                                    5.51181244468682, 1.78666188910249, 2.77722929034579, 47.1640853311575, 
                                    1.62373454319323, 0.372872951629435, 1.30436563951624, 0.579326822102917, 
                                    3.12588569957376, 20.8736961497066, 5.65813896653889, 1.98656965776803, 
                                    21.5523796708516, 7.09366708527148, 149.276733043801, 101.471298684207, 
                                    16.4068825690898, 0.496882580249856, 0.49994402733963, 0.49994402733963, 
                                    152.932974303114, 0, 19.8195249050568, 1.84284267619292, 0.204160289169411, 
                                    0.2905788654929, 0.240111790237215, 0.456408566534132, 0.134931241195279, 
                                    1.06705150861474, 4.87437520674109, 0.436275380005987, 0.2886774288338, 
                                    11.4276609129278, 8.18220521365634, 72.3380641696101, 74.216192115408, 
                                    27.3534126098377, 17.0156753755032, 10.0624961772818, 20.0107428126304, 
                                    0.308323005954232, 9.39686790945583, 4.12463633093258, 5.37980726148614, 
                                    0.761162759324289), .Names = c("patient", "ICULOS", "HR", "O2Sat", 
                                       "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", 
                                       "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", 
                                       "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", 
                                       "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", 
                                       "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", 
                                       "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "SepsisLabel", 
                                       "PulsePressure", "CO", "ShockIndex", "ModifiedShockIndex", "COvariation", 
                                       "HRTemp", "RRTemp", "comb_RR", "RRdiff", "comb_PotassiumMagnesium", 
                                       "comb_MagnesiumCalcium", "VS", "VSgap", "PF", "SF", "DO2", "HbBO2MAP", 
                                       "UreaCreat", "UreaCreatsum", "comb_Hgb", "comb_HCO3Lac", "comb_HCO3Lacdiff", 
                                       "AnionGap", "comb_ClpH"))
  

  # myModel coefficients
  myModel$coeffs<- structure(c(-9.45898663673839, 0.000936600472173807, 0.0636915201884217, 
                               -0.0108944702804866, 0.0720082724308512, -0.0012831280123557, 
                               -0.211947386990562, 0.0610542285658345, 0.0535819588754014, 0.0306275629948522, 
                               0.0418757072099501, 0.36425246255102, 0.0700336712508687, 0.0770263283117363, 
                               0.11303962784189, 0.0538419379276473, 0.146275363991929, 0.0590293473202436, 
                               0.440309854615722, 0.196026963779952, 0.00417167102669667, 0.112099549298492, 
                               -0.0309333134524298, 0.0913343934576508, 0.0488488421490149, 
                               -0.0150912104995382, -0.0834783414970238, 0.0876206035028467, 
                               0.0947337639877968, 0.04042035415292, -0.0221807218010848, -0.528278439316815, 
                               0.0634477242520585, 0.182895031624972, -0.0520942079600786, -0.106815996673077, 
                               -0.164045992583646, -0.0742081620371027, 0.03497759605518, -0.0261985412294366, 
                               0.0295757224485244, 0.0480674509867079, 0.0863674080972734, -0.423711532792242, 
                               0.522831184919593, -0.230800340526507, 0.474998449854071, 0.0672940526211311, 
                               0.10820080819108, -0.18598603983762, -0.0697523694040128, -0.318528651194152, 
                               -0.10522560214083, 0.219437216140557, 0.0382817652447438, -0.0480783532640684, 
                               0.0212867314175437, 0.397222177614644, -0.280953061867819, 0.385990842058202, 
                               -0.0636786620268184, -0.00594409020033552, -0.19222226117483, 
                               -0.0540434788514775, -0.0977394259414036, 0.139682431844422, 
                               0.0787633928456701, -0.00197630190403897, -0.0496012727713387, 
                               -0.0635445679398898, 0.381407470050751, -0.0178185044279085, 
                               -0.212952098806186, 0.0235001255088798, 0.00691807733005713, 
                               0.0329793498878727, -0.0142624084498745, 0.00104029991093197, 
                               -0.0112795087463029, 0.0484060405553696, 0.0370149544051675, 
                               -0.085980179872051, 0.0311068587869179, 0.0274808039399266, 0.0574472115480914, 
                               0.080763285625516, -0.187590859875682, 0.0539295768112296, 0.0626027472540064, 
                               -0.0167452565141239, 0.0795644825006939, -0.356986808110634, 
                               0.0133361634330504, 0.0944482893499473, 0.301435404151453, 0.0434902077026457, 
                               -0.0465621974493991, 0.0557373008001718, -0.0309205442399979, 
                               -0.482393501086643, 0.0553123391635784, 0.0653958446663858, 0.145519426870609, 
                               0.0611097861113366, -0.0193224372662968, 0.0618493753490291, 
                               0.00907213609572258, -0.0142532922059549, -0.0334650760599731, 
                               0.0152287384209438, -8.9757012289199e-05, -0.0245981912496177, 
                               -0.000193238956408052, -0.0650376705374133, 0.0787376450833685, 
                               0.0928950120875324, 0.00394804132873114, -2.34889685886428e-06, 
                               0.0413786070111466, 0.000102761869151875, 0.0136101403505218, 
                               0.0780536129108435, -0.00980630000545708, -0.010893936767629, 
                               0.738281367379502, -0.015071699285119, -0.0577558275182888, 0.0100477908487329, 
                               -0.0444629081518658, 0.236078662430384, -0.00579312730774447, 
                               0.0373961830108559, 0.0695751981814091, 0.525479846106556, -1.11659902519107, 
                               -0.00886511658529477, -0.00146997050553036, 0.222118804630279, 
                               0.000330934428399878, -0.099722841498513, -0.00504746286595061, 
                               0.0183156608773489, 0.00138575102999937, -0.000319481983357545, 
                               -4.27594845468878e-05, 0.0130623339399634), .Names = c("(Intercept)", 
                                    "Age", "Gender", "numLabs", "abn_HR", "abn_O2Sat", "abn_Temp", 
                                    "abn_SBP", "abn_MAP", "abn_DBP", "abn_Resp", "abn_EtCO2", "abn_BaseExcess", 
                                    "abn_HCO3", "abn_FiO2", "abn_pH", "abn_PaCO2", "abn_SaO2", "abn_AST", 
                                    "abn_BUN", "abn_Alkalinephos", "abn_Calcium", "abn_Chloride", 
                                    "abn_Creatinine", "abn_Bilirubin_direct", "abn_Glucose", "abn_Magnesium", 
                                    "abn_Phosphate", "abn_Potassium", "abn_Bilirubin_total", "abn_TroponinI", 
                                    "abn_Hct", "abn_PTT", "abn_WBC", "abn_Fibrinogen", "abn_Platelets", 
                                    "abn_PulsePressure", "abn_CO", "abn_ModifiedShockIndex", "abn_COvariation", 
                                    "abn_HRTemp", "abn_RRTemp", "abn_comb_RR", "abn_comb_PotassiumMagnesium", 
                                    "abn_comb_MagnesiumCalcium", "abn_PF", "abn_SF", "abn_DO2", "abn_HbBO2MAP", 
                                    "abn_UreaCreat", "abn_UreaCreatsum", "abn_comb_Hgb", "abn_comb_HCO3Lac", 
                                    "abn_comb_HCO3Lacdiff", "abn_AnionGap", "abn_comb_ClpH", "az_Age", 
                                    "az_HR", "az_O2Sat", "az_Temp", "az_DBP", "az_Resp", "az_EtCO2", 
                                    "az_BaseExcess", "az_HCO3", "az_FiO2", "az_pH", "az_PaCO2", "az_SaO2", 
                                    "az_AST", "az_BUN", "az_Alkalinephos", "az_Creatinine", "az_Bilirubin_direct", 
                                    "az_Lactate", "az_Magnesium", "az_Phosphate", "az_Potassium", 
                                    "az_Bilirubin_total", "az_TroponinI", "az_Hct", "az_Hgb", "az_PTT", 
                                    "az_WBC", "az_Fibrinogen", "az_Platelets", "az_ShockIndex", "az_ModifiedShockIndex", 
                                    "az_COvariation", "az_HRTemp", "az_RRTemp", "az_comb_RR", "az_RRdiff", 
                                    "az_comb_MagnesiumCalcium", "az_VS", "az_PF", "az_SF", "az_DO2", 
                                    "az_UreaCreat", "az_UreaCreatsum", "az_comb_Hgb", "az_comb_HCO3Lac", 
                                    "az_comb_HCO3Lacdiff", "az_AnionGap", "az_comb_ClpH", "Delta_O2Sat", 
                                    "Delta_Temp", "Delta_BaseExcess", "Delta_HCO3", "Delta_BUN", 
                                    "Delta_Alkalinephos", "Delta_Calcium", "Delta_Glucose", "Delta_Magnesium", 
                                    "Delta_Potassium", "Delta_Bilirubin_total", "Delta_Hgb", "Delta_PTT", 
                                    "Delta_WBC", "Delta_Platelets", "Delta_comb_PotassiumMagnesium", 
                                    "Delta_comb_MagnesiumCalcium", "Delta_HbBO2MAP", "Delta_UreaCreat", 
                                    "Delta_comb_Hgb", "Delta_AnionGap", "Delta_comb_ClpH", "ICULOS", 
                                    "O2Sat", "Temp", "SBP", "Resp", "CO", "ShockIndex", "RRTemp", 
                                    "VS", "BUN", "Creatinine", "Glucose", "Potassium", "Hct", "Hgb", 
                                    "WBC", "Platelets", "HospAdmTime", "UreaCreat"))

  myModel$f <- " ~ Age + Gender + numLabs + abn_HR + abn_O2Sat + abn_Temp + 
    abn_SBP + abn_MAP + abn_DBP + abn_Resp + abn_EtCO2 + abn_BaseExcess + 
  abn_HCO3 + abn_FiO2 + abn_pH + abn_PaCO2 + abn_SaO2 + abn_AST + 
  abn_BUN + abn_Alkalinephos + abn_Calcium + abn_Chloride + 
  abn_Creatinine + abn_Bilirubin_direct + abn_Glucose + abn_Magnesium + 
  abn_Phosphate + abn_Potassium + abn_Bilirubin_total + abn_TroponinI + 
  abn_Hct + abn_PTT + abn_WBC + abn_Fibrinogen + abn_Platelets + 
  abn_PulsePressure + abn_CO + abn_ModifiedShockIndex + abn_COvariation + 
  abn_HRTemp + abn_RRTemp + abn_comb_RR + abn_comb_PotassiumMagnesium + 
  abn_comb_MagnesiumCalcium + abn_PF + abn_SF + abn_DO2 + abn_HbBO2MAP + 
  abn_UreaCreat + abn_UreaCreatsum + abn_comb_Hgb + abn_comb_HCO3Lac + 
  abn_comb_HCO3Lacdiff + abn_AnionGap + abn_comb_ClpH + az_Age + 
  Gender + numLabs + az_HR + az_O2Sat + az_Temp + az_DBP + 
  az_Resp + az_EtCO2 + az_BaseExcess + az_HCO3 + az_FiO2 + 
  az_pH + az_PaCO2 + az_SaO2 + az_AST + az_BUN + az_Alkalinephos + 
  az_Creatinine + az_Bilirubin_direct + az_Lactate + az_Magnesium + 
  az_Phosphate + az_Potassium + az_Bilirubin_total + az_TroponinI + 
  az_Hct + az_Hgb + az_PTT + az_WBC + az_Fibrinogen + az_Platelets + 
  az_ShockIndex + az_ModifiedShockIndex + az_COvariation + 
  az_HRTemp + az_RRTemp + az_comb_RR + az_RRdiff + az_comb_MagnesiumCalcium + 
  az_VS + az_PF + az_SF + az_DO2 + az_UreaCreat + az_UreaCreatsum + 
  az_comb_Hgb + az_comb_HCO3Lac + az_comb_HCO3Lacdiff + az_AnionGap + 
  az_comb_ClpH + Gender + numLabs + +Delta_O2Sat + Delta_Temp + 
  Delta_BaseExcess + Delta_HCO3 + Delta_BUN + Delta_Alkalinephos + 
  Delta_Calcium + Delta_Glucose + Delta_Magnesium + Delta_Potassium + 
  Delta_Bilirubin_total + Delta_Hgb + Delta_PTT + Delta_WBC + 
  Delta_Platelets + Delta_comb_PotassiumMagnesium + Delta_comb_MagnesiumCalcium + 
  Delta_HbBO2MAP + Delta_UreaCreat + Delta_comb_Hgb + Delta_AnionGap + 
  Delta_comb_ClpH + Age + Gender + numLabs + ICULOS + O2Sat + 
  Temp + SBP + Resp + CO + ShockIndex + RRTemp + VS + BUN + 
  Creatinine + Glucose + Potassium + Hct + Hgb + WBC + Platelets + 
  HospAdmTime + UreaCreat"
  myModel$f <- as.formula(myModel$f)
  
  myModel$normdf <- structure(c(60, 100, 90, 100, 36.6, 37, 95, 140, 70, 100, 60, 
                                90, 12, 20, 30, 43, -2, 2, 22, 28, 0.21, 0.35, 7.35, 7.45, 38, 
                                42, 94, 100, 100, 200, 7, 21, 44, 147, 8.5, 10.2, 96, 106, 0.5, 
                                1.2, 0, 0.3, 70, 130, 0.5, 2, 1.7, 2.2, 2.5, 4.5, 3.6, 5.2, 0.1, 
                                1.2, 0, 0.045, 45, 52, 13.8, 17.2, 25, 35, 4.5, 11, 149, 353, 
                                150, 450, 30, 50, 3.5, 8, 0.5, 1, 0.85, 1.3, 0.45, 0.7, 1.6, 
                                2.75, 0.27, 0.54, 12, 20, -2, 2, 4.8, 8.5, 1.6, 3, -5, 10, -3, 
                                10, 300, 500, 300, 500, 100, 200, 174, 217, 6, 42, 7.5, 22.2, 
                                13.8, 17.2, 11, 56, 20, 27.5, 8, 16, 13.15, 14.42), .Dim = c(2L, 58L), 
                              .Dimnames = list(NULL, c("HR", "O2Sat", "Temp", "SBP", 
                                                    "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", 
                                                    "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", 
                                                    "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", 
                                                    "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", 
                                                    "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets", "PulsePressure", 
                                                    "CO", "ShockIndex", "ModifiedShockIndex", "COvariation", "HRTemp", 
                                                    "RRTemp", "comb_RR", "RRdiff", "comb_PotassiumMagnesium", "comb_MagnesiumCalcium", 
                                                    "VS", "VSgap", "PF", "SF", "DO2", "HbBO2MAP", "UreaCreat", "UreaCreatsum", 
                                                    "comb_Hgb", "comb_HCO3Lac", "comb_HCO3Lacdiff", "AnionGap", "comb_ClpH")))
  return(myModel)
}

