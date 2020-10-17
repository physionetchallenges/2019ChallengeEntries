import numpy as np

hr_b, hr_u = 30, 200
temp_b, temp_u = 30, 45
sbp_b, sbp_u = 40, 200
map_b, map_u = 40, 200
dbp_b, dbp_u = 40, 200
resp_b, resp_u = 1, 60
etco_b, etco_u = 1, 90
b_ex_b, b_ex_u = None, 40
fio2_b, fio2_u = 0, 1 # Set B max: 4000, min: -50
chlo_b, chlo_u = 60, None
crea_b, crea_u = 1, 30
bi_d_b, bi_d_u = None, 25
pota_b, pota_u = 1, 10
hgb_b, hgb_u = None, 25
ptt_b, ptt_u = None, 150
wbc_b, wbc_u = None, 300
fibr_b, fibr_u = None, 1000
plat_b, plat_u = None, 1000

def process_func1(ele):
    # Clip
    ele[:, 0] = np.clip(ele[:, 0], hr_b, hr_u) # HR
    ele[:, 2] = np.clip(ele[:, 2], temp_b, temp_u) # Temp
    ele[:, 3] = np.clip(ele[:, 3], sbp_b, sbp_u) # SBP
    ele[:, 4] = np.clip(ele[:, 4], map_b, map_u) # MAP
    ele[:, 5] = np.clip(ele[:, 5], dbp_b, dbp_u) # DBP
    ele[:, 6] = np.clip(ele[:, 6], resp_b, resp_u) # Resp
    ele[:, 7] = np.clip(ele[:, 7], etco_b, etco_u) # EtCO2
    ele[:, 8] = np.clip(ele[:, 8], b_ex_b, b_ex_u) # BaseExcess
    ele[:, 10] = np.clip(ele[:, 10], fio2_b, fio2_u) # FiO2
    ele[:, 18] = np.clip(ele[:, 18], chlo_b, chlo_u) # Chloride
    ele[:, 19] = np.clip(ele[:, 19], crea_b, crea_u) # Creatinine
    ele[:, 20] = np.clip(ele[:, 20], bi_d_b, bi_d_u) # Bilirubin_direct
    ele[:, 25] = np.clip(ele[:, 25], pota_b, pota_u) # Potassium
    ele[:, 29] = np.clip(ele[:, 29], hgb_b, hgb_u) # Hgb
    ele[:, 30] = np.clip(ele[:, 30], ptt_b, ptt_u) # PTT
    ele[:, 31] = np.clip(ele[:, 31], wbc_b, wbc_u) # WBC
    ele[:, 32] = np.clip(ele[:, 32], fibr_b, fibr_u) # Fibrinogen
    ele[:, 33] = np.clip(ele[:, 33], plat_b, plat_u) # Platelets
    return ele

def process_func2(ele):
    # Log scale
    ele[:, 5] = np.log1p(ele[:, 5]) # DBP
    ele[:, 6] = np.log1p(ele[:, 6]) # Resp
    ele[:, 12] = np.log1p(ele[:, 12]) # PaCO2
    ele[:, 14] = np.log1p(ele[:, 14]) # AST
    ele[:, 15] = np.log1p(ele[:, 15]) # BUN
    ele[:, 16] = np.log1p(ele[:, 16]) # Alkalinephos
    ele[:, 19] = np.log1p(ele[:, 19]) # Creatinine
    ele[:, 20] = np.log1p(ele[:, 20]) # Bilirubin_direct
    ele[:, 21] = np.log1p(ele[:, 21]) # Glucose
    ele[:, 22] = np.log1p(ele[:, 22]) # Lactate
    ele[:, 23] = np.log1p(ele[:, 23]) # Magnesium
    ele[:, 24] = np.log1p(ele[:, 24]) # Phosphate
    ele[:, 25] = np.log1p(ele[:, 25]) # Potassium
    ele[:, 26] = np.log1p(ele[:, 26]) # Bilirubin_total
    ele[:, 30] = np.log1p(ele[:, 30]) # PTT
    ele[:, 31] = np.log1p(ele[:, 31]) # WBC
    ele[:, 32] = np.log1p(ele[:, 32]) # Fibrinogen
    ele[:, 33] = np.log1p(ele[:, 33]) # Platelets
    return ele

o2sat_max = 101
sao2_max = 101
def process_func3(ele):
    ele[:, 1] = np.clip(ele[:, 1], 0, 100)
    ele[:, 13] = np.clip(ele[:, 13], 0, 100)
    ele[:, 1] = np.log1p(o2sat_max - ele[:, 1])
    ele[:, 13] = np.log1p(sao2_max - ele[:, 13])
    return ele

ph_min, ph_max = 1, 15
hco3_b, hco3_u = 1, 50
sao2_b, sao2_u = 40, 100
ast_b, ast_u = 0, 10000
bun_b, bun_u = 0, 160
glu_b, glu_u = 0, 500
lac_b, lac_u = 0, 25
mag_b, mag_u = 0, 5
phos_b, phos_u = 0, 20 # Need to check conversion from MIMIC
bi_t_b, bi_t_u = 0, 40
hct_b, hct_u = 0, 50
def process_func4(ele):
    ele[:, 9] = np.clip(ele[:, 9], hco3_b, hco3_u) # HCO3
    ele[:, 11] = np.clip(ele[:, 11], ph_min, ph_max) # pH
    ele[:, 13] = np.clip(ele[:, 13], sao2_b, sao2_u) # SaO2
    ele[:, 14] = np.clip(ele[:, 14], ast_b, ast_u) # AST
    ele[:, 15] = np.clip(ele[:, 15], bun_b, bun_u) # BUN
    ele[:, 21] = np.clip(ele[:, 21], glu_b, glu_u) # Glucose
    ele[:, 22] = np.clip(ele[:, 22], lac_b, lac_u) # Lactose
    ele[:, 23] = np.clip(ele[:, 23], mag_b, mag_u) # Magnesium
    ele[:, 24] = np.clip(ele[:, 24], phos_b, phos_u) # Phosphate
    ele[:, 26] = np.clip(ele[:, 26], bi_t_b, bi_t_u) # Bilirubin Total
    ele[:, 28] = np.clip(ele[:, 28], hct_b, hct_u) # HCT
    return ele

def manual_processor_v4(data):
    data = process_func1(data)
    data = process_func4(data)
    data = process_func2(data)
    data = process_func3(data)
    return data
