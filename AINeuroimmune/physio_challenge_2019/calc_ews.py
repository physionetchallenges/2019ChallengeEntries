'''
BSD 2-Clause License

Copyright (c) 2019, PhysioNet
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

def mews(rr, temp, sbp, hr):
	rr_score = 0
	if (rr == rr):
		if rr <= 8:
			rr_score = 2
		elif 8 < rr and rr <= 14:
			rr_score = 0
		elif 14 < rr and rr <= 20:
			rr_score = 1
		elif 20 < rr and rr <=29:
			rr_score = 2
		else:
			rr_score = 3
			
	temp_score = 0
	if (temp == temp):
		if temp <= 35:
			temp_score = 2
		elif 35 < temp and temp <= 38.4:
			temp_score = 0
		else:
			temp_score = 2

	sbp_score = 0
	if (sbp == sbp):
		if sbp <= 70:
			sbp_score = 3
		elif 70 < sbp and sbp <= 80:
			sbp_score = 2
		elif 80 < sbp and sbp <= 100:
			sbp_score = 1
		elif 100 < sbp and sbp <= 199:
			sbp_score = 0			
		else:
			sbp_score = 2

	hr_score = 0
	if (hr == hr):
		if hr <= 40:
			hr_score = 2
		elif 40 < hr and hr <= 50:
			hr_score = 1
		elif 50 < hr and hr <= 100:
			hr_score = 0
		elif 100 < hr and hr <= 110:
			hr_score = 1
		elif 110 < hr and hr <= 130:
			hr_score = 2
		else:
			hr_score = 3				

	return rr_score + temp_score + sbp_score + hr_score


def news(rr, o2sat, temp, sbp, hr):
	rr_score = 0
	if (rr == rr):
		if rr <= 8:
			rr_score = 3
		elif 8 < rr and rr <= 11:
			rr_score = 1
		elif 11 < rr and rr <= 20:
			rr_score = 0
		elif 20 < rr and rr <=24:
			rr_score = 2
		else:
			rr_score = 3
			
	o2_score = 0
	if (o2sat == o2sat):
		if o2sat <= 91:
			o2_score = 3
		elif 91 < o2sat and o2sat <= 93:
			o2_score = 2
		elif 93 < o2sat and o2sat <= 95:
			o2_score = 1
		else:
			o2_score = 0

	temp_score = 0
	if (temp == temp):
		if temp <= 35:
			temp_score = 3
		elif 35 < temp and temp <= 36:
			temp_score = 1
		elif 36 < temp and temp <= 38:
			temp_score = 0
		elif 38 < temp and temp <= 39:
			temp_score = 1			
		else:
			temp_score = 2

	sbp_score = 0
	if (sbp == sbp):
		if sbp <= 90:
			sbp_score = 3
		elif 90 < sbp and sbp <= 100:
			sbp_score = 2
		elif 100 < sbp and sbp <= 110:
			sbp_score = 1
		elif 110 < sbp and sbp <= 219:
			sbp_score = 0			
		else:
			sbp_score = 3

	hr_score = 0
	if (hr == hr):
		if hr <= 40:
			hr_score = 3
		elif 40 < hr and hr <= 50:
			hr_score = 1
		elif 50 < hr and hr <= 90:
			hr_score = 0
		elif 90 < hr and hr <= 110:
			hr_score = 1
		elif 110 < hr and hr <= 130:
			hr_score = 2
		else:
			hr_score = 3				

	return rr_score + o2_score + temp_score + sbp_score + hr_score


def q_sofa(gcs, rr, sbp):
    q_score = 0
    if not (gcs != gcs):
        if gcs < 15: 
            q_score += 1
    if not (rr != rr):
        if rr >= 22: 
            q_score += 1 
    if not (sbp != sbp):
        if sbp <= 100: 
            q_score += 1
    return q_score   
	
	
def sofa(fio2, pao2, vent, platelets, gcs, bilirubin_direct, bilirubin_total, mabp,
         dopamine, dobutamine, epinephrine, norepinephrine, creatinine):
    rr_score = 0
    rr = min(fio2, pao2)
    if (rr == rr): 
        if vent and rr < 100:
            rr_score = 4
        elif vent and rr < 200:
            rr_score = 3
        elif rr < 300:
            rr_score = 2
        elif rr < 400:
            rr_score = 1  
            
    nerv_score = 0
    if (gcs == gcs): 
        if gcs < 6:
            nerv_score = 4
        elif 6 <= gcs and gcs < 10:
            nerv_score = 3
        elif 10 <= gcs and gcs < 13:
            nerv_score = 2
        elif 13 <= gcs and gcs <= 14:
            nerv_score = 1             
 
    liver_score = 0
    if (bilirubin_direct == bilirubin_direct) and (bilirubin_total == bilirubin_total):
        bilirubin = max(bilirubin_direct, bilirubin_total)
    elif (bilirubin_direct == bilirubin_direct):
        bilirubin = bilirubin_direct
    else:
        bilirubin = bilirubin_total
		
    if (bilirubin == bilirubin): 
        if bilirubin < 1.2:
            liver_score = 0
        elif 1.2 <= bilirubin and bilirubin < 2:
            liver_score = 1
        elif 2 <= bilirubin and bilirubin < 6:
            liver_score = 2
        elif 6 <= bilirubin and bilirubin < 12:
            liver_score = 3 
        else:
            liver_score = 4		
            
    kidney_score = 0
    if (creatinine == creatinine): 
        if creatinine < 5:
            kidney_score = 4
        elif 3.5 <= creatinine and creatinine < 5:
            kidney_score = 3
        elif 2 <= creatinine and creatinine < 3.5:
            kidney_score = 2
        elif 1.2 <= creatinine and creatinine < 2:
            kidney_score = 1              
            
    cardio_score = 0
    if (((dopamine == dopamine) and (dopamine > 15))
        or ((epinephrine == epinephrine) and (epinephrine > 0.1))
        or ((norepinephrine == norepinephrine) and (norepinephrine > 0.1))):
            cardio_score = 4
    elif (((dopamine == dopamine) and (dopamine > 5))
        or ((epinephrine == epinephrine) and (epinephrine <= 0.1))
        or ((norepinephrine == norepinephrine) and (norepinephrine <= 0.1))):  
            cardio_score = 3  
    elif (((dopamine == dopamine) and (dopamine <= 5))
        or (dobutamine == dobutamine)):  
            cardio_score = 2             
    elif ((mabp == mabp) and mabp < 70):  
            cardio_score = 1                 
            
    return  rr_score + nerv_score + liver_score + kidney_score + cardio_score  	