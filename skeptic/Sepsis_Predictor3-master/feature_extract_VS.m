function F = feature_extract_VS(raw_data, normal_range)
% Copyright 2019, TATA Consultancy Services. All rights reserved.

% Total 150 features

l = size(raw_data,1);

cat_data = zeros(size(raw_data,1),34);
Ulimit_data = zeros(size(raw_data,1),34);
Llimit_data = zeros(size(raw_data,1),34);

% define limits of the categories
very_low_lim = repmat(normal_range(1,:)-(0.20*normal_range(1,:)),l,1);
lower_lim = repmat(normal_range(1,:),l,1);
upper_lim = repmat(normal_range(2,:),l,1);
very_high_lim = repmat(normal_range(2,:)+(0.20*normal_range(2,:)),l,1);

% 34 +34 +34 divide into categories
normal_cat = (raw_data(:,1:34) >= lower_lim) & (raw_data(:,1:34) <= upper_lim);
cat_data(normal_cat) = 1;
Ulimit_data(normal_cat) = upper_lim(normal_cat);
Llimit_data(normal_cat) = lower_lim(normal_cat);

low_cat = (raw_data(:,1:34) > very_low_lim) & (raw_data(:,1:34) < lower_lim);
cat_data(low_cat) = 2;
Ulimit_data(low_cat) = upper_lim(low_cat);
Llimit_data(low_cat) = lower_lim(low_cat);

very_low_cat = (raw_data(:,1:34) <= very_low_lim);
cat_data(very_low_cat) = 3;
Ulimit_data(very_low_cat) = upper_lim(very_low_cat);
Llimit_data(very_low_cat) = lower_lim(very_low_cat);

high_cat = (raw_data(:,1:34) > upper_lim) & (raw_data(:,1:34) < very_high_lim);
cat_data(high_cat) = 4;
Ulimit_data(high_cat) = upper_lim(high_cat);
Llimit_data(high_cat) = lower_lim(high_cat);

very_high_cat = (raw_data(:,1:34) >= very_high_lim);
cat_data(very_high_cat) = 5;
Ulimit_data(very_high_cat) = upper_lim(very_high_cat);
Llimit_data(very_high_cat) = lower_lim(very_high_cat);

% ignore BUN & Potassium (-2 Features)
cat_data(:,[16 26]) = [];

% 34 dummy features
dummy_feat = ~isnan(raw_data(:,1:34));

% ignore BUN & Potassium (-2 Features)
dummy_feat(:,[16 26]) = [];

% 3 AgeFeatures: dividing age into categories
age = raw_data(:,35);
age_cat = zeros(size(age,1),3);

very_low_age = age<=25 & age>=1;
if very_low_age
    age_cat(very_low_age,:) = repmat([1 1 25],sum(double(very_low_age)),1);
end

low_age = age<=50 & age>25;
if low_age
    age_cat(low_age,:) = repmat([2 25 50],sum(double(low_age)),1);
end

high_age = age<=75 & age>50;
if high_age
    age_cat(high_age,:) = repmat([3 50 75],sum(double(high_age)),1);
end

very_high_age = age<=100 & age>75;
if very_high_age
    age_cat(very_high_age,:) = repmat([4 75 100],sum(double(very_high_age)),1);
end

% 2 icu limit features
icuhrs =raw_data(:,40);
a = 0:10:500;
b = a(abs(a-icuhrs)==min(abs(a-icuhrs)));
if b(1)-icuhrs < 0
    LL = b(1);
    UL = b(1)+10;
else
    LL = b(1)-10;
    UL = b(1);
end
iculim = [LL UL];
%================================================%
HR_mews_data = zeros(size(raw_data,1),1);
resp_mews_data = zeros(size(raw_data,1),1);
SBP_mews_data = zeros(size(raw_data,1),1);
temp_mews_data = zeros(size(raw_data,1),1);

% divide into categories
%===========================
HR_mews1 = (raw_data(:,1) <= 40) | ((raw_data(:,1) < 111)& (raw_data(:,1) < 129));
HR_mews2 = ((raw_data(:,1) > 41) & (raw_data(:,1) <50)) | ((raw_data(:,1) < 110) & (raw_data(:,6) > 101));
HR_mews3 = (raw_data(:,1) > 51) & (raw_data(:,1) < 100);
HR_mews4 = (raw_data(:,1) >= 130);
HR_mews_data(HR_mews1) = 2;
HR_mews_data(HR_mews2) = 1;
HR_mews_data(HR_mews3) = 4;
HR_mews_data(HR_mews4) = 3;
%===============================
SBP_mews1 = (raw_data(:,4) > 81)& (raw_data(:,4) < 100);
SBP_mews2 = ((raw_data(:,4) > 70) & (raw_data(:,4) < 80)) | (raw_data(:,6) >= 200);
SBP_mews3 = (raw_data(:,4) <= 70);
SBP_mews4 = (raw_data(:,4) > 101)& (raw_data(:,4) < 199);
SBP_mews_data(SBP_mews1) = 1;
SBP_mews_data(SBP_mews2) = 2;
SBP_mews_data(SBP_mews3) = 3;
SBP_mews_data(SBP_mews4) = 4;

%==============================
resp_mews1 = (raw_data(:,7) < 9) | ((raw_data(:,7) > 21)& (raw_data(:,7) < 29));
resp_mews2 =  (raw_data(:,7) > 15)& (raw_data(:,7) < 20);
resp_mews3 =  (raw_data(:,7) > 9)& (raw_data(:,7) < 14);
resp_mews4 = (raw_data(:,7) >= 30);
resp_mews_data(resp_mews1) = 2;
resp_mews_data(resp_mews2) = 1;
resp_mews_data(resp_mews3) = 4;
resp_mews_data(resp_mews4) = 3;
%================================

temp_mews1 = (raw_data(:,3) < 35) | (raw_data(:,3) >= 38.5);
temp_mews2 = (raw_data(:,3) > 35) & (raw_data(:,3) < 38.4);
temp_mews_data(temp_mews1) = 1;
temp_mews_data(temp_mews2) = 2;
% 4 MEWS + 1 SUM(MEWS) = Total 5
MEWS =sum( [resp_mews_data HR_mews_data SBP_mews_data temp_mews_data]);
MEWS1 = [resp_mews_data HR_mews_data SBP_mews_data temp_mews_data];

%=========NEWS-2=======================
resp_news_data = zeros(size(raw_data,1),1);
O2sat_news_data = zeros(size(raw_data,1),1);
temp_news_data = zeros(size(raw_data,1),1);
HR_news_data = zeros(size(raw_data,1),1);
SBP_news_data = zeros(size(raw_data,1),1);
%==============================
resp_news1 = (raw_data(:,7) > 9) & (raw_data(:,7) < 11);
resp_news2 =  (raw_data(:,7) > 21)& (raw_data(:,7) < 24);
resp_news3 =  (raw_data(:,7) <= 8) | (raw_data(:,7) >=25);
resp_news4 = (raw_data(:,7) > 12) & (raw_data(:,7) < 20);
resp_news_data(resp_news1) = 1;
resp_news_data(resp_news2) = 2;
resp_news_data(resp_news3) = 3;
resp_news_data(resp_news4) = 4;


O2sat_news1 = (raw_data(:,2) > 94) & (raw_data(:,2) < 95);
O2sat_news2 =  (raw_data(:,2) > 92)& (raw_data(:,2) < 93);
O2sat_news3 =  (raw_data(:,2) <= 91);
O2sat_news4 = (raw_data(:,2) >= 96);
O2sat_news_data(O2sat_news1) = 1;
O2sat_news_data(O2sat_news2) = 2;
O2sat_news_data(O2sat_news3) = 3;
O2sat_news_data(O2sat_news4) = 4;

temp_news1 = ((raw_data(:,4) > 35.1) & (raw_data(:,4) < 36.0)) | ((raw_data(:,4) > 38.1) & (raw_data(:,4) < 39.0));
temp_news2 = (raw_data(:,4) >= 39.1);
temp_news3 = (raw_data(:,4) <= 35);
temp_news4 = (raw_data(:,4) > 36.1) & (raw_data(:,4) < 38);
temp_news_data(temp_news1) = 1;
temp_news_data(temp_news2) = 2;
temp_news_data(temp_news3) = 3;
temp_news_data(temp_news4) = 4;


HR_news1 = ((raw_data(:,1) > 41) & (raw_data(:,1) < 50)) | ((raw_data(:,1) > 91 ) & (raw_data(:,1) <110));
HR_news2 = (raw_data(:,1) > 111) & (raw_data(:,1) < 130 );
HR_news3 = (raw_data(:,1) <=40) | (raw_data(:,1) >= 131);
HR_news4 = (raw_data(:,1) > 51) & (raw_data(:,1) < 90);
HR_news_data(HR_news1) = 1;
HR_news_data(HR_news2) = 2;
HR_news_data(HR_news3) = 3;
HR_news_data(HR_news4) = 4;
%===============================
SBP_news1 = (raw_data(:,4) > 101)& (raw_data(:,4) < 110);
SBP_news2 = (raw_data(:,4) > 91)& (raw_data(:,4) < 100);
SBP_news3 = (raw_data(:,4) <= 90) | (raw_data(:,1) >= 220);
SBP_news4 = (raw_data(:,4) > 111)& (raw_data(:,4) < 219);

SBP_news_data(SBP_news1) = 1;
SBP_news_data(SBP_news2) = 2;
SBP_news_data(SBP_news3) = 3;
SBP_news_data(SBP_news4) = 4;

% 5 NEWS +1 SUM(NEWS) = TOTAL 6 
NEWS = sum([resp_news_data O2sat_news_data temp_news_data HR_news_data SBP_news_data]);
NEWS1 = [resp_news_data O2sat_news_data temp_news_data HR_news_data SBP_news_data];

% APACHE II
%Total 8+1 = 9;
% temp_APACHE_data = zeros(size(raw_data,1),1);
% MAP_APACHE_data = zeros(size(raw_data,1),1);
% HR_APACHE_data = zeros(size(raw_data,1),1);
% resp_APACHE_data = zeros(size(raw_data,1),1);
% pH_APACHE_data = zeros(size(raw_data,1),1);
% potass_APACHE_data = zeros(size(raw_data,1),1);
% Creatinine_APACHE_data = zeros(size(raw_data,1),1);
% AGE_APACHE_data = zeros(size(raw_data,1),1);
% 
% temp_APACHE1 = ((raw_data(:,4) > 38.5) & (raw_data(:,4) < 38.9)) | ((raw_data(:,4) > 34) & (raw_data(:,4) < 35.9));
% temp_APACHE2 = (raw_data(:,4) > 32) & (raw_data(:,4) < 33.9);
% temp_APACHE3 = ((raw_data(:,4) > 39) & (raw_data(:,4) < 40.9)) | ((raw_data(:,4) > 30) & (raw_data(:,4) < 31.9));
% temp_APACHE4 = (raw_data(:,4) >= 41) | (raw_data(:,4) <= 29.9);
% temp_APACHE5 = (raw_data(:,4) > 36) & (raw_data(:,4) < 38.4);
% temp_APACHE_data(temp_APACHE1) = 1;
% temp_APACHE_data(temp_APACHE2) = 2;
% temp_APACHE_data(temp_APACHE3) = 3;
% temp_APACHE_data(temp_APACHE4) = 4;
% temp_APACHE_data(temp_APACHE5) = 0;
% 
% MAP_APACHE2 = ((raw_data(:,5) > 110) & (raw_data(:,5) < 129)) | ((raw_data(:,5) > 50) & (raw_data(:,5) < 69));
% MAP_APACHE3 = (raw_data(:,5) > 130) & (raw_data(:,5) < 159);
% MAP_APACHE4 = (raw_data(:,5) <= 49) | (raw_data(:,5) >= 160);
% MAP_APACHE5 = (raw_data(:,5) > 70) & (raw_data(:,5) < 109);
% MAP_APACHE_data(MAP_APACHE2) = 2;
% MAP_APACHE_data(MAP_APACHE3) = 3;
% MAP_APACHE_data(MAP_APACHE4) = 4;
% MAP_APACHE_data(MAP_APACHE5) = 0;
% 
% HR_APACHE5 = (raw_data(:,1) > 70) & (raw_data(:,1) < 109);
% HR_APACHE2 = ((raw_data(:,1) > 110) & (raw_data(:,1) < 139)) | ((raw_data(:,1) > 55) & (raw_data(:,5) < 69));
% HR_APACHE3 = ((raw_data(:,1) > 140) & (raw_data(:,1) < 179)) | ((raw_data(:,1) > 40) & (raw_data(:,5) < 54));
% HR_APACHE4 = (raw_data(:,1) >= 180) | (raw_data(:,1) <= 39);
% HR_APACHE_data(HR_APACHE5) = 0;
% HR_APACHE_data(HR_APACHE2) = 2;
% HR_APACHE_data(HR_APACHE3) = 3;
% HR_APACHE_data(HR_APACHE4) = 4;
% 
% resp_APACHE1 = ((raw_data(:,7) > 25) & (raw_data(:,7) < 34)) | ((raw_data(:,7) > 10) & (raw_data(:,7) < 11));
% resp_APACHE2 =  (raw_data(:,7) > 6) & (raw_data(:,7) < 9);
% resp_APACHE3 =  (raw_data(:,7) > 35) & (raw_data(:,7) < 49);
% resp_APACHE4 = (raw_data(:,7) >= 50) | (raw_data(:,7) <= 5);
% resp_APACHE5 = (raw_data(:,7) > 12) & (raw_data(:,7) < 24);
% resp_APACHE_data(resp_APACHE1) = 1;
% resp_APACHE_data(resp_APACHE2) = 2;
% resp_APACHE_data(resp_APACHE3) = 3;
% resp_APACHE_data(resp_APACHE4) = 4;
% resp_APACHE_data(resp_APACHE5) = 0;
% 
% pH_APACHE1 = ((raw_data(:,12) > 7.5) & (raw_data(:,12) < 7.59));
% pH_APACHE2 = ((raw_data(:,12) > 7.25) & (raw_data(:,12) < 7.32));
% pH_APACHE3 = ((raw_data(:,12) > 7.6) & (raw_data(:,12) < 7.69)) | ((raw_data(:,6) > 7.15) & (raw_data(:,6) < 7.24));
% pH_APACHE4 = ((raw_data(:,12) >= 7.7) | (raw_data(:,12) < 7.15));
% pH_APACHE5 = ((raw_data(:,12) > 7.33) & (raw_data(:,12) < 7.49));
% pH_APACHE_data(pH_APACHE1) = 1;
% pH_APACHE_data(pH_APACHE2) = 2;
% pH_APACHE_data(pH_APACHE3) = 3;
% pH_APACHE_data(pH_APACHE4) = 4;
% pH_APACHE_data(pH_APACHE5) = 0;
% 
% Potass = raw_data(:,26);
% potass_APACHE0 = (Potass >3.5) & (Potass < 5.4);
% potass_APACHE1 = ((Potass >5.5) & (Potass < 5.9)) | ((Potass >3) & (Potass < 3.4));
% potass_APACHE2 = ((Potass >2.5) & (Potass < 9.9));
% potass_APACHE3 = ((Potass > 6) & (Potass < 6.9)) ;
% potass_APACHE4 = (Potass >= 7);
% potass_APACHE_data(potass_APACHE0) = 0;
% potass_APACHE_data(potass_APACHE1) = 1;
% potass_APACHE_data(potass_APACHE2) = 2;
% potass_APACHE_data(potass_APACHE3) = 3;
% potass_APACHE_data(potass_APACHE4) = 4;
% 
% Creatinine = raw_data(:,20);
% Creatinine_APACHE0 = (Creatinine >0.6) & (Creatinine < 1.4);
% Creatinine_APACHE2 = ((Creatinine >1.5) & (Creatinine < 1.9)) | (Creatinine < 0.6);
% Creatinine_APACHE3 = ((Creatinine >2) & (Creatinine < 3.4));
% Creatinine_APACHE4 = (Creatinine >= 3.5);
% Creatinine_APACHE_data(Creatinine_APACHE0) = 0;
% Creatinine_APACHE_data(Creatinine_APACHE2) = 4;
% Creatinine_APACHE_data(Creatinine_APACHE3) = 6;
% Creatinine_APACHE_data(Creatinine_APACHE4) = 8;
% 
% AGE= raw_data(:,26);
% AGE_APACHE0 = (AGE <= 44);
% AGE_APACHE2 = (AGE > 45) & (AGE < 54);
% AGE_APACHE3 = (AGE > 45) & (AGE < 54);
% AGE_APACHE5 = (AGE > 55) & (AGE < 64);
% AGE_APACHE6 = (AGE >= 75);
% AGE_APACHE_data(AGE_APACHE0) = 0;
% AGE_APACHE_data(AGE_APACHE2) = 2;
% AGE_APACHE_data(AGE_APACHE3) = 3;
% AGE_APACHE_data(AGE_APACHE5) = 5;
% AGE_APACHE_data(AGE_APACHE6) = 6;

%APACHE = [temp_APACHE_data MAP_APACHE_data HR_APACHE_data resp_APACHE_data pH_APACHE_data potass_APACHE_data Creatinine_APACHE_data AGE_APACHE_data];
%APACHE1 = sum([temp_APACHE_data MAP_APACHE_data HR_APACHE_data resp_APACHE_data pH_APACHE_data potass_APACHE_data Creatinine_APACHE_data AGE_APACHE_data]);

% 1 
FiO2_APACHE_data = zeros(size(raw_data,1),1);
FiO2_APACHE1 = (raw_data(:,11) >= 0.5); 
FiO2_APACHE2 = (raw_data(:,11) < 0.5);
FiO2_APACHE_data(FiO2_APACHE1) = 5;
FiO2_APACHE_data(FiO2_APACHE2) = 10;

%======================================================================%
% Considering Total 138 Categorical Features 
F = [cat_data dummy_feat age_cat raw_data(:,36) Ulimit_data  Llimit_data iculim MEWS NEWS MEWS1 NEWS1 FiO2_APACHE_data];

end