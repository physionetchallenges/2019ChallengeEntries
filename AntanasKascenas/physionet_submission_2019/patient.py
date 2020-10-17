# Copyright (C) 2019 Canon Medical Systems Corporation. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the Physionet challenge 2019 submission project.

import numpy as np
import random
from copy import deepcopy


class Patient:

    def __init__(self, patient_data, header, means, stds,
                 representation: int = 2,
                 remove_timesteps: int = 0,
                 dropout_measurements: float = 0.):

        self._patient_data = np.copy(patient_data)
        self.patient_data = patient_data
        self.means = means
        self.stds = stds
        self.header = header
        self.caching = True if remove_timesteps == 0 else False
        self.cached_representation = None
        self.representation_idx = representation
        self.representation = [self.represent1, self.represent2, self.represent3][representation]
        self.remove_timesteps = remove_timesteps
        self.dropout_measurements = dropout_measurements


        self.age = self._patient_data[0][34]
        self.gender = self._patient_data[0][35]
        self.units_exits = 0 if np.isnan(self._patient_data[0][36]) and np.isnan(self._patient_data[0][37]) else 1
        self.unit1 = self._patient_data[0][36] if not np.isnan(self._patient_data[0][36]) else 0
        self.unit2 = self._patient_data[0][37] if not np.isnan(self._patient_data[0][37]) else 0
        self.hospital_admission = self._patient_data[0][38] if not np.isnan(self._patient_data[0][38]) else 1

    # Patient data contains a list of hourly data

    @staticmethod
    def from_file(fn, means, stds, representation=2, remove_timesteps=0, dropout_measurements=0.):
        with open(fn) as f:
            header = f.readline().strip().split("|")
            d = np.loadtxt(f, delimiter="|")

        return Patient(d, header, means, stds,
                       representation=representation,
                       remove_timesteps=remove_timesteps,
                       dropout_measurements=dropout_measurements)

    def up_to(self, n: int):
        return Patient(self._patient_data[:n], self.header, self.means, self.stds,
                       representation=self.representation_idx,
                       remove_timesteps=self.remove_timesteps,
                       dropout_measurements=self.dropout_measurements)

    def get_measurements(self, var_no):
        #vt = [(hr_data[var_no], i) for i, hr_data in enumerate(self.patient_data) if not np.isnan(hr_data[var_no])]
        vt = self.patient_data[:, var_no]
        notnans = ~np.isnan(vt)
        values = vt[notnans].tolist()
        times = np.argwhere(notnans)[:,0].tolist()
        #values, times = zip(*vt) if vt else ([], [])
        return values, times

    def get_measurements_old(self, var_no):
        vt = [[hr_data[var_no], i] for i, hr_data in enumerate(self.patient_data) if not np.isnan(hr_data[var_no])]
        values, times = zip(*vt) if vt else ([], [])
        return values, times

    def get_static_data(self):
        return self._patient_data[0][34:38]

    def sepsis_time(self):
        if self.is_sepsis_patient():
            return np.argmax([hr[40] for hr in self._patient_data]) + 6
        else:
            return -1

    def is_sepsis_patient(self):
        if len(self.patient_data[0]) >= 41:
            return np.max([hr[40] for hr in self._patient_data]) > 0.5
        else:   # Case when no labels are available.
            return False

    def get_length(self):
        return len(self.patient_data)

    @staticmethod
    def utility_TP(n, t_sepsis):
        # Utility function parameters
        early_TP1 = -0.05
        early_TP2 = 1 / 6
        early_TP3 = -1 / 9
        early_TP3b = 1.
        late_TP = 0.

        res = list(range(n))
        for t in range(len(res)):
            if t < t_sepsis - 12:
                res[t] = early_TP1
            elif t <= t_sepsis - 6:
                res[t] = (t - (t_sepsis - 12)) * early_TP2
            elif t < t_sepsis + 3:
                res[t] = early_TP3b + (t - (t_sepsis - 6)) * early_TP3
            else:
                res[t] = late_TP
        return res

    @staticmethod
    def utility_FN(n, t_sepsis):

        early_FN = -2 / 9
        late_FN = 0.

        res = list(range(n))
        for t in range(len(res)):
            if t < t_sepsis - 6:
                res[t] = 0.
            elif t <= t_sepsis + 3:
                res[t] = (t - (t_sepsis - 6)) * early_FN
            else:
                res[t] = late_FN
        return res

    @staticmethod
    def utility_FP(n):
        return [-0.05] * n

    @staticmethod
    def utility_TN(n):
        return [0.] * n

    def get_age(self):
        return self._patient_data[0][34]

    def get_gender(self):
        return self._patient_data[0][35]

    def get_units_exist(self):
        #return 0 if self._patient_data[0][36] == "NaN" and self._patient_data[0][37] == "NaN" else 1
        return 0 if np.isnan(self._patient_data[0][36]) and np.isnan(self._patient_data[0][37]) else 1

    def get_unit1(self):
        unit1 = self._patient_data[0][36]
        #return unit1 if unit1 != "NaN" else 0
        return unit1 if not np.isnan(unit1) else 0

    def get_unit2(self):
        unit2 = self._patient_data[0][37]
        #return unit2 if unit2 != "NaN" else 0
        return unit2 if not np.isnan(unit2) else 0

    def get_hospital_admission(self):
        adm = self._patient_data[0][38]
        #return adm if adm != "NaN" else 1
        return adm if not np.isnan(adm) else 1

    def represent(self, cached=None):
        if self.remove_timesteps > 0:
            times = random.sample(range(len(self._patient_data)), self.remove_timesteps)
            #self.patient_data = [deepcopy(ln) for i, ln in enumerate(self._patient_data) if i not in times]
            self.patient_data = [np.copy(ln) for i, ln in enumerate(self._patient_data) if i not in times]

        if self.dropout_measurements > 0.0001:
            for ln in self.patient_data:
                for var in range(34):
                    if random.random() < self.dropout_measurements:
                        #ln[var] = "NaN"
                        ln[var] = np.nan

        if self.caching and self.cached_representation is not None:
            return self.cached_representation
        else:
            result = self.representation(cached)

        if self.caching:
            self.cached_representation = result

        return result

    def represent2(self, cached=None):
        """For each measurement (except demographics, hospital unit and admission time add the following features:
            1. whether the measurement is missing.
            2. measurement value  (0 if not available)
            3. 1 if an estimate can be made else 0.
            4. estimated value (if measurement is missing, 0 if can't be estimated)
            5. time since last measurement, -3 if no previous measurement.
            6. measurements done so far
            7. last measurement value (0) if unavailable.
            8. age
            9. gender
            10. whether there is hospital unit data
            11. whether hospital unit 1
            12. whether hospital unit 2
            13. hospital admission time.
            14. whether ICULOS >= 60
            15. ICULOS

            Output thea features concatenated in a tensor so that group convolutions can be used.
            The nontemporal data is included in the features so that the embedding layers can take it into account.

        """
        if cached is not None:
            start_time = cached[0].shape[-1]
        else:
            start_time = 0

        def normalise(value, m_idx):
            return (value - self.means[m_idx]) / self.stds[m_idx]

        all_features = []
        for variable in range(34):
            values, times = self.get_measurements(variable)
            var_features = np.zeros([16, self.get_length()])
            for t in range(start_time, self.get_length()):
                # Feature 1
                var_features[0, t] = 1. if t not in times else 0.

                # Feature 2
                var_features[1, t] = normalise(values[times.index(t)], variable) if t in times else 0.

                # Feature 3
                var_features[2, t] = 1. if len([x for x in times if x < t]) >= 2 else 0.

                # Feature 4
                if var_features[2, t] > 0.5:
                    pp_time, p_time = [x for x in times if x < t][-2:]
                    pp_value, p_value = values[times.index(pp_time)], values[times.index(p_time)]
                    slope = (p_value - pp_value) / (p_time - pp_time)
                    intercept = pp_value - pp_time * slope
                    var_features[3, t] = normalise(intercept + t * slope, variable)
                else:
                    var_features[3, t] = 0.

                # Feature 5
                p_times = [x for x in times if x < t]
                if len(p_times) >= 1:
                    var_features[4, t] = -normalise(t - p_times[-1], 39)
                else:
                    var_features[4, t] = -3. # Just an extreme value since there is no good way to encode this. Would be better to have a separate feature maybe?

                # Feature 6
                var_features[5, t] = normalise(len(p_times) + (1 if t in times else 0), 39)

                # Feature 7
                prev_vals = [v for v, t_ in zip(values, times) if t_ < t]
                var_features[6, t] = normalise(prev_vals[-1], variable) if prev_vals else 0

                # Feature 8
                var_features[7, t] = normalise(self.age, 34)

                # Feature 9
                var_features[8, t] = -self.gender

                # Feature 10
                var_features[9, t] = self.units_exits

                # Feature 11
                var_features[10, t] = self.unit1

                # Feature 12
                var_features[11, t] = -self.unit2

                # Feature 13
                var_features[12, t] = normalise(np.log(21 - min(20, self.hospital_admission)), 38)

                # Feature 14
                var_features[13, t] = 1 if t >= 60 else 0

                # Feature 15
                var_features[14, t] = t#normalise(t, 39)

                if variable == 19:
                    # SOFA score for Kidneys
                    prev_vals = [v for v, t_ in zip(values, times) if t_ <= t]
                    if prev_vals:
                        if prev_vals[-1] >= 5.0:
                            var_features[15, t] = 4
                        elif prev_vals[-1] >= 3.5:
                            var_features[15, t] = 3
                        elif prev_vals[-1] >= 2:
                            var_features[15, t] = 2
                        elif prev_vals[-1] >= 1.2:
                            var_features[15, t] = 1

                if variable == 33:
                    # SOFA score for Coagulation
                    prev_vals = [v for v, t_ in zip(values, times) if t_ <= t]
                    if prev_vals:
                        if prev_vals[-1] < 20:
                            var_features[15, t] = 4
                        elif prev_vals[-1] < 50:
                            var_features[15, t] = 3
                        elif prev_vals[-1] < 100:
                            var_features[15, t] = 2
                        elif prev_vals[-1] < 150:
                            var_features[15, t] = 1

                if variable == 26:
                    # SOFA score for Liver
                    prev_vals = [v for v, t_ in zip(values, times) if t_ <= t]
                    if prev_vals:
                        if prev_vals[-1] >= 12:
                            var_features[15, t] = 4
                        elif prev_vals[-1] >= 6:
                            var_features[15, t] = 3
                        elif prev_vals[-1] >= 2:
                            var_features[15, t] = 2
                        elif prev_vals[-1] >= 1.2:
                            var_features[15, t] = 1

                if variable == 4:
                    # SOFA score for Cardiovascular system
                    prev_vals = [v for v, t_ in zip(values, times) if t_ <= t]
                    if prev_vals:
                        var_features[15, t] = 1 if prev_vals[-1] < 70 else 0

                if variable == 10:
                    # Calculate SOFA score for Respiratory system.

                    available = True
                    # Need to have all components to approximate paO2/FiO2 in mmHg.
                    pb = 747 # Pressure in Cleveland (as an approximation, location of hospitals is unknown)
                    prev_vals = [v for v, t_ in zip(values, times) if t_ <= t]
                    if not prev_vals:
                        available = False
                    else:
                        FiO2 = prev_vals[-1]

                    R = 0.8
                    values12, times12 = self.get_measurements(12)
                    prev_vals = [v for v, t_ in zip(values12, times12) if t_ <= t]
                    if not prev_vals:
                        available = False
                    else:
                        PaCO2 = prev_vals[-1]

                    if available:
                        # Formula taken from http://www.ucdenver.edu/academics/colleges/medicalschool/departments/medicine/intmed/imrp/CURRICULUM/Documents/Oxygenation%20and%20oxygen%20therapy.pdf
                        PaO2 = FiO2 * (pb - 47) - PaCO2 * (FiO2 + (1 - FiO2) / R)
                        x = PaO2 / FiO2 if abs(FiO2) > 0.000001 else PaO2 / 0.2 
                        if x < 400:
                            var_features[15, t] = 1
                        if x < 300:
                            var_features[15, t] = 2
                        if x < 200:
                            var_features[15, t] = 3
                        if x < 100:
                            var_features[15, t] = 4

            all_features.append(var_features)

        pos_utility = self.utility_TP(self.get_length(),
                                      self.sepsis_time()) if self.is_sepsis_patient() else self.utility_FP(
            self.get_length())
        neg_utility = self.utility_FN(self.get_length(),
                                      self.sepsis_time()) if self.is_sepsis_patient() else self.utility_TN(
            self.get_length())

        features = np.concatenate(all_features, 0)

        if cached is not None:
            cached_features, _, _ = cached
            features[..., :start_time] = cached_features



        result = features, np.array(pos_utility), np.array(neg_utility)

        return result

    def represent1(self, cached):
        """For each measurement at each time step add the following features:
            1. whether the measurement is missing.
            2. measurement value  (0 if not available)
            3. 1 if an estimate can be made else 0.
            4. estimated value (if measurement is missing, 0 if can't be estimated)
            5. time since last measurement, -1 if no previous measurement.
            6. measurements done so far

            Measurement values are normalised.

            Return a 40-long list of tensors where each tensor is 6xhours and positive/negative utilities
        """

        if self.cached_representation is not None:
            return self.cached_representation

        # Do not normalise these variables.
        no_normalisation = [35, 36, 37, 38, 39]

        def normalise(value, m_idx):
            if m_idx in no_normalisation:
                return value
            else:
                return (value - self.means[m_idx]) / self.stds[m_idx]

        all_features = []
        for variable in range(40):
            values, times = self.get_measurements(variable)
            var_features = np.zeros([6, self.get_length()])
            for t in range(self.get_length()):
                # Feature 1
                var_features[0, t] = 1. if t not in times else 0.

                # Feature 2
                var_features[1, t] = normalise(values[times.index(t)], variable) if t in times else 0.

                # Feature 3
                var_features[2, t] = 1. if len([x for x in times if x < t]) >= 2 else 0.

                # Feature 4
                if var_features[2, t] > 0.5:
                    pp_time, p_time = [x for x in times if x < t][-2:]
                    pp_value, p_value = values[times.index(pp_time)], values[times.index(p_time)]
                    slope = (p_value - pp_value) / (p_time - pp_time)
                    intercept = pp_value - pp_time * slope
                    var_features[3, t] = normalise(intercept + t * slope, variable)
                else:
                    var_features[3, t] = 0.

                # Feature 5
                p_times = [x for x in times if x < t]
                if len(p_times) >= 1:
                    var_features[4, t] = t - p_times[-1]
                else:
                    var_features[4, t] = -1.

                # Feature 6
                var_features[5, t] = len(p_times) + (1 if t in times else 0)

            all_features.append(var_features)

        pos_utility = self.utility_TP(self.get_length(), self.sepsis_time()) if self.is_sepsis_patient() else self.utility_FP(self.get_length())
        neg_utility = self.utility_FN(self.get_length(), self.sepsis_time()) if self.is_sepsis_patient() else self.utility_TN(self.get_length())

        result = all_features, np.array(pos_utility), np.array(neg_utility)
        return result

    def represent3(self, cached=None):

        features, pus, nus = self.represent2(cached[:3] if cached is not None else None)

        if not self.is_sepsis_patient():
            classes = np.array([-13] * self.get_length())
        else:
            t_s = self.sepsis_time()
            classes = np.array([min(max(x - t_s, -13), 4) for x in range(self.get_length())])

        one_hot_classes = np.stack([classes == x for x in range(-13, 5)], axis=0)

        return features, pus, nus, classes, one_hot_classes
