from torch.utils.data import Dataset
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os

FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
            'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
            'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
            'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium',
            'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI',
            'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
            'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
LABEL = ['SepsisLabel']

# Labs and Vitals that needed indicators
LABS_VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']


class PhysionetDataset(Dataset):

    """
    Example usage:
    datadir = "Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/"
    dataset = PhysionetDataset(datadir)
    dataset.__preprocess__()
    # Use with PyTorch
    dataloader = DataLoader(dataset, batch_size=5)
    for i, batch in enumerate(dataloader):
        row_data = batch[0]
        label = batch[1]
        # Run model on batch!!
    # Use entire dataframe
    data = dataset.data
    Attributes:
        data (TYPE): pandas.DataFrame
    """

    def __init__(self, input_directory):

        filenames = os.listdir(input_directory)

        # Read all filenames
        all_patients_np_list = []
        for filename in filenames:
            patient_id = int(filename.split(".")[0][1:])
            patient_df = pd.read_csv(os.path.join(input_directory, filename),
                                     sep="|")
            patient_df["id"] = patient_id  # Add patient ID to data
            patient_df["filename"] = filename

            all_patients_np_list.append(patient_df.values)

        combined_np_array = np.vstack(all_patients_np_list)
        self.data = pd.DataFrame(combined_np_array)
        self.data.columns = patient_df.columns
        self.indices_outcome = self.data[self.data[LABEL[0]] == 1].index.values.astype(int)
        self.indices_no_outcome = self.data[self.data[LABEL[0]] == 0].index.values.astype(int)

    def __len__(self):
        return len(self.data)

    def __preprocess__(self, method="measured"):

        self.preprocessing_method = method

        # Forward fill
        self.data = self.data.groupby("id").ffill()

        if method == "measured":
            """Measured preprocessing
            - Forward fill
            - Add indicator '_measured' variable
            - Fill with patient-specific mean
            - Fill with -1
            - Normalize labs/vitals columns
            """

            # Add indicator variables & fill with means for labs/vitals
            for feature in LABS_VITALS:
                # Add indicator variable for each labs/vitals "xxx" with name "xxx_measured" and fill with 1 (measured) or 0 (not measured)
                self.data[feature + "_measured"] = [int(not(val)) for val in self.data[feature].isna().tolist()]
                # Fill NaNs in labs/vitals into averages for each patient
                self.data[feature] = self.data.groupby("id")[feature].apply(lambda x: x.fillna(x.mean()))

            # Fill the rest NaNs with -1
            self.data = self.data.fillna(-1)

            # Normalization for certain columns
            #selected_normalize = self.data.drop(["id", "Unit1", "Unit2", 'SepsisLabel'], axis=1)
            #x = selected_normalize.values
            #min_max_scaler = preprocessing.MinMaxScaler()
            #x_scaled = min_max_scaler.fit_transform(x)
            #self.data[selected_normalize.columns.tolist()] = x_scaled

        elif method == "simple":
            """Simple preprocessing:
            - Forward fill
            - Fill with -1s
            """
            self.data = self.data.fillna(-1)

        updated_features = self.data.columns.tolist()
        if "id" in updated_features:
            updated_features.remove("id")
        if "SepsisLabel" in updated_features:
            updated_features.remove("SepsisLabel")
        if "filename" in updated_features:
            updated_features.remove("filename")
        self.features = updated_features


    # Returns 1 row of data
    def __getitem__(self, index):
        patient_id = self.data.iloc[index]["id"]
        iculos = self.data.iloc[index]["ICULOS"]
        return (self.data.iloc[index][FEATURES].values.astype(float),
                self.data.iloc[index][LABEL].values.astype(int),
                patient_id,
                iculos)


# Inherit from PhysionetDataset --> use same preprocessing
class PhysionetDatasetCNN(PhysionetDataset):

    """
    Example usage:
    datadir = "Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/"
    dataset = PhysionetDatasetCNN(datadir)
    dataset.__preprocess__()
    dataset.__setwindow__(window = 8) # Generates 8 hour windows!
    # Use with PyTorch
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i, batch in enumerate(dataloader):
        cnn_data = batch[0]
        label = batch[1]
        # blah blah blah cnn
    Attributes:
        window (TYPE): Description
    """

    def __setwindow__(self, window):
        self.window = window

    # Override the __getitem__ function to return data for CNN instead
    # of one row
    def __getitem__(self, index):

        patient_id = self.data.iloc[index]["id"].astype(int)
        filename = self.data.iloc[index]["filename"]
        iculos = self.data.iloc[index]["ICULOS"]

        if index < self.window:
            window_data = self.data.iloc[:index + 1]
        else:
            window_data = self.data.iloc[index + 1 - self.window: index + 1]

        outcome = window_data[LABEL].values[-1].astype(int)

        if (window_data["id"].nunique() == 1 and
                len(window_data) == self.window):
            data = window_data[self.features].values
        else:
            data = np.zeros((self.window, len(self.features)))
            clipped_window = window_data[window_data["id"] == patient_id]
            data[-len(clipped_window):, :] = clipped_window[self.features].values

        # data has shape (self.window, len(FEATURES))
        return (data, outcome, patient_id, iculos, filename)

class PhysionetDatasetCNNInfer(PhysionetDatasetCNN):

    def __init__(self, np_array):
        self.data = pd.DataFrame(np_array)
        self.data.columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
        self.data["id"] = 1
        self.data["filename"] = "1"

    def __preprocess__(self, method="measured"):

        self.preprocessing_method = method

        # Forward fill
        self.data = self.data.ffill()

        if method == "measured":
            """Measured preprocessing
            - Forward fill
            - Add indicator '_measured' variable
            - Fill with patient-specific mean
            - Fill with -1
            - Normalize labs/vitals columns
            """

            # Add indicator variables & fill with means for labs/vitals
            for feature in LABS_VITALS:
                # Add indicator variable for each labs/vitals "xxx" with name "xxx_measured" and fill with 1 (measured) or 0 (not measured)
                self.data[feature + "_measured"] = [int(not(val)) for val in self.data[feature].isna().tolist()]
                # Fill NaNs in labs/vitals into averages for each patient
                # self.data[feature] = self.data.groupby("id")[feature].apply(lambda x: x.fillna(x.mean()))

            # Fill the rest NaNs with -1
            self.data = self.data.fillna(-1)

            # Normalization for certain columns
            #selected_normalize = self.data.drop(["id", "Unit1", "Unit2", 'SepsisLabel'], axis=1)
            #x = selected_normalize.values
            #min_max_scaler = preprocessing.MinMaxScaler()
            #x_scaled = min_max_scaler.fit_transform(x)
            #self.data[selected_normalize.columns.tolist()] = x_scaled

        elif method == "simple":
            """Simple preprocessing:
            - Forward fill
            - Fill with -1s
            """
            self.data = self.data.fillna(-1)

        updated_features = self.data.columns.tolist()
        if "id" in updated_features:
            updated_features.remove("id")
        if "SepsisLabel" in updated_features:
            updated_features.remove("SepsisLabel")
        if "filename" in updated_features:
            updated_features.remove("filename")
        self.features = updated_features

    # Override the __getitem__ function to return data for CNN instead
    # of one row
    def __getitem__(self, index):

        patient_id = self.data.iloc[index]["id"].astype(int)
        filename = self.data.iloc[index]["filename"]
        iculos = self.data.iloc[index]["ICULOS"]

        if index < self.window:
            window_data = self.data.iloc[:index + 1]
        else:
            window_data = self.data.iloc[index + 1 - self.window: index + 1]

        # outcome = window_data[LABEL].values[-1].astype(int)

        if (window_data["id"].nunique() == 1 and
                len(window_data) == self.window):
            data = window_data[self.features].values
        else:
            data = np.zeros((self.window, len(self.features)))
            clipped_window = window_data[window_data["id"] == patient_id]
            data[-len(clipped_window):, :] = clipped_window[self.features].values

        # data has shape (self.window, len(FEATURES))
        return (data, patient_id, iculos, filename)
