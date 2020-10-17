# -*- coding: utf-8 -*-
# @Author: Chloe
# @Date:   2019-07-22 13:06:21
# @Last Modified by:   Chloe
# @Last Modified time: 2019-07-23 14:23:05

import argparse
import torch
import os
import pandas as pd
import numpy as np
from model import CNN
from dataset import PhysionetDataset, PhysionetDatasetCNN, FEATURES, LABS_VITALS
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from evaluate_sepsis_score import compute_prediction_utility, compute_auc, compute_accuracy_f_measure

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)

def print_results(train_metrics, valid_metrics, train_loss, valid_loss,
                  header="", verbose=True):

    if verbose:
        log = "{} (train -- valid)\n".format(header)
        log += "Loss: {} -- {} \n".format(train_loss, valid_loss)
        log += "AUROC: {} -- {}\n".format(train_metrics["auroc"],
                                          valid_metrics["auroc"])
        log += "AUPRC: {} -- {}\n".format(train_metrics["auprc"],
                                          valid_metrics["auprc"])
        log += "Normalized utility : {} -- {}\n".format(train_metrics["normalized_observed_utility"],
            valid_metrics["normalized_observed_utility"])
        log += "Unnormalized utility : {} -- {}\n".format(train_metrics["unnormalized_observed_utility"],
            valid_metrics["unnormalized_observed_utility"])
        log += "Train classification report:\n"
        log += classification_report(y_true=train_results.SepsisLabel, y_pred=train_results.Prediction)
        log += "Valid classification report:\n"
        log += classification_report(y_true=valid_results.SepsisLabel, y_pred=valid_results.Prediction)
        log += "\n"
    else:
        log = "{} -- Train loss: {} -- Valid loss: {} -- ".format(header, train_loss, valid_loss)
        log += "Train AUROC: {} -- Valid AUROC: {}".format(train_metrics["auroc"], valid_metrics["auroc"])
    print(log)

def train_model(model, loss_fn, optimizer, train_dataloader, valid_dataloader,
                num_epochs=100, cuda=False):

    model.train()
    # Early Stopping
    # TODO: Need to put them into hyperparameters
    es = EarlyStopping(patience=5, min_delta=1e-10)
    for epoch in range(num_epochs):
        loss_epoch = 0.0
        for batch in train_dataloader:
            # Get data
            data = batch[0].float()
            target = batch[1]

            if cuda:
                data = data.cuda()
                target = target.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            prediction = model(data)
            loss = loss_fn(prediction, target.view(-1))
            loss.backward()
            optimizer.step()

        train_results, train_loss = evaluate_model(model, train_dataloader, loss_fn, cuda=cuda)
        valid_results, valid_loss = evaluate_model(model, valid_dataloader, loss_fn, cuda=cuda)
        train_metrics = compute_metrics(train_results)
        valid_metrics = compute_metrics(valid_results)

        # Early Stopping
        if es.step(valid_loss):
            print("Early stopping at epoch: ", epoch)
            print("Validation loss: ", valid_loss)
            break

        print_results(train_metrics, valid_metrics, train_loss, valid_loss,
            header="Epoch {}".format(epoch), verbose=False)

    return model

def evaluate_model(model, dataloader, loss_fn, threshold=0.5, cuda=False):

    model.eval()
    targets = []
    probabilities = []
    patient_ids = []
    iculos_vals = []
    total_loss = 0.0

    for batch in dataloader:
        data = batch[0].float()
        target = batch[1]

        if cuda:
            data = data.cuda()

        prediction = model(data)

        if cuda:
            prediction = prediction.cpu()
        loss = loss_fn(prediction, target.view(-1))

        patient_id = list(batch[2])
        iculos = list(batch[3].numpy())
        probability = list(torch.softmax(prediction, 1)[:, 1].detach().numpy())
        targets += list(target[:, 0].numpy())
        probabilities += probability
        patient_ids += patient_id
        iculos_vals += iculos

        # Get total loss
        total_loss += loss.item()

    results = pd.DataFrame({
        "id": patient_ids,
        "ICULOS": iculos_vals,
        "SepsisLabel": targets,
        "PredictedProbability": probabilities,
        "Prediction": [1 if prob > threshold else 0 for prob in probabilities]
    })
    return results.sort_values(["id", "ICULOS"]), total_loss / dataloader.__len__()

def compute_metrics(results):
    auroc, auprc = compute_auc(labels=results.SepsisLabel,
                               predictions=results.PredictedProbability)
    accuracy, f_measure = compute_accuracy_f_measure(labels=results.SepsisLabel,
                                                     predictions=results.Prediction)

    # Compute utility.
    num_files = results.id.nunique()
    ids = results.id.unique()
    observed_utilities = np.zeros(num_files)
    best_utilities     = np.zeros(num_files)
    worst_utilities    = np.zeros(num_files)
    inaction_utilities = np.zeros(num_files)

    dt_early   = -12
    dt_optimal = -6
    dt_late    = 3

    max_u_tp = 1
    min_u_fn = -2
    u_fp     = -0.05
    u_tn     = 0
    for k in range(num_files):
        patient_id = ids[k]
        # labels = cohort_labels[k]
        labels = results[results.id == patient_id].SepsisLabel.values
        num_rows          = len(labels)
        # observed_predictions = cohort_predictions[k]
        observed_predictions = results[results.id == patient_id].Prediction.values
        best_predictions     = np.zeros(num_rows)
        worst_predictions    = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)

        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            best_predictions[max(0, t_sepsis + dt_early) : min(t_sepsis + dt_late + 1, num_rows)] = 1
        worst_predictions = 1 - best_predictions

        observed_utilities[k] = compute_prediction_utility(labels, observed_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        best_utilities[k]     = compute_prediction_utility(labels, best_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        worst_utilities[k]    = compute_prediction_utility(labels, worst_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_utilities[k] = compute_prediction_utility(labels, inaction_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility     = np.sum(best_utilities)
    unnormalized_worst_utility    = np.sum(worst_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)

    normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "f_measure": f_measure,
        "normalized_observed_utility": normalized_observed_utility,
        "unnormalized_observed_utility": unnormalized_best_utility
    }


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_dir",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/")
    argparser.add_argument("--valid_dir",
                           default="Z:/LKS-CHART/Projects/physionet_sepsis_project/data/small_data/")
    argparser.add_argument("--cuda", action="store_true")
    argparser.add_argument("--batch_size", default=100, type=int)
    argparser.add_argument("--window_size", default=24, type=int)
    argparser.add_argument("--num_epochs", default=10, type=int)
    argparser.add_argument("--learning_rate", default=0.0001, type=float)
    argparser.add_argument("--output_dir", default=".")
    argparser.add_argument("--preprocessing_method", default="measured",
                           help="""Possible values:
                           - measured (forward-fill, add indicator variables, normalize, impute with -1),
                           - simple (only do forward-fill and impute with -1) """)
    args = argparser.parse_args()

    cuda = args.cuda and torch.cuda.is_available()
    window_size = args.window_size
    num_features = len(FEATURES) + len(LABS_VITALS)
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    print("Loading train data")
    train_dataset = PhysionetDatasetCNN(args.train_dir)
    train_dataset.__preprocess__(method=args.preprocessing_method)
    train_dataset.__setwindow__(window_size)
    print("Loading valid data")
    valid_dataset = PhysionetDatasetCNN(args.valid_dir)
    valid_dataset.__preprocess__(method=args.preprocessing_method)
    valid_dataset.__setwindow__(window_size)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size, shuffle=False)

    model = CNN(input_height=window_size, input_width=num_features)
    if cuda:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model = train_model(model, loss_fn=criterion,
                        optimizer=optimizer,
                        train_dataloader=train_dataloader,
                        valid_dataloader=valid_dataloader,
                        num_epochs=num_epochs,
                        cuda=cuda)

    # Evaluate model across diff thresholds
    for thr in np.arange(0.1, 1.0, 0.1):
        train_results, train_loss = evaluate_model(model, train_dataloader, criterion, threshold=thr, cuda=cuda)
        valid_results, valid_loss = evaluate_model(model, valid_dataloader, criterion, threshold=thr, cuda=cuda)
        train_metrics = compute_metrics(train_results)
        valid_metrics = compute_metrics(valid_results)
        print_results(train_metrics, valid_metrics, train_loss, valid_loss,
                      header="Results for threshold {}".format(thr), verbose=True)

    print("Saving model")
    torch.save(model.state_dict(), os.path.join(args.output_dir,
                                                "model.pt"))