from classifier import load_classifier_context, predict_with_context
from full_seq import build_model


def get_sepsis_score(data, model): 
    return predict_with_context(data, model)


def load_sepsis_model(): 
    return load_classifier_context(build_model)
