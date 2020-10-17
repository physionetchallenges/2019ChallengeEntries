import sepsis

def get_sepsis_score(data, model):
    data = data[-8:]
    return sepsis.get_sepsis_score(data, model)

def load_sepsis_model():
    return sepsis.load_sepsis_model()