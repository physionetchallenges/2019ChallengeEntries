from fastai.torch_core import *
from data import *
import scipy

__all__ = ["classify"]

def predict(model:nn.Module, data:Collection[Tensor]):
    preds = model(*data).squeeze_(dim=0).sigmoid_().softmax(dim=0)
    probs, classes = preds.max(dim=0)
    probs, classes = probs.detach().numpy(), classes.numpy()
    return probs, classes


def predict_ensemble(models:Collection[nn.Module], data:Collection[Tensor], maximum:int=1):
    all_c, all_p = [], []
    if maximum > 0: models = np.random.permutation(models)[:maximum]

    for model in models:
        probs, classes = predict(model, data)
        all_p += [probs]
        all_c += [classes]
    all_c, all_p = np.stack(all_c), np.stack(all_p)

    if maximum == 1:
        return all_c.squeeze(0), all_p.squeeze(0)

    # Use mode for voting the final result
    modes, _ = scipy.stats.mode(all_c, axis=0)
    modes = modes[0]
    all_p = np.asarray([np.mean(all_p[all_c[:, i]==modes[i], i]) for i in range(len(modes))])
    all_c = modes
    return all_p, all_c


def load_multiple_models(config_dir: Path, build_model: Callable, provider: DataProvider):
    states = [torch.load(m, map_location='cpu' if not torch.cuda.is_available() else None)['model'] for m in config_dir.joinpath('models').glob("*-fold-*.pth")]
    models = []
    for state in states:
        model = build_model(provider)
        model.load_state_dict(state)
        model.eval()
        models +=[model]
    return models


def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))


def classify(input_dir:str, output_dir:str, build_model:Callable, config_dir:str='./',load_model_params:Callable=load_multiple_models, predictor:Callable=predict_ensemble):
    input_dir, output_dir, config_dir = Path(input_dir), Path(output_dir), Path(config_dir)
    if not os.path.isdir(output_dir) or not output_dir.exists: os.makedirs(output_dir, exist_ok=True)

    with open(config_dir.joinpath('settings.json')) as f: settings = json.load(f)
    provider = DataProvider(config_dir.joinpath('params.json'), cache_root=None, **settings['provider']).preprocess(root=input_dir, lazy=False)
    dataset = SepsisItemList.from_folder(input_dir, extensions=".psv", processor=SepsisPreprocessor(provider, **settings['processor'])).split_none().label_empty()
    model = load_model_params(config_dir, build_model, provider)

    with torch.no_grad():
        for i, (x, y) in progress_bar(enumerate(dataset.train), total=len(dataset.train), parent=None, leave=False):
            fname, data = x.fname, x.data
            data = [item.unsqueeze(0) for item in data]
            scores, labels = predictor(model, data)
            save_challenge_predictions(output_dir.joinpath(fname.name), scores, labels)

    return scores, labels, fname


def load_classifier_context(build_model, config_dir: str = Path('./'),
                            load_model_params: Callable = load_multiple_models):
    with open(config_dir.joinpath('settings.json')) as f: settings = json.load(f)
    provider = DataProvider(config_dir.joinpath('params.json'), cache_root=None, **settings['provider'])
    model = load_model_params(config_dir, build_model, provider)
    return model, provider, settings


__HEAD = 'HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel'
__HEAD = __HEAD.split('|')


def predict_with_context(data, context, input_dir=Path('.tmp/.input/'), input_name="whatever.psv", predictor=predict_ensemble):
    # Prepare data and models
    model, provider, settings = context
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=__HEAD if data.shape[1] == len(__HEAD) else __HEAD[:-1])

    # Setup input
    if not input_dir.exists():
        os.makedirs(input_dir, exist_ok=True)
        data.to_csv(input_dir.joinpath(input_name), index=False)
    provider.update_input(input_name, data)
    dataset = SepsisItemList.from_folder(input_dir, extensions=".psv", processor=SepsisPreprocessor(provider, **settings['processor'])).split_none().label_empty()

    # Make the prediction
    with torch.no_grad():
        x, y = dataset.train[0]
        fname, data = x.fname, x.data
        data = [item.unsqueeze(0) for item in data]
        scores, labels = predictor(model, data)
        return scores[-1], labels[-1]  # only the last one
