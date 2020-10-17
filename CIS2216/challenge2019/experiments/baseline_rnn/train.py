import os
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

from sampler import ImbalancedDatasetSampler
from model import MyRNN
from util import compute_prediction_utility

g_seed = 1203
np.random.seed(g_seed)
torch.manual_seed(g_seed)


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        item = self.data[index]
        return item.values.astype(np.float)

    def __len__(self):
        return len(self.data)


def my_collate_fn(batch):
    batch_x = [item[:, :-1] for item in batch]  # (B,L,D)
    batch_y = [item[:, -1] for item in batch]  # (B,L)

    # shift for early prediction
    # shift = 6
    # for item in batch_y:
    #     indices = np.nonzero(item)[0]
    #     if len(indices) != 0:
    #         first_nonzero = indices[0]
    #         item[first_nonzero - shift if first_nonzero -
    #              shift > 0 else 0:first_nonzero] = 1
    # no shift
    for item in batch_y:
        item[:] = item[-1]

    seq_lengths = list(map(len, batch_y))  # (B)
    max_seq_len = max(seq_lengths)

    # padding
    new_batch_x = []
    for item in batch_x:
        if len(item) == max_seq_len:
            new_batch_x.append(item)
        else:
            new_batch_x.append(np.concatenate(
                (item, np.zeros(((max_seq_len - len(item)), item.shape[1]))), axis=0))

    batch_x = new_batch_x
    batch_y = [list(item) + [-1] * (max_seq_len - len(item))
               for item in batch_y]

    return torch.FloatTensor(batch_x), torch.LongTensor(batch_y)


def run():
    data_path = '../data/data.pkl'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    EPOCH = 100
    EVAL_EVERY = 1
    Therhold = 0.3
    ONLY_TESTING = True
    model_name = 'rnn'
    save_path = 'saved-Epoch:{}-Therhold:{}-Model:{}/'.format(
        EPOCH, Therhold, model_name)
    resume_model_path = 'saved-Epoch:100-Therhold:0.3-Model:rnn-weight1:8-shift6-no-residual/rnn-weight1:8-shift6-no-residual_epoch:30_score:-0.1570.pkl'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # read and process data
    with open(data_path, 'rb') as fin:
        res = pickle.load(fin)
    data = res['data']
    label = np.array(res['final_label'], np.int16)

    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.2, random_state=g_seed, stratify=label)
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=my_collate_fn,
                                  sampler=ImbalancedDatasetSampler(train_dataset))
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 collate_fn=my_collate_fn,
                                 sampler=SequentialSampler(test_dataset))

    # build model
    # input_dim = len(dateframe cols) - 1 which is (label col)
    model = MyRNN(input_dim=X_train[0].shape[1] - 1)
    model.to(device)

    # only testing
    if ONLY_TESTING:
        model.load_state_dict(torch.load(resume_model_path))
        model.eval()
        with torch.no_grad():
            for therhold in [0.3, 0.33, 0.34, 0.37]:
                y_pred = []
                y_gt = []
                prog_iter = tqdm(
                    test_dataloader, desc="Evaluating", leave=False)
                for batch in prog_iter:
                    input_x, input_y = tuple(t.to(device) for t in batch)
                    logits = model(input_x)
                    prob = F.softmax(logits, dim=-1)[:, :, 1]
                    pred = torch.where(prob > therhold, torch.ones_like(
                        prob), torch.zeros_like(prob))
                    pred = pred.detach().cpu().numpy()

                    for gt_idx, gt in enumerate(input_y.detach().cpu().numpy()):
                        padding_len = np.sum(gt == -1)
                        y_gt.append(gt[:len(gt) - padding_len])
                        y_pred.append(
                            pred[gt_idx][:len(pred[gt_idx]) - padding_len])

                # score
                all_score = []
                for pred_idx, pred in enumerate(y_pred):
                    score, _ = compute_prediction_utility(y_gt[pred_idx], pred)
                    all_score.append(score)

                mean_score = np.mean(all_score)
                print('therhold:%.4f, testing score:%.4f' %
                      (therhold, mean_score))
        return

    # training code below
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    best_eval_score = -100
    best_eval_epoch = 0
    for epoch_idx in tqdm(range(EPOCH), desc='Epoch: '):
        model.train()
        prog_iter = tqdm(train_dataloader, desc="Training", leave=False)
        tr_loss = 0
        tr_example_cnt = 0
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch)
            logits = model(input_x)
            loss = model.get_loss(logits, input_y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tr_loss += loss.item()
            tr_example_cnt += input_x.size(0)

            prog_iter.set_postfix(loss='%.4f' % (tr_loss / tr_example_cnt))

        scheduler.step(epoch_idx)
        # debug
        print(F.softmax(logits[0, :, :], dim=-1))
        print(input_y[0])

        if epoch_idx % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                y_pred = []
                y_gt = []
                prog_iter = tqdm(
                    test_dataloader, desc="Evaluating", leave=False)
                for batch in prog_iter:
                    input_x, input_y = tuple(t.to(device) for t in batch)
                    logits = model(input_x)
                    prob = F.softmax(logits, dim=-1)[:, :, 1]
                    pred = torch.where(prob > Therhold, torch.ones_like(
                        prob), torch.zeros_like(prob))
                    pred = pred.detach().cpu().numpy()

                    for gt_idx, gt in enumerate(input_y.detach().cpu().numpy()):
                        padding_len = np.sum(gt == -1)
                        y_gt.append(gt[:len(gt) - padding_len])
                        y_pred.append(
                            pred[gt_idx][:len(pred[gt_idx]) - padding_len])

                # score
                all_score = []
                for pred_idx, pred in enumerate(y_pred):
                    score, _ = compute_prediction_utility(y_gt[pred_idx], pred)
                    all_score.append(score)

                mean_score = np.mean(all_score)
                torch.save(model.state_dict(),
                           os.path.join(save_path, model_name + '_epoch:%d_score:%.4f.pkl' % (epoch_idx, mean_score)))

                if mean_score > best_eval_score:
                    best_eval_score = mean_score
                    best_eval_epoch = epoch_idx

            print('score', mean_score)

    print('best_score:%.4f, best_epoch:%d' %
          (best_eval_score, best_eval_epoch))


if __name__ == '__main__':
    run()
