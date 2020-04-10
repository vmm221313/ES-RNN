from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data_loading import create_datasets, SeriesDataset
from config import get_config
from trainer import ESRNNTrainer
from model import ESRNN
from DRNN import DRNN
import time

import os
import time
import numpy as np
import copy
import torch
import torch.nn as nn
from loss_modules import PinballLoss, sMAPE, np_sMAPE
from utils.logger import Logger
import pandas as pd

print('loading config')
config = get_config('Daily')

config['output_size'] = 30

print('loading data')
info = pd.read_csv('data/energy_daily.csv')

train_path = 'data/train/%s-train.csv' % (config['variable'])
test_path = 'data/test/%s-test.csv' % (config['variable'])

df = pd.read_csv('data/energy_daily.csv')

df = df['W']

df.max()

df.min()

# +
#df = (df - df.min())/(df.max() - df.min()) #model normalizes
# -

values = df.values

# data = []
# seq_len = 30
# for i in range(len(values)-seq_len+1):
#     #print(list(values[i:i+seq_len]))
#     data.append(list(values[i:i+seq_len].reshape(-1)))

# df = pd.DataFrame(data)

# df.head()

train_df = pd.DataFrame(df[:int(0.8*len(df))])
test_df = pd.DataFrame(df[int(0.8*len(df)):])

train_df.to_csv(train_path, index = False)
test_df.to_csv(test_path, index = False)

train = train_df.values.T
test = test_df.values.T

train.shape

len(train)

test.shape

val = train[:, -config['output_size']:]
train = train[:, :-config['output_size']]

train.shape

val.shape

# train, val, test = create_datasets(train_path, test_path, config['output_size'])

import torch
from torch.utils.data import Dataset


class SeriesDataset(Dataset):

    def __init__(self, dataTrain, dataVal, dataTest, variable, chop_value, device):
        #dataTrain, mask = chop_series(dataTrain, chop_value)

        #self.dataInfoCatOHE = pd.get_dummies(info[info['SP'] == variable]['category'])
        #self.dataInfoCatHeaders = np.array([i for i in self.dataInfoCatOHE.columns.values])
        #self.dataInfoCat = torch.from_numpy(self.dataInfoCatOHE[mask].values).float()
        self.dataTrain = [torch.tensor(dataTrain[i]) for i in range(len(dataTrain))]  # ALREADY MASKED IN CHOP FUNCTION
        self.dataVal = [torch.tensor(dataVal[i]) for i in range(len(dataVal))]
        self.dataTest = [torch.tensor(dataTest[i]) for i in range(len(dataTest))]
        self.device = device

    def __len__(self):
        return len(self.dataTrain)

    def __getitem__(self, idx):
        return self.dataTrain[idx].to(self.device), \
                self.dataVal[idx].to(self.device), \
                self.dataTest[idx].to(self.device), \
                idx
                #self.dataInfoCat[idx].to(self.device), \


dataset = SeriesDataset(train, val, test, config['variable'], config['chop_val'], config['device'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

for batch in dataloader:
    print(batch[0].shape)

len(dataset)

run_id = str(int(time.time())) #UNIX time
run_id

config['dilations']

for grp_num in range(len(config['dilations'])):
    print(grp_num)


class ResidualDRNN(nn.Module):
    def __init__(self, config):
        super(ResidualDRNN, self).__init__()
        self.config = config

        layers = []
        for grp_num in range(len(self.config['dilations'])):

            if grp_num == 0:
                input_size = self.config['input_size']# + self.config['num_of_categories']
            else:
                input_size = self.config['state_hsize']
            
            #input_size = self.config['input_size']
            
            l = DRNN(input_size,
                     self.config['state_hsize'],
                     n_layers=len(self.config['dilations'][grp_num]),
                     dilations=self.config['dilations'][grp_num],
                     cell_type=self.config['rnn_cell_type'])

            layers.append(l)

        self.rnn_stack = nn.Sequential(*layers)

    def forward(self, input_data):
        for layer_num in range(len(self.rnn_stack)):
            residual = input_data
            out, _ = self.rnn_stack[layer_num](input_data)
            if layer_num > 0:
                out += residual
            input_data = out
        return out


class ESRNN(nn.Module):
    def __init__(self, num_series, config):
        super(ESRNN, self).__init__()
        self.config = config
        self.num_series = num_series
        self.add_nl_layer = self.config['add_nl_layer']

        init_lev_sms = []
        init_seas_sms = []
        init_seasonalities = []

        for i in range(num_series):
            init_lev_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
            init_seas_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
            init_seasonalities.append(nn.Parameter((torch.ones(config['seasonality']) * 0.5), requires_grad=True))

        self.init_lev_sms = nn.ParameterList(init_lev_sms)
        self.init_seas_sms = nn.ParameterList(init_seas_sms)
        self.init_seasonalities = nn.ParameterList(init_seasonalities)

        self.nl_layer = nn.Linear(config['state_hsize'],
                                  config['state_hsize'])
        self.act = nn.Tanh()
        self.scoring = nn.Linear(config['state_hsize'], config['output_size'])

        self.logistic = nn.Sigmoid()

        self.resid_drnn = ResidualDRNN(self.config)

    def forward(self, train, val, test, idxs, testing=False):
        # GET THE PER SERIES PARAMETERS
        lev_sms = self.logistic(torch.stack([self.init_lev_sms[idx] for idx in idxs]).squeeze(1))
        seas_sms = self.logistic(torch.stack([self.init_seas_sms[idx] for idx in idxs]).squeeze(1))
        init_seasonalities = torch.stack([self.init_seasonalities[idx] for idx in idxs])

        seasonalities = []
        # PRIME SEASONALITY
        for i in range(self.config['seasonality']):
            seasonalities.append(torch.exp(init_seasonalities[:, i]))
        seasonalities.append(torch.exp(init_seasonalities[:, 0]))

        if testing:
            train = torch.cat((train, val), dim=1)

        train = train.float()

        levs = []
        log_diff_of_levels = []

        levs.append(train[:, 0] / seasonalities[0])
        for i in range(1, train.shape[1]):
            # CALCULATE LEVEL FOR CURRENT TIMESTEP TO NORMALIZE RNN
            new_lev = lev_sms * (train[:, i] / seasonalities[i]) + (1 - lev_sms) * levs[i - 1]
            levs.append(new_lev)

            # STORE DIFFERENCE TO PENALIZE LATER
            log_diff_of_levels.append(torch.log(new_lev / levs[i - 1]))

            # CALCULATE SEASONALITY TO DESEASONALIZE THE DATA FOR RNN
            seasonalities.append(seas_sms * (train[:, i] / new_lev) + (1 - seas_sms) * seasonalities[i])

        seasonalities_stacked = torch.stack(seasonalities).transpose(1, 0)
        levs_stacked = torch.stack(levs).transpose(1, 0)

        loss_mean_sq_log_diff_level = 0
        if self.config['level_variability_penalty'] > 0:
            sq_log_diff = torch.stack(
                [(log_diff_of_levels[i] - log_diff_of_levels[i - 1]) ** 2 for i in range(1, len(log_diff_of_levels))])
            loss_mean_sq_log_diff_level = torch.mean(sq_log_diff)

        if self.config['output_size'] > self.config['seasonality']:
            start_seasonality_ext = seasonalities_stacked.shape[1] - self.config['seasonality']
            end_seasonality_ext = start_seasonality_ext + self.config['output_size'] - self.config['seasonality']
            seasonalities_stacked = torch.cat((seasonalities_stacked, seasonalities_stacked[:, start_seasonality_ext:end_seasonality_ext]),
                                              dim=1)

        window_input_list = []
        window_output_list = []
        for i in range(self.config['input_size'] - 1, train.shape[1]):
            input_window_start = i + 1 - self.config['input_size']
            input_window_end = i + 1

            train_deseas_window_input = train[:, input_window_start:input_window_end] / seasonalities_stacked[:,
                                                                                        input_window_start:input_window_end]
            train_deseas_norm_window_input = (train_deseas_window_input / levs_stacked[:, i].unsqueeze(1))
            #train_deseas_norm_cat_window_input = torch.cat((train_deseas_norm_window_input), dim=1)
            window_input_list.append(train_deseas_norm_window_input)

            output_window_start = i + 1
            output_window_end = i + 1 + self.config['output_size']

            if i < train.shape[1] - self.config['output_size']:
                train_deseas_window_output = train[:, output_window_start:output_window_end] / \
                                             seasonalities_stacked[:, output_window_start:output_window_end]
                train_deseas_norm_window_output = (train_deseas_window_output / levs_stacked[:, i].unsqueeze(1))
                window_output_list.append(train_deseas_norm_window_output)

        window_input = torch.cat([i.unsqueeze(0) for i in window_input_list], dim=0)
        window_output = torch.cat([i.unsqueeze(0) for i in window_output_list], dim=0)

        self.train()
        network_pred = self.series_forward(window_input[:-self.config['output_size']])
        network_act = window_output

        self.eval()
        network_output_non_train = self.series_forward(window_input)

        # USE THE LAST VALUE OF THE NETWORK OUTPUT TO COMPUTE THE HOLDOUT PREDICTIONS
        hold_out_output_reseas = network_output_non_train[-1] * seasonalities_stacked[:, -self.config['output_size']:]
        hold_out_output_renorm = hold_out_output_reseas * levs_stacked[:, -1].unsqueeze(1)

        hold_out_pred = hold_out_output_renorm * torch.gt(hold_out_output_renorm, 0).float()
        hold_out_act = test if testing else val

        hold_out_act_deseas = hold_out_act.float() / seasonalities_stacked[:, -self.config['output_size']:]
        hold_out_act_deseas_norm = hold_out_act_deseas / levs_stacked[:, -1].unsqueeze(1)

        self.train()
        # RETURN JUST THE TRAINING INPUT RATHER THAN THE ENTIRE SET BECAUSE THE HOLDOUT IS BEING GENERATED WITH THE REST
        return network_pred, \
               network_act, \
               (hold_out_pred, network_output_non_train), \
               (hold_out_act, hold_out_act_deseas_norm), \
               loss_mean_sq_log_diff_level

    def series_forward(self, data):
        data = self.resid_drnn(data)
        if self.add_nl_layer:
            data = self.nl_layer(data)
            data = self.act(data)
        data = self.scoring(data)
        return data


num_series = 1

# +
init_lev_sms = []
init_seas_sms = []
init_seasonalities = []

for i in range(num_series):
    init_lev_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
    init_seas_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
    init_seasonalities.append(nn.Parameter((torch.ones(config['seasonality']) * 0.5), requires_grad=True))
# -

init_lev_sms
init_seas_sms
init_seasonalities

init_lev_sms = nn.ParameterList(init_lev_sms)
init_seas_sms = nn.ParameterList(init_seas_sms)
init_seasonalities = nn.ParameterList(init_seasonalities)

nl_layer = nn.Linear(config['state_hsize'], config['state_hsize'])
act = nn.Tanh()
scoring = nn.Linear(config['state_hsize'], config['output_size'])
logistic = nn.Sigmoid()
resid_drnn = ResidualDRNN(config)

for train, val, test, idxs in dataloader:
    print(idxs)

# GET THE PER SERIES PARAMETERS
lev_sms = logistic(torch.stack([init_lev_sms[idx] for idx in idxs]).squeeze(1))
seas_sms = logistic(torch.stack([init_seas_sms[idx] for idx in idxs]).squeeze(1))
init_seasonalities = torch.stack([init_seasonalities[idx] for idx in idxs])

lev_sms
seas_sms
init_seasonalities

seasonalities = []
# PRIME SEASONALITY
for i in range(config['seasonality']):
    seasonalities.append(torch.exp(init_seasonalities[:, i]))
seasonalities.append(torch.exp(init_seasonalities[:, 0]))
seasonalities

train = train.float()
train.shape

testing = False

if testing:
    train = torch.cat((train, val), dim=1)
train.shape

train[:, 0]

levs = []
log_diff_of_levels = []
levs.append(train[:, 0] / seasonalities[0]) 

train[:, 0]
seasonalities[0]
train[:, 0] / seasonalities[0]

levs

lev_sms # = alpha

seas_sms # = gamma

for i in range(1, train.shape[1]):
    # CALCULATE LEVEL FOR CURRENT TIMESTEP TO NORMALIZE RNN
    new_lev = lev_sms * (train[:, i] / seasonalities[i]) + (1 - lev_sms) * levs[i - 1] #level update equation
    levs.append(new_lev)

    # STORE DIFFERENCE TO PENALIZE LATER
    log_diff_of_levels.append(torch.log(new_lev / levs[i - 1]))

    # CALCULATE SEASONALITY TO DESEASONALIZE THE DATA FOR RNN
    seasonalities.append(seas_sms * (train[:, i] / new_lev) + (1 - seas_sms) * seasonalities[i]) #seas update eqn

len(seasonalities) # for each step of the input, we use the seas of 'seasonality' steps back (in our case 7) 485=7+478

seasonalities

seasonalities_stacked = torch.stack(seasonalities).transpose(1, 0) #torch.stack stacks the elements of a list to
                                                                   #form a tensor

levs_stacked = torch.stack(levs).transpose(1, 0)

loss_mean_sq_log_diff_level = 0
if config['level_variability_penalty'] > 0:
    sq_log_diff = torch.stack(
        [(log_diff_of_levels[i] - log_diff_of_levels[i - 1]) ** 2 for i in range(1, len(log_diff_of_levels))])
    loss_mean_sq_log_diff_level = torch.mean(sq_log_diff)

loss_mean_sq_log_diff_level

config['output_size']
config['seasonality']

if config['output_size'] > config['seasonality']:
    start_seasonality_ext = seasonalities_stacked.shape[1] - config['seasonality']
    end_seasonality_ext = start_seasonality_ext + config['output_size'] - config['seasonality']
    seasonalities_stacked = torch.cat((seasonalities_stacked, seasonalities_stacked[:, start_seasonality_ext:end_seasonality_ext]),
                                      dim=1)

start_seasonality_ext

end_seasonality_ext

seasonalities_stacked[:].shape #added more length to make longer predictions

window_input_list = []
window_output_list = []
for i in range(config['input_size'] - 1, train.shape[1]):
    input_window_start = i + 1 - config['input_size']
    input_window_end = i + 1

    train_deseas_window_input = train[:, input_window_start:input_window_end] / seasonalities_stacked[:,
                                                                                input_window_start:input_window_end]
    train_deseas_norm_window_input = (train_deseas_window_input / levs_stacked[:, i].unsqueeze(1))
    #train_deseas_norm_cat_window_input = torch.cat((train_deseas_norm_window_input), dim=1)
    window_input_list.append(train_deseas_norm_window_input)

    output_window_start = i + 1
    output_window_end = i + 1 + config['output_size']

    if i < train.shape[1] - config['output_size']:
        train_deseas_window_output = train[:, output_window_start:output_window_end] / \
                                     seasonalities_stacked[:, output_window_start:output_window_end]
        train_deseas_norm_window_output = (train_deseas_window_output / levs_stacked[:, i].unsqueeze(1))
        window_output_list.append(train_deseas_norm_window_output)

len(window_input_list) #rolling window works correctly

len(window_output_list) 

window_input = torch.cat([i.unsqueeze(0) for i in window_input_list], dim=0)
window_output = torch.cat([i.unsqueeze(0) for i in window_output_list], dim=0)

window_input.shape #now I get where we got 472 and 7

window_output.shape

add_nl_layer = config['add_nl_layer']


def series_forward(data): #series_forward calls resid_rnn and adds nl_layer
    data = resid_drnn(data)
    if add_nl_layer:
        data = nl_layer(data)
        data = act(data)
    data = scoring(data) #scoring converts from hidden_dim shape to output_dim shape
    return data


#self.train() since ESRNN inherits nn.Module it has train and eval funcs
network_pred = series_forward(window_input[:-config['output_size']])

window_input[:-config['output_size']].shape

network_pred.shape 

network_act = window_output #true values
network_act.shape

window_input.shape

#self.eval()
network_output_non_train = series_forward(window_input)

network_pred.shape



network_output_non_train.shape

window_output.shape

network_act.shape

# USE THE LAST VALUE OF THE NETWORK OUTPUT TO COMPUTE THE HOLDOUT PREDICTIONS
# reseasonalize and renormalize by multiplying by seas and lev
hold_out_output_reseas = network_output_non_train[-1] * seasonalities_stacked[:, -config['output_size']:] 
hold_out_output_renorm = hold_out_output_reseas * levs_stacked[:, -1].unsqueeze(1)

network_output_non_train.shape

hold_out_output_renorm

torch.gt(hold_out_output_renorm, 0)

hold_out_pred = hold_out_output_renorm * torch.gt(hold_out_output_renorm, 0).float()
hold_out_act = test if testing else val

hold_out_act

hold_out_pred

hold_out_act_deseas = hold_out_act.float() / seasonalities_stacked[:, -config['output_size']:]
hold_out_act_deseas_norm = hold_out_act_deseas / levs_stacked[:, -1].unsqueeze(1)

self.train()

        self.train()
        # RETURN JUST THE TRAINING INPUT RATHER THAN THE ENTIRE SET BECAUSE THE HOLDOUT IS BEING GENERATED WITH THE REST
        return network_pred, \
               network_act, \
               (hold_out_pred, network_output_non_train), \
               (hold_out_act, hold_out_act_deseas_norm), \
               loss_mean_sq_log_diff_level

















a = torch.randn(2,5,6)
a
a.mean(dim = 0)











class ESRNNTrainer(nn.Module):
    def __init__(self, model, dataloader, run_id, config):
        super(ESRNNTrainer, self).__init__()
        self.model = model.to(config['device'])
        self.config = config
        self.dl = dataloader
        #self.ohe_headers = ohe_headers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        # self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=config['lr_anneal_step'],
                                                         gamma=config['lr_anneal_rate'])
        self.criterion = PinballLoss(self.config['training_tau'],
                                     self.config['output_size'] * self.config['batch_size'], self.config['device'])
        self.epochs = 0
        self.max_epochs = config['num_of_train_epochs']
        self.run_id = str(run_id)
        self.prod_str = 'prod' if config['prod'] else 'dev'
        self.log = Logger("../logs/train%s%s%s" % (self.config['variable'], self.prod_str, self.run_id))
        self.csv_save_path = None

    def train_epochs(self):
        max_loss = 1e8
        start_time = time.time()
        for e in range(self.max_epochs):
            self.scheduler.step()

            epoch_loss = self.train()
            if epoch_loss < max_loss:
                self.save()
            epoch_val_loss = self.val()
            if e == 0:
                file_path = os.path.join(self.csv_save_path, 'validation_losses.csv')
                with open(file_path, 'w') as f:
                    f.write('epoch,training_loss,validation_loss\n')
            with open(file_path, 'a') as f:
                f.write(','.join([str(e), str(epoch_loss), str(epoch_val_loss)]) + '\n')
        print('Total Training Mins: %5.2f' % ((time.time()-start_time)/60))

    def train(self):
        self.model.train()
        epoch_loss = 0
        for batch_num, (train, val, test, idx) in enumerate(self.dl):
            start = time.time()
            print("Train_batch: %d" % (batch_num + 1))
            loss = self.train_batch(train, val, test, idx)
            epoch_loss += loss
            end = time.time()
            self.log.log_scalar('Iteration time', end - start, batch_num + 1 * (self.epochs + 1))
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1

        # LOG EPOCH LEVEL INFORMATION
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f' % (
            self.epochs, self.max_epochs, epoch_loss))
        info = {'loss': epoch_loss}

        self.log_values(info)
        self.log_hists()

        return epoch_loss

    def train_batch(self, train, val, test, idx):
        self.optimizer.zero_grad()
        network_pred, network_act, _, _, loss_mean_sq_log_diff_level = self.model(train, val,
                                                                                  test,
                                                                                  idx)

        loss = self.criterion(network_pred, network_act)
        loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.config['gradient_clipping'])
        self.optimizer.step()
        self.scheduler.step()
        return float(loss)

    def val(self):
        self.model.eval()
        with torch.no_grad():
            acts = []
            preds = []
            

            hold_out_loss = 0
            for batch_num, (train, val, test, idx) in enumerate(self.dl):
                _, _, (hold_out_pred, network_output_non_train), \
                (hold_out_act, hold_out_act_deseas_norm), _ = self.model(train, val, test, idx)
                hold_out_loss += self.criterion(network_output_non_train.unsqueeze(0).float(),
                                                hold_out_act_deseas_norm.unsqueeze(0).float())
                acts.extend(hold_out_act.view(-1).cpu().detach().numpy())
                preds.extend(hold_out_pred.view(-1).cpu().detach().numpy())
                
            hold_out_loss = hold_out_loss / (batch_num + 1)

            
            _hold_out_df = pd.DataFrame({'acts': acts, 'preds': preds})
            #cats = [val for val in self.ohe_headers[info_cat_overall.argmax(axis=1)] for _ in
            #        range(self.config['output_size'])]
            #_hold_out_df['category'] = cats

            overall_hold_out_df = copy.copy(_hold_out_df)
            #overall_hold_out_df['category'] = ['Overall' for _ in cats]

            overall_hold_out_df = pd.concat((_hold_out_df, overall_hold_out_df))
            grouped_results = overall_hold_out_df.groupby(['category']).apply(
                lambda x: np_sMAPE(x.preds, x.acts, x.shape[0]))

            results = grouped_results.to_dict()
            results['hold_out_loss'] = float(hold_out_loss.detach().cpu())

            self.log_values(results)

            file_path = os.path.join('..', 'grouped_results', self.run_id, self.prod_str)
            os.makedirs(file_path, exist_ok=True)

            print(results)
            grouped_path = os.path.join(file_path, 'grouped_results-{}.csv'.format(self.epochs))
            grouped_results.to_csv(grouped_path)
            self.csv_save_path = file_path

        return hold_out_loss.detach().cpu().item()

    def save(self, save_dir='..'):
        print('Loss decreased, saving model!')
        file_path = os.path.join(save_dir, 'models', self.run_id, self.prod_str)
        model_path = os.path.join(file_path, 'model-{}.pyt'.format(self.epochs))
        os.makedirs(file_path, exist_ok=True)
        torch.save({'state_dict': self.model.state_dict()}, model_path)

    def log_values(self, info):

        # SCALAR
        for tag, value in info.items():
            self.log.log_scalar(tag, value, self.epochs + 1)

    def log_hists(self):
        # HISTS
        batch_params = dict()
        for tag, value in self.model.named_parameters():
            if value.grad is not None:
                if "init" in tag:
                    name, _ = tag.split(".")
                    if name not in batch_params.keys() or "%s/grad" % name not in batch_params.keys():
                        batch_params[name] = []
                        batch_params["%s/grad" % name] = []
                    batch_params[name].append(value.data.cpu().numpy())
                    batch_params["%s/grad" % name].append(value.grad.cpu().numpy())
                else:
                    tag = tag.replace('.', '/')
                    self.log.log_histogram(tag, value.data.cpu().numpy(), self.epochs + 1)
                    self.log.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), self.epochs + 1)
            else:
                print('Not printing %s because it\'s not updating' % tag)

        for tag, v in batch_params.items():
            vals = np.concatenate(np.array(v))
            self.log.log_histogram(tag, vals, self.epochs + 1)

model = ESRNN(num_series=len(dataset), config=config)

tr = ESRNNTrainer(model, dataloader, run_id, config)

tr.train_epochs()































for train, val, test, idx in dataloader:
    print(idx)

# +
#padding problem??
# -

import torch.autograd as autograd

use_cuda = torch.cuda.is_available()

# class DRNN(nn.Module):
#
#     def __init__(self, n_input, n_hidden, n_layers, dilations, dropout=0, cell_type='GRU', batch_first=False):
#
#         super(DRNN, self).__init__()
#
#         self.dilations = dilations
#         self.cell_type = cell_type
#         self.batch_first = batch_first
#
#         layers = []
#         if self.cell_type == "GRU":
#             cell = nn.GRU
#         elif self.cell_type == "RNN":
#             cell = nn.RNN
#         elif self.cell_type == "LSTM":
#             cell = nn.LSTM
#         else:
#             raise NotImplementedError
#
#         for i in range(n_layers):
#             if i == 0:
#                 c = cell(n_input, n_hidden, dropout=dropout)
#             else:
#                 c = cell(n_hidden, n_hidden, dropout=dropout)
#             layers.append(c)
#         self.cells = nn.Sequential(*layers)
#
#     def forward(self, inputs, hidden=None):
#         if self.batch_first:
#             inputs = inputs.transpose(0, 1)
#         outputs = []
#         for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)): ###### self.dilations
#             if hidden is None:
#                 inputs, _ = self.drnn_layer(cell, inputs, dilation)
#             else:
#                 inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])
#
#             outputs.append(inputs[-dilation:])
#
#         if self.batch_first:
#             inputs = inputs.transpose(0, 1)
#         return inputs, outputs
#
#     def drnn_layer(self, cell, inputs, rate, hidden=None):
#
#         n_steps = len(inputs)
#         batch_size = inputs[0].size(0)
#         hidden_size = cell.hidden_size
#         
#         print("##")
#         print(len(inputs))
#         print(inputs.shape)
#         print('##')
#         inputs, dilated_steps = self._pad_inputs(inputs, n_steps, rate) ## I think this is where the problem is 
#         print("###")
#         print(inputs.shape)
#         print('###')
#         dilated_inputs = self._prepare_inputs(inputs, rate) ##
#         print(dilated_inputs.shape)
#         print(torch.eq(inputs, dilated_inputs))
#
#         if hidden is None:
#             dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
#         else:
#             hidden = self._prepare_inputs(hidden, rate)
#             dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size,
#                                                        hidden=hidden)
#
#         splitted_outputs = self._split_outputs(dilated_outputs, rate)
#         outputs = self._unpad_outputs(splitted_outputs, n_steps)
#
#         return outputs, hidden
#
#     def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
#         if hidden is None:
#             if self.cell_type == 'LSTM':
#                 c, m = self.init_hidden(batch_size * rate, hidden_size)
#                 hidden = (c.unsqueeze(0), m.unsqueeze(0))
#             else:
#                 hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)
#         
#         print(cell)
#         #print()
#         print(hidden[0].shape)
#         print(hidden[1].shape)
#         print(dilated_inputs.shape)
#         dilated_outputs, hidden = cell(dilated_inputs, hidden)
#
#         return dilated_outputs, hidden
#
#     def _unpad_outputs(self, splitted_outputs, n_steps):
#         return splitted_outputs[:n_steps]
#
#     def _split_outputs(self, dilated_outputs, rate):
#         batchsize = dilated_outputs.size(1) // rate
#
#         blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]
#
#         interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
#         interleaved = interleaved.view(dilated_outputs.size(0) * rate,
#                                        batchsize,
#                                        dilated_outputs.size(2))
#         return interleaved
#
#     def _pad_inputs(self, inputs, n_steps, rate):
#         iseven = (n_steps % rate) == 0 ###########
#
#         if not iseven:
#             dilated_steps = n_steps // rate + 1
#
#             zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
#                                  inputs.size(1),
#                                  inputs.size(2))
#             if use_cuda:
#                 zeros_ = zeros_.cuda()
#
#             inputs = torch.cat((inputs, autograd.Variable(zeros_)))
#         else:
#             dilated_steps = n_steps // rate
#
#         return inputs, dilated_steps
#
#     def _prepare_inputs(self, inputs, rate):
#         dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
#         return dilated_inputs
#
#     def init_hidden(self, batch_size, hidden_dim):
#         hidden = autograd.Variable(torch.zeros(batch_size, hidden_dim))
#         if use_cuda:
#             hidden = hidden.cuda()
#         if self.cell_type == "LSTM":
#             memory = autograd.Variable(torch.zeros(batch_size, hidden_dim))
#             if use_cuda:
#                 memory = memory.cuda()
#             return hidden, memory
#         else:
#             return hidden

# num_series = 1
#
# init_lev_sms = []
# init_seas_sms = []
# init_seasonalities = []
#
# for i in range(num_series):
#     init_lev_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
#     init_seas_sms.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
#     init_seasonalities.append(nn.Parameter((torch.ones(config['seasonality']) * 0.5), requires_grad=True))
#
# init_seasonalities
#
# nn.ParameterList(init_lev_sms)
#
# init_lev_sms = nn.ParameterList(init_lev_sms)
# init_seas_sms = nn.ParameterList(init_seas_sms)
# init_seasonalities = nn.ParameterList(init_seasonalities)
#
# init_seasonalities
#
# idxs = [torch.tensor(0)]
#
# [init_lev_sms[idx] for idx in idxs]
#
# nn.Sigmoid()(torch.stack([init_lev_sms[idx] for idx in idxs]).squeeze(1))
#
# logistic = nn.Sigmoid()
#
# lev_sms = logistic(torch.stack([init_lev_sms[idx] for idx in idxs]).squeeze(1))
# seas_sms = logistic(torch.stack([init_seas_sms[idx] for idx in idxs]).squeeze(1))
# init_seasonalities = torch.stack([init_seasonalities[idx] for idx in idxs])
#
# seasonalities = []
# # PRIME SEASONALITY
# for i in range(config['seasonality']):
#     seasonalities.append(torch.exp(init_seasonalities[:, i]))
# seasonalities.append(torch.exp(init_seasonalities[:, 0]))
#
# seasonalities
#
# levs = []
# log_diff_of_levels = []
#
# levs.append(train[:, 0] / seasonalities[0])
# for i in range(1, train.shape[1]):
#     # CALCULATE LEVEL FOR CURRENT TIMESTEP TO NORMALIZE RNN
#     new_lev = lev_sms * (train[:, i] / seasonalities[i]) + (1 - lev_sms) * levs[i - 1]
#     levs.append(new_lev)
#
#     # STORE DIFFERENCE TO PENALIZE LATER
#     log_diff_of_levels.append(torch.log(new_lev / levs[i - 1]))
#
#     # CALCULATE SEASONALITY TO DESEASONALIZE THE DATA FOR RNN
#     seasonalities.append(seas_sms * (train[:, i] / new_lev) + (1 - seas_sms) * seasonalities[i])
#
# seasonalities_stacked = torch.stack(seasonalities).transpose(1, 0)
# levs_stacked = torch.stack(levs).transpose(1, 0)
#
# loss_mean_sq_log_diff_level = 0
# if config['level_variability_penalty'] > 0:
#     sq_log_diff = torch.stack(
#         [(log_diff_of_levels[i] - log_diff_of_levels[i - 1]) ** 2 for i in range(1, len(log_diff_of_levels))])
#     loss_mean_sq_log_diff_level = torch.mean(sq_log_diff)
#
# if config['output_size'] > config['seasonality']:
#     start_seasonality_ext = seasonalities_stacked.shape[1] - config['seasonality']
#     end_seasonality_ext = start_seasonality_ext + config['output_size'] - config['seasonality']
#     seasonalities_stacked = torch.cat((seasonalities_stacked, seasonalities_stacked[:, start_seasonality_ext:end_seasonality_ext]),
#                                       dim=1)
#
#
#










