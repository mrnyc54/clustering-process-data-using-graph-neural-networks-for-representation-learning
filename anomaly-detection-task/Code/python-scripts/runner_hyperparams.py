from TopKPoolingExample import Net as TopKNet
from GCNet import Net as GCNet
from GINConvNet import Net as GINConvNet
from GraphDataset import GraphDataset
from gen_hyperparams import gen_df
import pandas as pd
import torch
from sqlalchemy import create_engine
from torch_geometric.loader import DataLoader
import numpy as np
from datetime import datetime
import copy
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from gen_name import gen_name
from collections import Counter

def training(model, train_loader, device, optimizer, loss_function):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        loss_all += data.num_graphs * loss.item()
    return loss_all / len(train_loader.dataset)

def training_early_stop(model, epochs, train_loader, val_loader, device, optimizer, loss, max_tries, min_diff):
    best_val_score = -1
    num_misses = max_tries
    best_model = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        loss_score = training(model, train_loader, device, optimizer, loss)
        valf1_score, valroc_score, valrec, valprec = eval(model, val_loader, device)
        if best_val_score + min_diff >= valf1_score:
            if num_misses == 0:
                print('Early stopping after {} epochs.'.format(epoch))
                break
            else:
                num_misses -= 1
        else:
            num_misses = max_tries
            best_model = copy.deepcopy(model.state_dict())
            best_val_score = valf1_score
        print('Epoch: {:03d}, Loss: {:.5f}, Val Auc: {:.5f}, Val F1: {:.5f}, Val Rec (neg Class): {:.5f}, Val Prec (neg Class): {:.5f}'.
              format(epoch, loss_score, valroc_score, valf1_score, valrec, valprec))
    return best_model, epoch

def eval(model, eval_loader, device):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in eval_loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    if len(np.unique(label)) <= 1:
        print('Warning, only one label in Dataset! Evaluation metrics work differently!')
    roc_auc = roc_auc_score(labels, predictions > 0.5) if len(np.unique(label)) > 1 else -1
    f1 = f1_score(labels, predictions > 0.5) if len(np.unique(label)) > 1 else f1_score(labels, predictions > 0.5, pos_label=0)
    rec = recall_score(labels,predictions>0.5, pos_label=0)
    prec = precision_score(labels, predictions>0.5, pos_label=0, zero_division=0)
    return f1, roc_auc, rec, prec

def get_max_activity_code(graphset):
    """Returns the highest number in second column of all nodes in graphset.
    Assumes that each activity has its own code starting from 0 and increases steadily"""
    num_embeddings = 0
    # Get highest code as approximation of number of distinct events
    for graph in graphset:
        for node in graph.x:
            if node[1].item() > num_embeddings:
                num_embeddings = int(node[1].item())
    return num_embeddings

def get_ratio_split(graphset, dataloader):
    test_a_label = dict(Counter(dataloader.dataset.data.y.tolist()))
    ratio_anomaly_gesamt = test_a_label[1] / len(graphset.data.y) if 1 in test_a_label else 0
    ratio_normal_gesamt = test_a_label[0] / len(graphset.data.y)
    anteil_test_a = len(dataloader.dataset) / len(graphset.data.y)
    return ratio_anomaly_gesamt, ratio_normal_gesamt, anteil_test_a


def split_dataset(graphset, graphset_2, batch_size, ratio_train, ratio_val, ratio_test=None, test_anom = False):
    # 1. determine split ratios
    # 2. shuffle noisy set
    # 3. determine caseid of each graph that would be in noisy test set
    # 4. recreate test set with 50/50 noisy set
    # 5. check if test set has x percent noise

    # A. Include flag to switch modes. Also possible to keep old mode (just split one dataset)
    # B. Include split ratios as parameter (to enable different ratios for synthetic and real world test set

    # Mode activated to ensure 50/50 split in test dataset
    # acceptable deviation is 1% of total cases of dataset
    #print('Debug input params: ', ratio_train, ratio_val, ratio_test)
    id_train = np.floor(len(graphset) * ratio_train).astype(int)
    id_val = id_train + np.floor(len(graphset) * ratio_val).astype(int)
    id_test = id_val + np.floor(len(graphset) * ratio_test).astype(int) if ratio_test is not None else None

    dataset, permutation = graphset.shuffle(return_perm=True)
    #print('debug list index out of range while copy:', len(dataset), id_train, len(graphset), ratio_train)
    train_dataset = dataset[:id_train].copy()
    val_dataset = dataset[id_train:id_val].copy()
    test_dataset = dataset[id_val:id_test].copy()
    graphset_2 = graphset_2.copy()
    if test_anom:
        #check if ratio is fulfilled:
        ratio_aim, ratio_is, acceptable_min_max = check_testset_ratio(dataset, test_dataset)
        print('debug no acceptable_min_max[1]: ', acceptable_min_max)
        if ratio_is < acceptable_min_max[0] or ratio_is > acceptable_min_max[1]: #too many anomalies or too few anomalies
            # check if recreating shuffle with dataset 2 will solve problem? --> if not: case level
            dataset_2 = graphset_2.index_select(permutation) # recreate the same shuffling as graphset
            test_dataset_2 = dataset_2[id_val:id_test].copy()
            ratio_aim_2, ratio_is_2, acceptable_min_max_2 = check_testset_ratio(dataset_2, test_dataset_2)
            if ratio_is_2 > acceptable_min_max_2[0] and ratio_is_2 < acceptable_min_max_2[1]: # is second dataset in limits?
                test_dataset = test_dataset_2
                print(f"backup dataset fulfills condition! ratio_is_2: {ratio_is_2}, min: {acceptable_min_max_2[0]} max: {acceptable_min_max_2[1]}")

            else: #look into caselevel manipulation
                test_dataset = manipulate_case_level(test_dataset, test_dataset_2, ratio_aim - ratio_is)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def check_testset_ratio(dataset, testset):
    lookup = dict(Counter(testset.data.y.tolist()))
    ratio_aim = len(testset.data.y.tolist()) / 2
    ratio_is = lookup[1] if 1 in lookup else 0
    acceptable_min_max = (
        ratio_aim - len(dataset.data.y.tolist()) / 100, ratio_aim + len(dataset.data.y.tolist()) / 100
    )
    return ratio_aim, ratio_is, acceptable_min_max


def manipulate_case_level(testset, testset2, diff):
    # create list of caseids --> mapping are they anomal in both or
    #   1: anomal in 1 but not in 2
    #   0: both same
    #   -1: normal in 1 but not in 2
    # determine diff
    # swap according to diff
    # return swapped

    lookup = [(graph.data.x[0], graph.data.y[0]) for graph in testset]
    lookup2 = [(graph.data.x[0], graph.data.y[0]) for graph in testset2]
    caseid_lookup = dict(lookup)
    caseid_lookup2 = dict(lookup2)
    caseid_lookup_idx = [(entry[0], c) for c, entry in enumerate(lookup)]
    print("test manipulate_case_level: \n\t", lookup,"\n\t", lookup2, "\n\t", caseid_lookup, "\n\t", caseid_lookup2, "\n\t", caseid_lookup_idx)

    diff_label_dict = {}
    for caseid, label in caseid_lookup:
        # assumption: caseid_label and caseid_label_2 are identical in amount and order of caseids
        if label > caseid_lookup2[caseid]:
            # label is anomalous, but second graph has case as normal
            diff_label_dict[caseid] = 1

        elif label == caseid_lookup2[caseid]:
            # labels identical
            diff_label_dict[caseid] = 0

        elif label < caseid_lookup2[caseid]:
            # labels is normal but second graph has case as anormal
            diff_label_dict[caseid] = -1

    for c in range(diff):
        if diff > 0:
            keys = [k for k, v in diff_label_dict.items() if v > 0]
            index_case = np.random.randint(0, len(keys))
            picked_case = diff_label_dict[index_case]
            id_testcase = caseid_lookup_idx[picked_case]
            testset[id_testcase] = testset2[id_testcase]
            diff_label_dict.pop(keys)
        elif diff < 0:
            keys = [k for k, v in diff_label_dict.items() if v < 0]
            index_case = np.random.randint(0, len(keys))
            picked_case = diff_label_dict[index_case]
            id_testcase = caseid_lookup_idx[picked_case]
            testset[id_testcase] = testset2[id_testcase]
            diff_label_dict.pop(keys)
    return testset

def log_final_metrics(model, test_loader, device, engine, name_model, name_hyper, dataset_name , actual_epoch, share_ano_train, share_train, share_ano_val, share_val, share_ano_test, share_test):
    test_score_f1, test_score_roc,_,_ = eval(model, test_loader, device)

    # assumption: Only one string value for any name, only one int vlaue for any epoch
    ls = [test_score_f1, test_score_roc, actual_epoch, str(dataset_name), str(name_hyper), str(name_model), share_ano_train, share_train, share_ano_val, share_val, share_ano_test, share_test]
    df_tmp = pd.DataFrame([ls], columns=['F1','ROC_AUC','actual_epochs','dataset','hyperparam_name','model', 'share_anomalies_train_set', 'share_train_set', 'share_anomalies_val_set', 'share_val_set','share_anomalies_test_set', 'share_test_set'])
    tablename = 'result_' + str(name_model)
    df_tmp.to_sql(tablename, engine,if_exists='append')
    return test_score_f1, test_score_roc

# pandas settings for output
pd.set_option('display.width', 240)
pd.set_option('display.max_columns', None)

# This file is supposed to load hyperparameter combinations, execute one of the nets and calculate + log the f1 score

#create connection to sqlitedb to store results
engine = create_engine('sqlite:///../../Data/Database/results.db', echo=False)
resultname = 'result'

# Set Dataset
bpic = ['bpi2013','bpi2012']

# load datamodel
graph_dir = '../../Data/Graph/'
graph_names = ['anomaly_cases_0_5.graphset','anomaly_cases_0_4.graphset','anomaly_cases_0_3.graphset','anomaly_cases_0_2.graphset','anomaly_cases_0_1.graphset','anomaly_cases_0_0.graphset',
               'anomaly_events_0_5.graphset','anomaly_events_0_4.graphset','anomaly_events_0_3.graphset','anomaly_events_0_2.graphset','anomaly_events_0_1.graphset','anomaly_events_0_0.graphset']
synth_graph_names = ['huge_0_0.graphset','large_0_0.graphset',
                     'huge_0_1.graphset','large_0_1.graphset',
                     'huge_0_2.graphset','large_0_2.graphset',
                     'huge_0_3.graphset','large_0_3.graphset',
                     'huge_0_4.graphset','large_0_4.graphset',
                     'huge_0_5.graphset','large_0_5.graphset']

graph_names_all = [bpi+'_'+graph for graph in graph_names for bpi in bpic]
graph_names_all.extend(synth_graph_names)


# generate hyperparams
length = 25 #No. of hyperparameter configurations each Dataset gets run at (*3 for each Model)
hyperparams = gen_df(length)
hyperparams.to_sql('hyperparams', engine, if_exists='append')
print('Hyperparameters of length {} created and saved!'.format(length))

# specify share per dataset split
share_train = 0.5
share_val = 0.2
share_test = 0.3

# Early stopping
patience = 5
min_diff = 0

# run models
starttime = datetime.now()
print('Run started at ', starttime)

# calculation how many trainings to be conducted
total_trainings = len(hyperparams) * len(graph_names_all) * 3
training_cntr = 1

graphset_name_log = []

for index, scenario in hyperparams.iterrows():
    epochs = scenario['epochs']
    learning_rate = scenario['alpha']
    batch_size = scenario['batch_size']
    hyperparam_name = scenario['name']

    hidden_gin = scenario['hidden_gin']
    ratio_topk = scenario['ratio_topk']

    #device = torch.device('cpu') #for debugging
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Uses device ', device)


    for graph_name in graph_names_all:

        print('loading {}{}.'.format(graph_dir, graph_name))
        dataset_name = graph_name
        graphset_name_log.append(dataset_name)
        graphset = GraphDataset(graph_dir, pd.DataFrame(), graph_name, overwrite=False)
        print(f'Graphset has {sum(graphset.data.y == 1) / len(graphset.data.y):.2%} anomalies')
        weight = torch.div(torch.sum(graphset.data.y == 0),
                           torch.sum(graphset.data.y == 1))  # get ratio of true to false cases

        # dataset specific configurations
        if 'event' in graph_name:
            print(f"{graph_name} contains event!. Selecting original mode and disregarding any ratio of anomalous cases in test set")
            train_loader, val_loader, test_loader = split_dataset(graphset, graphset, batch_size, share_train, share_val, ratio_test=share_test, test_anom=False)
            percent_true_train, percent_false_train, share_train = get_ratio_split(graphset, train_loader)
            percent_true_val, percent_false_val, share_val = get_ratio_split(graphset, val_loader)
            percent_true_test, percent_false_test, share_test = get_ratio_split(graphset, test_loader)
        else:
            print("Ensuring that test set has 50% anomalies and 50% normal cases.")
            backup_graph_name = graph_name[:-len('_5.graphset')] + '_5.graphset'
            print("loading graphset ", backup_graph_name)
            backup_graphset = GraphDataset(graph_dir, pd.DataFrame(), backup_graph_name, overwrite=False)
            if 'case' not in graph_name:
                print('large or huge detected. Overwriting shares of test, val and training set')
                share_train = 0.54
                share_val = 0.34
                share_test = 0.12
            if '_0_0' in dataset_name:
                #benchmarking dataset. Exempt from 50/50 anomaly ratio in test set.
                train_loader, val_loader, test_loader = split_dataset(graphset, backup_graphset, batch_size,
                                                                      share_train, share_val, ratio_test=share_test,
                                                                      test_anom=False)
            else:
                train_loader, val_loader, test_loader = split_dataset(graphset, backup_graphset, batch_size, share_train, share_val, ratio_test=share_test, test_anom=True)
            percent_true_train, percent_false_train, share_train = get_ratio_split(graphset, train_loader)
            percent_true_val, percent_false_val, share_val = get_ratio_split(graphset, val_loader)
            percent_true_test, percent_false_test, share_test = get_ratio_split(graphset, test_loader)


        print(f'Dataset shuffeld and split. Following percentage of anomalous cases: Training: {percent_true_train:.2%}({share_train:.2%}), Validation: {percent_true_val:.2%}({share_val:.2%}), Testing: {percent_true_test:.2%}({share_test:.2%}).')
        print('Scenario: \n', scenario)

        #GCN Network
        print('Start Training GCN Model. Training {} of {}.'.format(training_cntr, total_trainings))
        model_gcn = GCNet(graphset.num_features).to(device)
        optimizer = torch.optim.Adam(model_gcn.parameters(), lr=learning_rate)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        best_model, epoch = training_early_stop(model_gcn, epochs, train_loader, val_loader, device, optimizer,
                                               crit, patience, min_diff)
        missing, unexpected = model_gcn.load_state_dict(best_model)
        test_score_f1, test_score_roc = log_final_metrics(model_gcn, test_loader, device, engine, 'GCN', hyperparam_name, dataset_name, epoch, percent_true_train, share_train, percent_true_val, share_val, percent_true_test, share_test)
        print('Done Training GCN Model. Eval: F1: {:.5f}, AUC:{:.5f}'.format(test_score_f1, test_score_roc))
        training_cntr += 1

        #GINConv Network
        tmp_dim = 32
        print('Start Training GINConv Model. Training {} of {}.'.format(training_cntr, total_trainings))
        model_gin = GINConvNet(graphset.num_features, hidden_gin).to(device)
        optimizer = torch.optim.Adam(model_gin.parameters(), lr=learning_rate)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        best_model, epoch = training_early_stop(model_gin, epochs, train_loader, val_loader, device, optimizer,
                                                crit, patience,
                                                min_diff)
        missing, unexpected = model_gin.load_state_dict(best_model)
        test_score_f1, test_score_roc = log_final_metrics(model_gin, test_loader, device, engine, 'GIN', hyperparam_name, dataset_name, epoch, percent_true_train, share_train, percent_true_val, share_val, percent_true_test, share_test)
        print('Done Training GINConv Model. Eval: F1: {:.5f}, AUC:{:.5f}'.format(test_score_f1, test_score_roc))
        training_cntr += 1

        #Top-K-Pooling example network
        print('Start Training TopK Model. Training {} of {}.'.format(training_cntr, total_trainings))
        model_topk = TopKNet(graphset.num_features, ratio_topk).to(device)
        optimizer = torch.optim.Adam(model_topk.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(model_topk.parameters(), lr=0.005)

        #weight = torch.div(torch.sum(graphset.data.y==0), torch.sum(graphset.data.y==1)) #get ratio of true to false cases
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        #crit = torch.nn.BCELoss() #Not as numerically stable as BCEWithLogitsLoss()
        #crit = torch.nn.NLLLoss(reduction='none') #not as stabole as Crossentropy
        best_model, epoch = training_early_stop(model_topk, epochs, train_loader, val_loader, device, optimizer, crit, patience,
                                                min_diff)
        missing, unexpected = model_topk.load_state_dict(best_model)
        test_score_f1, test_score_roc = log_final_metrics(model_topk, test_loader, device, engine, 'TopK', hyperparam_name, dataset_name, epoch, percent_true_train, share_train, percent_true_val, share_val, percent_true_test, share_test)
        print('Done Training Top-K Model. Eval: F1: {:.5f}, AUC:{:.5f}'.format(test_score_f1, test_score_roc))
        training_cntr += 1

        #reset share variables
        share_train = 0.5
        share_val = 0.2
        share_test = 0.3

        print(scenario)

endtime = datetime.now()

duration = endtime - starttime
duration_min = divmod(duration.total_seconds(), 60)[0]
duration_sec = divmod(duration.total_seconds(), 60)[1]
print('Done. Started at {} and took {:.0f} minutes and {:.2f} seconds'.format(starttime, duration_min, duration_sec))

# Log Runs
run_name = [gen_name(2)]
run_log = pd.DataFrame(run_name, columns=['run_name'])
run_log['start'] = starttime
run_log['end'] = endtime
run_log['diff_min'] = duration_min
run_log['diff_sec'] = duration_sec
run_log['total_trainings'] = total_trainings
run_log.to_sql('runs', engine, if_exists='append')

# map run name to hyperparam name
run_map = pd.DataFrame()
run_map['run_name'] = run_name
num_hyperparam_name = hyperparams['name'].nunique()
run_map = run_map.loc[run_map.index.repeat(num_hyperparam_name)]
run_map['hyperparam_name'] = hyperparams['name'].unique()
run_map.to_sql('run_hyperparam', engine, if_exists='append')

