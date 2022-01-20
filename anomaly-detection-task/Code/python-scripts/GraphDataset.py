# inspired by https://github.com/khuangaf/PyTorch-Geometric-YooChoose
import torch
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

class GraphDataset(InMemoryDataset):
    def __init__(self, root, pd_dataframe, setname, transform=None, pre_transform=None, overwrite=False, output_prefix='', label_id=None, exclude_node_features=None):
        '''label_id: specify number of column to be used as label. exclude node features: list of columns to be excluded from pandas df for node features. includes labelid'''
        self.output_prefix = output_prefix

        if overwrite:
            if os.path.exists(root+'processed/'+setname):
                print(self.output_prefix, 'Path does exist: ', root, 'processed/', setname, '. Removing file...', flush=True)
                os.remove(root+'processed/'+setname)
            else:
                print(self.output_prefix, 'Path does not exist:', root, 'processed/', setname, '. Creating file...')
        self.pd_dataframe = pd_dataframe
        self.setname = setname
        self.whitelist_id = []
        #print('test labelid', label_id)
        if label_id is None and overwrite:
            self.label_id = 'undefined'
            #print('debug exclude n_f 1', exclude_node_features)
            # translate column names to their column_indexes if specified as string. Else take over integer. Mix of str and int allowed, but no negative ints
            exclude_node_features = [name if isinstance(name, int) else self.pd_dataframe.columns.get_loc(name) for name
                                     in exclude_node_features]
            #print('debug exclude n_f 2', exclude_node_features)

            # final whitelist of column_indices that can be used for node attributes. Excluding label + excluded feature indices
            self.whitelist_id = [col_id for col_id in range(len(self.pd_dataframe.columns)) if
                                 col_id not in exclude_node_features]
            #print('No label in pandas dataframe specified. calculating label based of values...')
        elif overwrite and isinstance(label_id,int):

            self.label_id = label_id if label_id > -1 else len(self.pd_dataframe.columns) + label_id  # translate negative integers
            exclude_node_features.append(self.label_id)

            # translate column names to their column_indexes if specified as string. Else take over integer. Mix of str and int allowed, but no negative ints
            exclude_node_features = [name if isinstance(name, int) else self.pd_dataframe.columns.get_loc(name) for name in exclude_node_features]
            #print('debug exclude n_f', exclude_node_features, self.label_id)

            # final whitelist of column_indices that can be used for node attributes. Excluding label + excluded feature indices
            self.whitelist_id = [col_id for col_id in range(len(self.pd_dataframe.columns)) if col_id not in exclude_node_features]

        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return [self.setname]

    def download(self):
        pass
    
    def process(self):
        data_list = []
        
        #Assumption: Data is already encoded (CaseId and ActivityName + whatever else you want to include)
        # sort data frame by caseid and timestamp
        sorted = self.pd_dataframe.sort_values(['case:CaseId', 'time:timestamp'], ascending=True) 

        # group data frame by caseid
        grouped = sorted.groupby('case:CaseId')

        if '_0_0' in self.setname:
            print(f'"_0_0" detected in {self.setname}! Special behaviour activated to include labels into node attributes.')

        for caseId, group in tqdm(grouped, position=0,leave=True):
            #print('Debug whitelist: ', self.whitelist_id)
            node_features = group.loc[group['case:CaseId']==caseId].values[:,self.whitelist_id] #Only include whitelisted ids

            node_features = torch.FloatTensor(node_features)
            #print('Test node_features', node_features)
            # Use positions of each node. Since they are ordered by timestamp we can just generate the numbers up to number of nodes in graph
            target_nodes = np.arange(1, len(group))
            source_nodes = np.arange(0, len(group) - 1)

            edge_index = np.array([source_nodes, target_nodes])

            edge_index = torch.tensor(edge_index, dtype=torch.long)
            x = node_features
            #print('Test node_features', node_features)
            # Label of case based off of either ActivityLabel or TimeLabel. If at least one is set to 1, the whole case is anomalous
            if self.label_id == 'undefined':
                if 'case:Municipality' in group:
                    label = [group['case:Municipality'].mean()] * len(node_features)
                elif 'ActivityLabel' in group:
                    #check, if timelabel also is in group
                    if 'TimeLabel' in group:
                        activity_anomaly_indicator = [group['ActivityLabel'].max()]
                        time_anomaly_indicator = [group['TimeLabel'].max()]
                        label = time_anomaly_indicator if sum(time_anomaly_indicator) >= sum(activity_anomaly_indicator) else activity_anomaly_indicator
                    else:
                        label = [group['ActivityLabel'].max()]
                elif 'TimeLabel' in group: #No ActivityLabel present as otherwise previous case would have been taken
                    label = [group['TimeLabel'].max()]
                else:
                    label = [0]
            else:
                # assumption: all labels in group (aka label per node within one case) have same value [0,1]
                label = [group.loc[group['case:CaseId']==caseId].values[:,self.label_id].max()]

            y = torch.FloatTensor(label)

            # if setname has '_0_0' in it, this indicates as a benchmark set. So labels will be included into node features as to proof that model can learn --> also print line at beginning of graph building warning
            if '_0_0' in self.setname:
                x = x + y

            data = Data(x=x, edge_index=edge_index, y=y)

            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])