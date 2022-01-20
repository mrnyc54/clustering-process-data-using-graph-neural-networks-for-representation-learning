# inspired by https://github.com/khuangaf/PyTorch-Geometric-YooChoose
import torch
import os
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

class GraphDataset(InMemoryDataset):
    def __init__(self, root, pd_dataframe, setname, transform=None, pre_transform=None, overwrite=False, output_prefix=''):
        self.output_prefix = output_prefix
        if overwrite:
            if os.path.exists(root+'processed/'+setname):
                print(self.output_prefix, 'Path does exist: ', root, 'processed/', setname, '. Removing file...')
                os.remove(root+'processed/'+setname)
            else:
                print(self.output_prefix, 'Path does not exist:', root, 'processed/', setname, '. Creating file...')
        self.pd_dataframe = pd_dataframe
        self.setname = setname
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
        for caseId, group in tqdm(grouped):
            # For now only CaseId and activityName make their way into node feature. Add more later
            #node_features = group.loc[group['case:CaseId']==caseId,['case:CaseId','activityNameENEnc']].values
            
            #need to encode all values
            #print(type(grouped))
            node_features = group.loc[group['case:CaseId']==caseId].values #old command. may be more efficient this way:
            #node_features = node_features.astype('float64')
            #print(node_features[0])
            #print(node_features.dtype)
            #node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group['activityNameEN'].values[1:].astype(int)
            source_nodes = group['activityNameEN'].values[:-1].astype(int)
            
            edge_index = torch.tensor([source_nodes,
                                   target_nodes], dtype=torch.long)
            x = node_features
            
            # Generate Label for vizualization. May be removed later
            label = [group['case:Municipality'].mean()] * len(node_features)
            y = torch.FloatTensor(label)

            data = Data(x=x, edge_index=edge_index, y=y)

            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])