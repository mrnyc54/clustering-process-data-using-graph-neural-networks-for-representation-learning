import ntpath
import pickle
from itertools import chain

import pandas as pd
import numpy as np
import torch
import copy
from sqlalchemy import create_engine
from datetime import datetime
from GraphDataset import GraphDataset
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from typing import List, Dict
from joblib import Parallel, delayed
import math

# redesign graph2vec with class + parallelization w/ 'embarassingly parallel' in mind
class graph2vec():
    def __init__(self, graphset, graphset_name, output_prefix='', mode = False):
        #store engine, graphset and subgraphs here.
        self.graphset_name = graphset_name
        self.graphset = graphset
        self.subgraphs = []
        self.subgraphs_att = []
        self.embedding = []
        self.degree = -1
        self.blacklist = []
        self.whitelist = []
        #self.num_attr = -1
        self.dict_of_words = {}
        self.taggedDocuments = []
        self.output_prefix = output_prefix
        self.mode = mode

    def embed(self, epochs, vector_size, negative_samples, workers, alpha, window, min_count):
        progress = MonitorCallback(epochs, self.output_prefix)
        print(self.output_prefix + 'Training Embedding w {} epochs, {} negative samples, {} window size'.format(epochs,negative_samples,window))
        model = Doc2Vec(self.taggedDocuments,
                        vector_size=vector_size,
                        dm=0,
                        hs=0,  # 0=negative sampling, 1=hierarchical softmax
                        negative=negative_samples,  # no. of negative samples
                        workers=workers,
                        epochs=epochs,
                        alpha=alpha,
                        window=window,
                        callbacks=[progress],
                        min_count=min_count  # minimal occurence per "word" to be included into vocabulary.
                        )
        self.embedding = self.__embedding_DataFrame(model)

    def transform_dataset_subgraph(self, degree, blacklist=None, whitelist=None, timing_only=0):
        """
        transforms classes dataset into subgraphs of specified degree. allows for reducing node attributes to be considered.
        :param degree: neighborhood degree of nodes to be considered. if 0, only the node itself will be included.
        :param blacklist: list of indexes of node attributes to be left out
        :param whitelist: list of indexes of node attributes to be left in
        :param timing_only: optional integer. If larger than 0, only the subgraph will be created, no documents based on whitelist will be created. default is 0
        :return:
        """
        self.blacklist = blacklist
        self.whitelist = whitelist if whitelist is not None else blacklist_to_whitelist(self.graphset, blacklist)
        self.degree = degree

        splits = 1

        split_graphset = self._split_gs(self.graphset, splits)
        self.subgraphs = self._subgraph_splits(split_graphset, degree, self.whitelist)
        if timing_only < 1:
            print(self.output_prefix + 'Filling Subgraph blueprint with Attributes...', datetime.now().time())
            self.subgraphs_att = subgraphset_to_attributes(self.graphset,self.subgraphs,self.whitelist)
            print(self.output_prefix + 'Done filling with Attributes!', datetime.now().time())

            self.taggedDocuments, dict_of_words = list_to_taggedDocument(self.subgraphs_att)
            self.dict_of_words = dict_of_words

    def __embedding_DataFrame(self, model: Doc2Vec) -> pd.DataFrame:
        tmp_out = []
        for key in model.dv.index_to_key:
            # print(counter)
            tmp_out2 = [key]
            tmp_out2.extend(model.dv[key])
            tmp_out.append(tmp_out2)
        out = pd.DataFrame(tmp_out)
        return out

    def _split_gs(self, graphset, splits, min_step = 50):
        #determine step size in order to  evenly split the dataset into n (aka splits) parts.
        stepsize = int(len(self.graphset) / splits)
        if stepsize < min_step:
            splitted_graphset = [graphset]
        else:
            splitted_graphset = [graphset[n*stepsize:(n+1)*stepsize] if n < splits-1 else graphset[n*stepsize:] for n in range(splits) ]
        return splitted_graphset


    def _subgraph_splits(self, splitset, degree, whitelist):
        extended_mode = self.mode

        out = self._load_subgraph(self.graphset_name, self.degree, extended_mode)
        if len(out) == 0:
            print(self.output_prefix + 'No subgraph blueprint found. Calculating from graphset.')
            for index, set in enumerate(splitset):
                description = self.output_prefix + 'Calculating subgraphs for splitset {} of {}'.format(index+1, len(splitset))
                #set_subgraph = Parallel(n_jobs=1)(delayed(graph_to_subgraph_2)(graph, degree, whitelist, extended_mode) for graph in tqdm(set, desc=description))
                set_subgraph = (graph_to_subgraph_2(graph, degree, whitelist, extended_mode) for graph in
                    tqdm(set, desc=description))
                out.extend(set_subgraph)
            self._persist_subgraph(self.graphset_name, self.degree, out, extended_mode)
        return out

    def _persist_subgraph(self, name, degree, list, mode):
        path = '../Data/Graph/Subgraphs/'
        suffix = '_extended' if mode else '_simplified'
        name = str(name) + '_' + str(degree) + suffix +'.pickle'
        full_path = path + name
        with open(full_path, "wb") as persist:
            pickle.dump(list,persist)

    def _load_subgraph(self, name, degree, mode):
        path = '../Data/Graph/Subgraphs/'
        suffix = '_extended' if mode else '_simplified'
        name = str(name) + '_' + str(degree) + suffix +'.pickle'
        full_path = path + name
        list=[]
        if ntpath.exists(full_path):
            with open(full_path, "rb") as load:
                list = pickle.load(load)
        return list

    #getter / setter functions
    def set_graphset(self, graphset):
        self.graphset = graphset
        #delete stored subgraphs. No different states of attributes within instance allowed
        self.subgraphs = []
        self.embedding = []
        self.degree = -1
        self.blacklist = []
        self.whitelist = []
        self.num_attr = -1
        self.dict_of_words = {}
        self.taggedDocuments = []

    def get_graphset(self):
        return self.graphset

    def get_subgraphs(self):
        return self.subgraphs

    def get_embedding(self):
        return self.embedding

    def get_taggedDocuments(self):
        return self.taggedDocuments

    def get_words(self):
        return self.dict_of_words

    def get_degree(self):
        return self.degree


class MonitorCallback(CallbackAny2Vec):
    def __init__(self, epochtotal: int, output_prefix=''):
        self.output_prefix = output_prefix
        self._epoch = 0
        self.epochtotal = epochtotal
        self._starttime = datetime.now()
        self._lasttime = datetime.now()

    def time_formatter(self, delta):
        hours = int(math.floor(delta.seconds / 3600))
        minutes = int(math.floor((delta.seconds - hours * 3600) / 60))
        seconds = delta.seconds - hours * 3600 - minutes * 60
        text = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
        return text

    def on_epoch_end(self, model):
        self._currenttime = datetime.now()
        if self._epoch == 0 or self._epoch == self.epochtotal:
            self.output_status(model)
        elif self._epoch % 10 == 0 and self.epochtotal > 15:
            self.output_status(model)
        self._epoch += 1
        self._lasttime = self._currenttime

    def output_status(self, model):
        self._currenttime = datetime.now()
        self._totalduration = self._currenttime - self._starttime
        self._totalduration = self.time_formatter(self._totalduration)
        self._epochduration = self.time_formatter(self._currenttime - self._lasttime)

        print(self.output_prefix + "Epoch {:>3} of {:>3}. Duration Epoch: {} \ttotal: {} \tstarttime: {} \tcurrenttime: {}". \
              format(self._epoch + 1, self.epochtotal, self._epochduration, self._totalduration,
                     datetime.strftime(self._starttime, "%H:%M:%S"),
                     datetime.strftime(self._currenttime, "%H:%M:%S")))  # print loss


# helper functions outside of class
def node_id_to_label(graph, nodeid, debug=False):
    if debug: print('function node_id_to_label of graph ', graph.x[0][0], ' \t and node_id ', nodeid, type(nodeid))
    labels = -1
    if debug: print('debug node_id_to_label part 2: ', graph.x[nodeid].tolist())
    # labels = (graph.x[nodeid].tolist()[0][1]) #assumption: datalabel is second entry of feature vector
    labels = (graph.x[nodeid].tolist()[1])  # assumption: datalabel is second entry of feature vector
    return labels


def blacklist_to_whitelist(graphset, blacklist):
    """
    check return whitelist matching to blacklist, if provided. Else return whitelist.
    :param graphset: graphset to be reduced
    :param blacklist: list if indexes to exclude
    :return: whitelist:list of col indexes to be included
    """
    whitelist = np.arange(len(graphset.data.x[0][0]))  # assumption: all nodes in graphsdataset have the same number of attributes as the first node in the first graph
    if blacklist is not None:
        whitelist = np.delete(whitelist, blacklist).tolist()
    return whitelist


def label_to_node_ids(graph, label):
    node_ids = []
    # node_ids.extend((graph.x[:,1] == label).nonzero(as_tuple=True)[0].tolist()) #multiple node_ids can respond to one label (loops)
    node_ids.extend(
        (graph.x[:, 1] == label).nonzero()[0].tolist())  # multiple node_ids can respond to one label (loops)
    return node_ids

def list_to_taggedDocument(graphset: List[List[List[str]]], prefix: str = None) -> List[TaggedDocument]:
    """convert a list of list of strings to a List of TaggedDocuments. Optional prefix for string can be supplied."""
    output = []
    words = {}
    if prefix == None: prefix = 'graph_'
    for counter, graph in enumerate(graphset):
        #reduce subgraphs: The input is formatted in a slightly easier-to-read way, but not in a way useful to TaggedDocument
        #e.g. Input format has a list of subgraphs per attribute on node level. This needs to be concatenated to graph level in order to count each subgraph for embedding
        tmp_node = [item for sublist in graph for item in sublist]
        for word in tmp_node:
            words[word] = 1 + words.get(word,0)
        output.append(TaggedDocument(tmp_node, tags=[prefix + str(counter)]))
    return output, words

def embedding_to_dataframe(model: Doc2Vec) -> pd.DataFrame:
    tmp_out = []
    for key in model.dv.index_to_key:
        # print(counter)
        tmp_out2 = [key]
        tmp_out2.extend(model.dv[key])
        tmp_out.append(tmp_out2)
    out = pd.DataFrame(tmp_out)
    return out

# actual legwork functions
def graphset_to_subgraph(graphset, degree, attr_dims=None, extend_degree=False):
    subgraphs = (graph_to_subgraph_2(graph, degree, attr_dims, extend_degree) for graph in graphset)
    return subgraphs

def graph_to_subgraph_2(graph, degree, attr_dims=None, extend_degree=False):
    #since graph.x stores multiples of the same node (for each connection ) we only need to iterate through each version of each unique node once.
    node_labels = graph.x[:, 1] #assumption: activityName /main node label is at index 1 / position 2 get all labels
    unique_labels_node, unique_labels_node_ids = np.unique(node_labels, return_index=True) # get indices of first node of each label in datastructure
    label_to_nodeid = {label: index for label, index in zip(unique_labels_node,unique_labels_node_ids)}

    sorted_unique_labels_node_ids = np.sort(unique_labels_node_ids)
    if extend_degree:
        graph_to_subgraphs = [node_to_subgraph(node, graph, degree, label_to_nodeid, extend_degree) for node in
                              sorted_unique_labels_node_ids]
        graph_to_subgraphs = list(chain.from_iterable(graph_to_subgraphs))
    else:
        graph_to_subgraphs = [node_to_subgraph(node, graph, degree, label_to_nodeid, extend_degree) for node in sorted_unique_labels_node_ids]
    #print('debug: graph subgraph: ', graph_to_subgraphs)
    # idea: I have now a blueprint of node indices. I just need to fill it with each node's attributes at each point.


    #print('test subgraph_versions: ', subgraph_versions)
    return graph_to_subgraphs

def subgraphset_to_attributes(graphset, subgraphset,attr_dims):
    """

    :param graphset: list of GraphDataset. needs to contain the attributes
    :param subgraphset: list of list of integers, representing nodeids. "BluePrint" of Subgrpahs per node.
    :param attr_dims: list of integers, representing attributes. For each attribute, subgraph will get a copy
    :return: list of list of strings representing subgraphs
    """
    subgraphset_att = []
    for index, graph in enumerate(subgraphset):
        attributes = graphset[index].x
        subgraph_versions = []
        for attr_dim in attr_dims:
            attr = dict(enumerate(attributes[:, attr_dim].flatten()))
            subgraph = subgraph_to_attributes(graph, attr)
            subgraph_versions.extend(subgraph)
        subgraphset_att.append(subgraph_versions)
    return subgraphset_att



def node_to_subgraph(node_id, graph, degree, label_to_node_id, extend_degree=False):
    """
    target method for parallelization on node level instead of graph level. I.e. one process does not calculate all subgraphs of one graph but one process calculates the subgraph of one particular node of one graph (in all of its dimensions)
    :param node_id: Id of node to get subgraphs from
    :param graph: graph from which node comes from
    :param degree: degree to which neighboring nodes are to be considered
    :param attr_dims: list of indices that represent each nodes attributes to get included. Default is None (all attributes from root node get considered)
    :param extend_degree: boolean wether to include subgraphs with degrees leading up to target degree or not (i.e. degree 4 would include degrees 1,2 and 3). Default is False (no inclusion)
    :return: list of strings representing a number of subgraphs (one for each attribute dimension) of this node
    """
    node_subgraphs = []
    if extend_degree:
        subgraph_per_att = []
        for d in range(degree + 1):
            subgraph_per_deg_per_att = get_subgraph(node_id, graph, d, label_to_node_id).tolist()
            #joined = ','.join(str(item) for item in subgraph_per_deg_per_att.tolist())
            subgraph_per_att.append(subgraph_per_deg_per_att)
        node_subgraphs.extend(subgraph_per_att)
    else:
        subgraph_per_att = get_subgraph(node_id, graph, degree, label_to_node_id).tolist()
        node_subgraphs.extend(subgraph_per_att)
    return node_subgraphs


# Next try of this method: I only want the indexes of the nodes that make up the subgraph, and I only want the lowest index if multiple are available.
def get_subgraph(node_id, graph, degree, label_lookup):
    sgn = [] #variable used by graph2vec paper
    label = node_id_to_label(graph, node_id) #label is necessary to uniquely id nodes. necessary to "move" around graph

    if degree == 0:
        sgn = [node_id] # assumption: sgn will always be empty at this point of code
    else:
        #find neighbors, get their subgraphs with degree-1 and concat with this node's subgraph of degree-1

        #find neighbor nodes:
        from_index = (graph.edge_index[0] == label).nonzero(as_tuple=True)[0].numpy() #find all 'versions' of this node in graph
        target_labels = torch.unique(graph.edge_index[1, from_index]).tolist() #find target label of each 'version' + remove duplicates
        target_index = [label_lookup[label] for label in target_labels]

        #get neighbors' subgraphs aka M as variable in paper graph2vec
        neighbor_subgraph = np.array([])
        for neighbor_index in target_index:
            subgraph = get_subgraph(neighbor_index, graph, degree - 1, label_lookup) #recursion
            # np arrays work differently, if they are empty. need to find out, if neighbor_subgraph is empty or not:
            if neighbor_subgraph.size > 0 and subgraph.size > 0:  # neighbor_subgraph has prior subgraphs
                if np.ndim(neighbor_subgraph) < 2 and np.ndim(subgraph) < 2:
                    neighbor_subgraph = np.concatenate((neighbor_subgraph, subgraph))
                else:
                    neighbor_subgraph = np.concatenate((neighbor_subgraph, subgraph), axis=1)
            elif neighbor_subgraph.size == 0:  # neighbor_subgraph has no prior subgraphs
                neighbor_subgraph = subgraph
        neighbor_subgraph = np.reshape(neighbor_subgraph, (-1, 1))
        root_subgraph = get_subgraph(node_id, graph, degree - 1, label_lookup)

        #sort M + concat
        neighbor_subgraph.sort()
        if root_subgraph.size > 0 and neighbor_subgraph.size > 0:
            sgn = np.unique(np.concatenate((root_subgraph, neighbor_subgraph), axis=None)) #axis None is to ensure, that output is one dimensional. Sometimes M is of dim 2x1 unnecessarily
        elif root_subgraph.size == 0 and neighbor_subgraph.size > 0:
            sgn = np.unique(neighbor_subgraph)
        elif root_subgraph.size > 0 and neighbor_subgraph.size == 0:
            sgn = root_subgraph
    return np.array(sgn)

def subgraph_to_attributes(graph_subgraphs,attribute_dict):
    """
    Replaces a given graph's subgrpah (list of lists, with arbitrary length and nodeid as value) with the attribute mapped via attribute_dict
    :param graph_subgraphs: subgraph set. list of subgraphs, that are list of integers.
    :param attribute_dict: dictionary mappnig nodeid to attribute value
    :return: same structure as grpah_subgraph but values have been replaced.
    """
    # Hier wäre möglicher ansatz für implementierung von "ausklappen" der node_ids aka inkludierung der verschiedenen nodes mit gleichem label/activityname
    # aktuell würde bei loop nur attr von ersten event übernommen werden. bei 2 mal gleiches event in gleichem case wäre eigenschaften von allen nicht-ersten events irrelevant
    graph_subgraphs_copy = [[','.join([str(attribute_dict[node_id]) for node_id in node_subgraph])] for node_subgraph in graph_subgraphs]
    return graph_subgraphs_copy