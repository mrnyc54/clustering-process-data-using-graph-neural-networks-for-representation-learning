import pandas as pd
import glob
from GraphDataset import GraphDataset
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sqlalchemy.exc import OperationalError
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
import ntpath

class dataset_generator():
    """
    class for importing and formatting data. Meant to be used for BPIC15 and BPIC19 Eventlogs in XES format as input and can output pandas dataframe and/or GraphDataset.
    initialized with paths to eventlog and graph, as well as doptional name, database conncetion and mode (BPIC15 or BPIC19).
    Offers methods to convert from xes to df (dataframe) and from df to gs (graphset aka GraphDataset).
    By doing so, it also adds columns 'caseId' and 'Municipality' (basically a y-label to evaluate later on).
    Inherently casts fields in dataframe and can also encode non-numeric fields (necessary for graph2vec implementation)
    For debugging, it can return the data it holds.
    """
    def __init__(self, eventlogdir, graphdir, log_name= "Generic_Log", graph_name='Generic_Graph', engine=None, mode=None, output_prefix=''):
        self.eventlogdir = eventlogdir if ntpath.exists(eventlogdir) else None
        self.graphdir = graphdir if ntpath.exists(graphdir) else None
        self.graph_name=graph_name
        self.log_name=log_name
        self.engine = engine
        modes = ['bpic15','bpic19']
        self.mode = mode if mode in modes else None
        self.dataframe = None
        self.xes = None
        self.field_lookup, self.mapping_lookup, self.as_type_lookup = self._create_casting_dict()
        self.graphset=None
        self.output_prefix = output_prefix

    def import_xes(self):
        print(self.output_prefix + "importing xes")
        paths = self._dirpath_to_filepaths(self.eventlogdir)
        num_workers = len(paths) if len(paths) < 5 else -3 #use as many workers as there are files, unless there are 5+ files. then use all cpus but two.
        xes_logs_list = Parallel(n_jobs=num_workers)(delayed(xes_importer.apply)(path) for path in paths) #pool.map(xes_importer.apply, paths)
        self.xes = xes_logs_list
        if self.mode == 'bpic15':
            print(self.output_prefix + "Flag for BPIC 15 set. Converting xes to fit that specific xes better")
            self._add_xes_bpic15()
        elif self.mode == 'bpic19':
            print(self.output_prefix + "Flag for BPIC 19 set. Converting xes to fit that specific xes better")
            self._add_xes_bpic19()
        print(self.output_prefix + "done importing xes")

    def get_xes_stats(self):
        num_logs = len(self.xes)
        num_cases_per_log = [len(log) for log in self.xes]
        num_cases = sum(num_cases_per_log)
        out = {'# Logs':num_logs, '# Cases':num_cases, '#Cases per Log':num_cases_per_log}
        return out

    def import_gs(self):
        print(self.output_prefix + "importing gs " )
        if self.engine is not None:
            sql = "SELECT * FROM ["+self.log_name+"_enc]"
            try:
                self.dataframe = pd.read_sql(sql, self.engine)
            except OperationalError:
                self.dataframe = None
        self.graphset = GraphDataset(self.graphdir, self.dataframe, self.graph_name, overwrite=False)
        num_nodes = sum(len(graph.x) for graph in self.graphset)
        if self.dataframe is None:
            print(self.output_prefix + "Warning! No DB connection established")
        elif num_nodes != len(self.dataframe):
            print(len(self.graphset) , len(self.dataframe))
            print(self.output_prefix + "Warning! Number of lines in Dataframe " + self.log_name+"_enc" + " does not match number of nodes in Graphset " + self.graph_name + ". This may lead to problems down when evaulating. You may need to speicify different dataframe for providing truth then.")
        print(self.output_prefix + "done importing gs")

    def conv_xes_df(self):
        num_workers = len(self.xes) if len(self.xes) < 5 else -3  # use as many workers as there are files, unless there are 5+ files. then use all cpus but two.
        description = self.output_prefix + 'converting log, completed logs'
        pd_list = Parallel(n_jobs=num_workers)(delayed(log_converter.apply)(xes_log, parameters={}, variant=log_converter.Variants.TO_DATA_FRAME) for xes_log in tqdm(self.xes, desc=description))
        self.dataframe = pd.concat(pd_list)
        if self.mode == 'bpic15':
            self._sort_df()
            self._cast_df()
        elif self.mode == 'bpic19':
            self.dataframe = self.dataframe[~(self.dataframe['case:Municipality'] == 'Standard')]
            self._sort_df()
            self._cast_df()

    def conv_df_gs(self, list_of_cols=None):
        print(self.output_prefix + 'converting pd to graphset')
        if list_of_cols:
            self.dataframe = self.dataframe[list_of_cols]
        self.graphset = GraphDataset(self.graphdir, self.dataframe, self.graph_name, overwrite=True)
        print(self.output_prefix + 'done converting to graphset')

    def encode_df(self):
        """Replace all object type columns with integer values depending on each columns value"""
        d = defaultdict(LabelEncoder)
        pd_log_enc = self.dataframe.apply(
            lambda x: x.astype('category').cat.codes if x.dtype == 'object' else x)
        self.dataframe = pd_log_enc

    def normalize_df(self, start_index=None, end_index=-1, exclude_cols=None):
        """normalize dataframe values from start_index to end_index (excluding end_index) excluding columns named in exclude_cols list."""

        norm_cols = self.dataframe.columns[start_index:end_index].drop(exclude_cols, errors='ignore')
        self.dataframe[norm_cols] = self.dataframe[norm_cols].apply(lambda x: (x - x.mean()) / x.std())

    def save_df_sql(self, name=None, engine=None):
        if engine is None:
            engine = self.engine
            if self.engine is None:
                print(self.output_prefix + "Trying to save to sql, but no engine defined!")
                pass
        if name is None:
            name = self.log_name
        self.dataframe.to_sql(name, con=engine, if_exists='replace', index=False)

    def _sort_df(self):
        print(self.output_prefix + "swapping case:CaseId and activityNameEN on first and second position in columns in dataframe")
        columns = self.dataframe.columns.to_list()
        columns = self._swap_list_items(columns, 'case:CaseId', 0)
        columns = self._swap_list_items(columns, 'activityNameEN', 1)
        columns = self._swap_list_items(columns, 'case:Municipality', -1)
        self.dataframe = self.dataframe[columns]

    def _cast_df(self):
        '''casts self. dataframe according to dictionaries set in _create_casting_dict'''
        print(self.output_prefix + 'casting according to mode ', self.mode,'. Includes {} for mapping dict, {} for field dict and {} for as_type_lookup'.format(len(self.mapping_lookup), len(self.field_lookup), len(self.as_type_lookup)))
        for col in self.dataframe.columns.to_list():
            if col in self.mapping_lookup:
                self.dataframe[col] = self.dataframe[col].map(self.mapping_lookup[col][1])
            if col in self.field_lookup:
                if self.field_lookup[col] == 'datetime':
                    self.dataframe[col] = pd.to_datetime(self.dataframe[col], utc=True)
                elif self.field_lookup[col] == 'numeric':
                    self.dataframe[col] = pd.to_numeric(self.dataframe[col], downcast='unsigned')
            if col in self.as_type_lookup:
                self.dataframe[col] = self.dataframe[col].astype(self.as_type_lookup[col][1])

    #getter setter functions
    def get_eventlogdir(self):
        return self.eventlogdir

    def set_eventlogdir(self, eventlogdir):
        self.eventlogdir = eventlogdir if ntpath.exists(eventlogdir) else None

    def get_graphdir(self):
        return self.graphdir

    def set_graphdir(self, graphdir):
        self.graphdir = graphdir if ntpath.exists(graphdir) else None

    def get_engine(self):
        return self.engine

    def set_engine(self, engine):
        self.engine = engine

    def get_xes(self):
        return self.xes

    def get_df(self):
        return self.dataframe

    def get_gs(self):
        return self.graphset

    #def _wrapper(self,func, arg):

    #helper functions
    def _dirpath_to_filepaths(self,dirpath):
        """gets path to dir and returns list of paths of all xes files in dir
        """
        list_xes_path = glob.glob(dirpath + '*.xes')
        list_xes_path.sort()
        return list_xes_path

    #bpic15 helper functions
    def _cast_df_bpic15(self):
        pass

    def _add_xes_bpic15(self):
        """
        adds the case attributes 'caseid' and 'municipality' to list of xes logs. should only be apllied, if mode is set to 'bpic15'
        :return:
        """
        if self.mode == 'bpic15':
            print(self.output_prefix + "Adding CaseAttributes CaseId and Municipality")
            cntr = 1
            for municipality, log in enumerate(self.xes):  # each eventlog is one municipality
                for trace in log:
                    trace.attributes['CaseId'] = cntr
                    trace.attributes['Municipality'] = municipality
                    cntr += 1
            print(self.output_prefix, cntr, " Cases affected.")

    #bpic19 helper functions
    def _add_xes_bpic19(self):
        if self.mode == 'bpic19':
            print(self.output_prefix + "Adding CaseAttribute CaseId")
            cntr = 1
            #print(self.xes)
            for log in self.xes:
                for trace in log:
                    trace.attributes['CaseId'] = cntr
                    trace.attributes['Municipality']=trace.attributes['Item Type']
                    for event in trace:
                        #print(event)
                        event['activityNameEN'] = event['concept:name']
                    cntr += 1
            print(cntr, " Cases affected.")


    def _create_casting_dict(self):
        field_lookup, mapping_lookup, as_type_lookup = '', '', ''
        if self.mode == 'bpic15':
            as_type_lookup = {'case:Responsible_actor': (pd.Series.astype,pd.Int32Dtype()), 'case:IDofConceptCase': (pd.Series.astype,pd.Int32Dtype()), 'case:landRegisterID': (pd.Series.astype,pd.Int32Dtype()), 'question':(pd.Series.astype,pd.Int8Dtype()) ,'case:Includes_subCases': (pd.Series.astype,pd.Int8Dtype())}
            mapping_lookup = {'question':(pd.Series.map,{'False': 0, 'True': 1}),'case:requestComplete':(pd.Series.map,
                {'FALSE': 0, 'TRUE': 1}),'case:Includes_subCases':(pd.Series.map,{'N': 0, 'J': 1})}
            field_lookup = {'dateFinished': 'datetime','dueDate': 'datetime','planned': 'datetime','time:timestamp': 'datetime','case:endDate': 'datetime','case:endDatePlanned': 'datetime','case:startDate': 'datetime','dateStop': 'datetime','monitoringResource': 'numeric','org:resource': 'numeric','case:case_type': 'numeric','case:concept:name': 'numeric','case:Responsible_actor': 'numeric','case:IDofConceptCase': 'numeric','case:landRegisterID': 'numeric','case:SUMleges': 'numeric','question': 'numeric','case:requestComplete': 'numeric','case:Includes_subCases': 'numeric'}
        elif self.mode == 'bpic19':
            as_type_lookup = {}
            mapping_lookup = {}
            field_lookup = {'case:CaseId':'numeric', 'Cumulative net worth (EUR)':'numeric', 'time:timestamp':'datetime',
                            'case:GR-Based Inv. Verif.':'numeric', 'case:Item':'numeric', 'case:Goods Receipt':'numeric'}
        return field_lookup, mapping_lookup, as_type_lookup


    def _swap_list_items(self,list, column_name, to_position):
        """list to be operated on (list of colums)
        column_name to be switched to to_position
        to_position index of list, to which name is to be put"""

        index_of_name = list.index(column_name)
        list[to_position], list[index_of_name] = list[index_of_name], list[to_position]
        return list
