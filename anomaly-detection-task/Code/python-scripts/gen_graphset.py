import pandas as pd
import gzip
import json
from GraphDataset import GraphDataset
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sqlalchemy import create_engine


def create_graphset(source_dir, source_name, graph_dir, graph_name, engine, sql_name, mode='none'):
    #masterclass. differentiating between synthetic and bpic logs
    if mode == 'bpic' or ('bpi' in source_dir.lower()) or ('bpi' in source_name.lower()):
        df = get_df_from_csv_bpic(source_dir, source_name)
        mode='bpic'
    else:
        # assuming syntheitic file as .json.gz
        df = get_df_from_json(source_dir, source_name)
    #print('mode: ', mode)
    print('Encoding dataframe')
    df_enc, encoding = encode_df(df)

    print('Normalizing dataframe')
    df_norm = normalize_df(df_enc,end_index=-2) #assumption: first two columns are case id and activityname, last two columns are labels or label-related

    print('Graphset to be generated of normalized df:\n', df_norm.head())
    if mode == 'bpic':
        graphset = GraphDataset(graph_dir, df_norm, graph_name, label_id=None, exclude_node_features=['ActivityLabel','TimeLabel','time:timestamp'], overwrite=True)
    else: #synthetic
        graphset = GraphDataset(graph_dir, df_norm, graph_name, label_id=-2, exclude_node_features=['case:kind'], overwrite=True)

    print(f'Graphset has {sum(graphset.data.y==1)/len(graphset.data.y):.2%} anomalies')

    print('Saving dfs to sql')
    df.to_sql(sql_name, engine, if_exists='replace', index=False)
    df_enc.to_sql(sql_name+'_encoded', engine, if_exists='replace', index=False)
    df_norm.to_sql(sql_name+'_normalized', engine, if_exists='replace', index=False)

    print('Calculating KPIs for sanity check')
    df_cases = df['case:CaseId'].nunique()
    num_graphs = len(graphset)

    df_events = len(df.index)
    num_nodes = len(graphset.data.x)
    print(f'Generated Graph {graph_name} and saved at {graph_dir} . \nSanity Check: Original df: {df_cases:,} cases with {df_events:,} events. Graphdataset: {num_graphs:,} Graphs with {num_nodes:,} nodes.')


def get_df_from_csv_bpic(csv_dir, csv_name):
    print('Importing {}{}!'.format(csv_dir,csv_name))
    # import csv eventlog into pandas df
    df = pd.read_csv(csv_dir + csv_name)

    #detect if bpic (2012 or 2013) data is used
    if 'AnomalousCompleteTimestamp' in df.columns:
        # rename columns for consistency between
        df = df.rename(columns={'CaseID': 'case:CaseId', 'AnomalousCompleteTimestamp': 'time:timestamp',
                            'AnomalousActivity': 'activityNameEN'})

        # convert timestamp column into correct datatype specific to bpic eventlogs + split teimestamp in multi columns
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
        df['year'] = pd.DatetimeIndex(df['time:timestamp']).year
        df['month'] = pd.DatetimeIndex(df['time:timestamp']).month
        df['day'] = pd.DatetimeIndex(df['time:timestamp']).day
        df['hour'] = pd.DatetimeIndex(df['time:timestamp']).hour
        df['minute'] = pd.DatetimeIndex(df['time:timestamp']).minute
        df['second'] = pd.DatetimeIndex(df['time:timestamp']).second
        df['time:timestamp'] = df['time:timestamp'].astype('uint64') / 10 ** 9

        # reorder columns. necessary to exclude unwanted cols by "take first n columns" later
        df = df[['case:CaseId', 'activityNameEN', 'AnomalousDuration', 'AnomalousCumDuration', 'year', 'month', 'day', 'hour', 'minute',
                 'second', 'time:timestamp','ActivityLabel', 'TimeLabel']]
        df['AnomalousDuration'] = df['AnomalousDuration'] / (60*60*24) #convert to days
        df['AnomalousCumDuration'] = df['AnomalousCumDuration'] / (60*60*24) #convert to days
        print('Highest value AnomalousDuration: ', df["AnomalousCumDuration"].max())
    else:
        print("No columns 'AnomalousCompleteTimestamp' or 'CompleteTimestamp' found in dataframe. Assuming 'time:timestamp' exists.")
    return df


    #print('debug df, output:\n',df.head())


def get_df_from_json(dir, name):
    """Imports .json.gzip files into dataframe assuming they are 'large' or 'huge' synthetic dataset
    timestamps are recreated from order of events in order to fit rest of pipeline"""
    path = dir + name
    with gzip.open(path, "rb") as f:
        j_eventlog = json.loads(f.read().decode("ascii"))
    ls_eventlog = []
    for case in j_eventlog['cases']:
        #obtain label and kind of anomaly
        if isinstance(case['attributes']['label'], str):
            cattr = 0 #normal case
            kind = ''
        else:
            cattr = 1 #anomalous case
            kind = case['attributes']['label']['anomaly']
        caseattr = cattr
        kind = kind
        #combine event and case attributes into one line
        for event_keys in range(len(case['events'])):
            min_event_data = [case['id'], case['events'][event_keys]['name'], event_keys]
            event_attributes = list(case['events'][event_keys]['attributes'].values())
            min_event_data.extend(event_attributes)
            min_event_data.append(caseattr)
            min_event_data.append(kind)
            ls_eventlog.append(min_event_data)
    df_eventlog = pd.DataFrame(ls_eventlog,
                      columns=['case:CaseId', 'ActivityNameEN', 'time:timestamp', 'company', 'country', 'day', 'user',
                               'case:label', 'case:kind'])
    return df_eventlog


def encode_df(df):
    # encode all fields that are of type object
    encoder = defaultdict(LabelEncoder)
    #df_enc = df.apply(lambda x: x.astype('category').cat.codes if x.dtype == 'object' else x)
    df_enc = df.apply(lambda x: encoder[x.name].fit_transform(x) if x.dtype == 'object' else x)
    return df_enc, encoder


def normalize_df(df, start_index=None, end_index = None, exclude_cols=None):
    """normalize dataframe values from start_index to end_index (excluding end_index) excluding columns named in exclude_cols list."""

    norm_cols = df.columns[start_index:end_index].drop(exclude_cols, errors='ignore')
    df[norm_cols] = df[norm_cols].apply(lambda x: (x - x.mean()) / x.std())
    return df


# pandas settings for output
pd.set_option('display.width', 240)
pd.set_option('display.max_columns', None)


graph_dir = '../../Data/Graph/'
graph_names = [
    'bpi2012_anomaly_events_0_0.graphset',
    'bpi2012_anomaly_events_0_1.graphset','bpi2012_anomaly_events_0_2.graphset','bpi2012_anomaly_events_0_3.graphset','bpi2012_anomaly_events_0_4.graphset','bpi2012_anomaly_events_0_5.graphset',
    'bpi2013_anomaly_events_0_0.graphset',
    'bpi2013_anomaly_events_0_1.graphset','bpi2013_anomaly_events_0_2.graphset','bpi2013_anomaly_events_0_3.graphset','bpi2013_anomaly_events_0_4.graphset','bpi2013_anomaly_events_0_5.graphset',
    'bpi2012_anomaly_cases_0_0.graphset',
    'bpi2012_anomaly_cases_0_1.graphset', 'bpi2012_anomaly_cases_0_2.graphset','bpi2012_anomaly_cases_0_3.graphset', 'bpi2012_anomaly_cases_0_4.graphset', 'bpi2012_anomaly_cases_0_5.graphset',
    'bpi2013_anomaly_cases_0_0.graphset',
    'bpi2013_anomaly_cases_0_1.graphset', 'bpi2013_anomaly_cases_0_2.graphset',
    'bpi2013_anomaly_cases_0_3.graphset', 'bpi2013_anomaly_cases_0_4.graphset', 'bpi2013_anomaly_cases_0_5.graphset',
    'large_0_0.graphset', 'huge_0_0.graphset',
    'large_0_1.graphset', 'huge_0_1.graphset',
    'large_0_2.graphset', 'huge_0_2.graphset',
    'large_0_3.graphset', 'huge_0_3.graphset',
    'large_0_4.graphset', 'huge_0_4.graphset',
    'large_0_5.graphset', 'huge_0_5.graphset'
]

df_dir = ['../../Data/EventLogs/bpi_2012/','../../Data/EventLogs/bpi_2013/', '../../Data/EventLogs/synthetic/']
df_names = [
    'bpi_2012_anomolous_events_0.0.csv',
    'bpi_2012_anomolous_events_0.1.csv','bpi_2012_anomolous_events_0.2.csv','bpi_2012_anomolous_events_0.3.csv','bpi_2012_anomolous_events_0.4.csv','bpi_2012_anomolous_events_0.5.csv',
    'bpi_2013_anomolous_events_0.0.csv',
    'bpi_2013_anomolous_events_0.1.csv','bpi_2013_anomolous_events_0.2.csv','bpi_2013_anomolous_events_0.3.csv','bpi_2013_anomolous_events_0.4.csv','bpi_2013_anomolous_events_0.5.csv',
    'bpi_2012_anomalous_cases_0.0.csv',
    'bpi_2012_anomalous_cases_0.1.csv','bpi_2012_anomalous_cases_0.2.csv','bpi_2012_anomalous_cases_0.3.csv','bpi_2012_anomalous_cases_0.4.csv','bpi_2012_anomalous_cases_0.5.csv',
    'bpi_2013_anomalous_cases_0.0.csv',
    'bpi_2013_anomalous_cases_0.1.csv','bpi_2013_anomalous_cases_0.2.csv','bpi_2013_anomalous_cases_0.3.csv','bpi_2013_anomalous_cases_0.4.csv','bpi_2013_anomalous_cases_0.5.csv',
    'large-0.0-1.json.gz', 'huge-0.0-1.json.gz',
    'large-0.1-1.json.gz', 'huge-0.1-1.json.gz',
    'large-0.2-1.json.gz', 'huge-0.2-1.json.gz',
    'large-0.3-1.json.gz', 'huge-0.3-1.json.gz',
    'large-0.4-1.json.gz', 'huge-0.4-1.json.gz',
    'large-0.5-1.json.gz', 'huge-0.5-1.json.gz'
]

df_sql_names = [
    'bpi2012_anomaly_events_0_0',
    'bpi2012_anomaly_events_0_1','bpi2012_anomaly_events_0_2','bpi2012_anomaly_events_0_3','bpi2012_anomaly_events_0_4','bpi2012_anomaly_events_0_5',
    'bpi2013_anomaly_events_0_0',
    'bpi2013_anomaly_events_0_1','bpi2013_anomaly_events_0_2','bpi2013_anomaly_events_0_3','bpi2013_anomaly_events_0_4','bpi2013_anomaly_events_0_5',
    'bpi2012_anomaly_cases_0_0',
    'bpi2012_anomaly_cases_0_1','bpi2012_anomaly_cases_0_2','bpi2012_anomaly_cases_0_3','bpi2012_anomaly_cases_0_4','bpi2012_anomaly_cases_0_5',
    'bpi2013_anomaly_cases_0_0',
    'bpi2013_anomaly_cases_0_1','bpi2013_anomaly_cases_0_2','bpi2013_anomaly_cases_0_3','bpi2013_anomaly_cases_0_4','bpi2013_anomaly_cases_0_5',
    'large_0_0', 'huge_0_0',
    'large_0_1', 'huge_0_1',
    'large_0_2', 'huge_0_2',
    'large_0_3', 'huge_0_3',
    'large_0_4', 'huge_0_4',
    'large_0_5', 'huge_0_5'
]

engine = create_engine('sqlite:///../../Data/Database/eventlogs.db', echo=False)

for id, name in enumerate(graph_names):
    if '2012' in name:
        dir = df_dir[0]
    elif '2013' in name:
        dir = df_dir[1]
    else:
        dir = df_dir[2]
    if '_0_0' in name:
        source_name = df_names[id].replace('_0_0','_0_1')
        source_name = source_name.replace('0.0', '0.1')
    else:
        source_name = df_names[id]
    create_graphset(dir, source_name, graph_dir, name, engine, df_sql_names[id])

