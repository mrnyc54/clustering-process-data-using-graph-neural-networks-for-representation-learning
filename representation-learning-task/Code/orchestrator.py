import pandas as pd

from graph2vec import graph2vec
from dataset_generator import dataset_generator
from gen_hyperparams import gen_df
from kmeans import kmeans
from sqlalchemy import create_engine
from datetime import datetime
import ntpath
import math


def import_graph(logdir, graphdir, dataframename, graphname, engine_info, engine_embedding, mode=None, overwrite=False, output_prefix=''):
    '''
    handles functions of dataset generator_2
    :param logdir: path to xes logs
    :param graphdir: dir to graph. Means root folder(without processed folder)
    :param dataframename: name of dataframe. Name if table in DB (suffices 'no-enc' and 'enc' will be added)
    :param graphname: name of file, graph will be saved. must include file ending (e.g. '.graphset')
    :param engine: engine conncetion to sqlite db
    :param mode: either bpic15 or bpic19 or None. If you specify, casting and editing will be done accordingly
    :param overwrite: if True, overwrite saved graphset. If False, import from file if possible and do not consider xes files
    :return: graphset from gen.get_gs()
    '''

    path = graphdir + 'processed/' + graphname

    gen = dataset_generator(logdir, graphdir, log_name=dataframename, graph_name=graphname, engine=engine_embedding, mode=mode, output_prefix=output_prefix)
    if ntpath.exists(path) and not overwrite:
        gen.import_gs()

    if gen.get_gs() is None or gen.get_df() is None:
        print(output_prefix + "No graphset found or no dataframe imported. Recalculating Dataframe and graphset from xes eventlog")
        print(output_prefix + "importing xes logs")
        gen.import_xes()
        print(output_prefix , gen.get_xes_stats())
        print(output_prefix + 'convert xes to df')
        gen.conv_xes_df()
        print(output_prefix + 'done converting')
        print(output_prefix + 'save unencoded df')
        gen.save_df_sql( dataframename+ '_no-enc')
        print(output_prefix + 'encode df')
        gen.encode_df()
        print(output_prefix + 'normalize df')
        gen.normalize_df()
        print(output_prefix + 'save encoded + normalized df')
        gen.save_df_sql(dataframename + '_enc')
        gen.set_engine(engine_info)
        gen.save_df_sql(dataframename + '_enc')
        print(output_prefix + 'convert df to gs')
        gen.conv_df_gs()
        print(output_prefix + 'done converting!')
    return gen.get_gs()


def gen_hyperparams(engine, length, seed=None, whitelist=None, att_chance=None, extended_mode=0):
    hyperparams = gen_df(length, seed=seed, whitelist=whitelist, att_chance=att_chance,extended_mode=extended_mode)
    hyperparams.to_sql('hyperparams', con=engine, if_exists='append', index=False)

    return hyperparams


def embed(hyperparams, graphset, engine, engine_info, name='embedding_', graphsetname='default', bpic=None, default_prefix =''):
    inner_default_prefix = default_prefix + '\t'

    g2v = graph2vec(graphset, graphsetname, output_prefix=default_prefix+'\t')
    log=[]
    scenario = hyperparams
    start_scenario = datetime.now()
    start_scenario_time = start_scenario.time()
    #print(default_prefix+'scenario {} of {}'.format(index+1, len(hyperparams.index)), start_scenario_time)
    log_entry = [scenario['name'],datetime.now().date(),start_scenario_time]

    if g2v.get_degree() != scenario['degree'] or g2v.whitelist != scenario['columns'] or g2v.mode != scenario['extended_mode'] :
        g2v.mode = scenario['extended_mode'] == 1
        if bpic == 15:
            whitelist_lookup = pd.read_sql('PRAGMA TABLE_INFO(\'bpic-15_enc\') ', engine, index_col=['cid'])
            whitelist_lookup = whitelist_lookup.filter(['cid','name'])
            whitelist_lookup = [whitelist_lookup.at[id, 'name'] for id in scenario['columns']]
        elif bpic == 19:
            whitelist_lookup = pd.read_sql('PRAGMA TABLE_INFO(\'bpic-19_enc\') ', engine, index_col=['cid'])
            whitelist_lookup = whitelist_lookup.filter(['cid', 'name'])
            whitelist_lookup = [whitelist_lookup.at[id, 'name'] for id in scenario['columns']]
        else:
            whitelist_lookup = 'bpic value unknown. No attribute lookup possible. unknown attribute names.'


        print(default_prefix+'\t' + 'Recomputing subgraphs with degree {} and {} attributes and {} mode.'.format(scenario['degree'],
                                                                               len(scenario['columns']), 'extended' if scenario['extended_mode'] == 1 else 'simplified'))
        print(default_prefix+'\t' +'Name of Attributes: ', whitelist_lookup)
        print(default_prefix + '\t' + "Done transforming dataset to subgraphs", datetime.now().time())

        g2v.transform_dataset_subgraph(scenario['degree'], whitelist=scenario['columns'])
        words = g2v.get_words()
        num_words = len(words)
        num_unique_words = sum(1 for i in words.values() if int(i) == 1)
        num_words_below_mincount = sum(1 for i in words.values() if int(i) < scenario['min_count'])
        max_word = max(words, key=words.get)
        max_occurences = max(words.values())
        print(inner_default_prefix+'\t' + "{} distinct words / subgraphs generated. {} of these are unique ({:.2%}). ".format(num_words, num_unique_words, num_unique_words / num_words))
        print(inner_default_prefix+'\t' + "Most common word is {} with {} occurences.".format(max_word, max_occurences))
        print(inner_default_prefix+'\t' + "Min_count of {} is selected. Meaning, {} of words ({:.2%}) will be ignored, as they are below. " \
              .format(scenario['min_count'], num_words_below_mincount, num_words_below_mincount / num_words))
        # print(g2v.get_subgraphs()[-1:])
        timediff_subgraph = (datetime.now() - start_scenario).seconds
    log_entry.extend([num_words, num_unique_words, num_unique_words / num_words, scenario['min_count'], num_words_below_mincount, num_words_below_mincount / num_words])
    log_entry.extend([timediff_subgraph, datetime.now().date(),datetime.now().time()])
    g2v.embed(
        scenario['epochs'],
        scenario['vector_dimensions'],
        scenario['negative_sampling'],
        4,
        scenario['alpha'],
        scenario['window'],
        scenario['min_count']
    )
    #print('list of tagged docs: ', g2v.get_taggedDocuments()[:10])
    embedding = g2v.get_embedding()
    timestamp_id = str(datetime.now().year) + '-' +str(datetime.now().month) + '-' + str(datetime.now().day)+'_'+str(datetime.now().microsecond)
    table_name = name + scenario['name'] + '-' + timestamp_id
    embedding.to_sql(table_name, con=engine, if_exists='append', index=False)
    lookup = pd.DataFrame({'scenario': scenario['name'], 'embedding': table_name}, columns=['scenario','embedding'], index = [0])
    lookup.to_sql('hyperparams_to_embedding', con=engine, if_exists='append', index=False)
    end_scenario_time = datetime.now().time()
    print(default_prefix + 'done calculating scenario. Started at {} ended at {}'.format(start_scenario_time, end_scenario_time))
    log_entry.extend([datetime.now().date(),end_scenario_time])
    log.append(log_entry)
    log_df = pd.DataFrame(log,columns=['name','startdate','starttime','num_words/subgraphs','num_distinct_words/subgraphs','sharedisctinctwords','minimumcount','num_words_below_mincount','share_below_mincount','time_graph_to_string(seconds)','datestartembedding','timestartembedding','dateendembedding','timeendembedding'])

    runtime = pd.to_timedelta(pd.to_datetime(log_df['dateendembedding'].astype(str)+' ' + log_df['timeendembedding'].astype(str)) - pd.to_datetime(log_df['startdate'].astype(str) + ' ' + log_df['starttime'].astype(str)),unit='s')
    runtime = runtime.dt.seconds
    log_df['runtime(seconds)'] =  runtime

    log_df.to_sql('log',engine_info,if_exists='append')

def evaluate(engine_embedding, engine_info, name, truth, output_prefix =''):
    km = kmeans(engine_embedding, engine_info, name, true_table=truth, output_prefix=output_prefix)
    nmi_df = km.evaluate()
    km.write_to_sql(nmi_df)


def time_formatter(delta):
    hours = int(math.floor(delta.seconds / 3600))
    minutes = int(math.floor((delta.seconds - hours * 3600) / 60))
    seconds = delta.seconds - hours * 3600 - minutes * 60
    text = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return text


def main(whitelist=None, att_chance=None):
    starttime = datetime.now()


    # START OF PARAMETERS

    length = 25  # number of scenarios / hyperparam combinations to be run per whitelist
    # whitelist = [26]  # set whitelist. will be overwritten by run_cols_list
    att_chance = 20  # only active, if whitelist not specified. between 0 and 100. lower -> less attributes. overwritten by run_cols_list
    overwriteGraph = False # set to true to rebuild graphset from xes, independent of the graph already existing

    bpics = [15,19]   #select mode (15 or 19, anything else)
    extended_mode = 3 #0:default, simplified; 1: extended, 2: random, 3: both running the same, i.e. duplicate length.

    # END OF PARAMETERS



    mode=None   #dont change this. Only matters, if bpic is not 15 or 19
    log_dir = '../Data/EventLogs/' #dont change this. Only matters, if bpic is not 15 or 19.
    engine_embedding = create_engine('sqlite:///../Data/Database/embedding_default.db', echo=False) #dont change this. only relevant if dataste is neither bpic15 nor bpic19
    engine_info = create_engine('sqlite:///../Data/Database/graph2vec_info.db', echo=False)

    run_cols_list = [0]
    #num_cols = len(graphset.data.x[0][0])
    hyperparams_original = gen_hyperparams(engine_info, length, att_chance=att_chance, extended_mode=extended_mode)

    # add dataset and

    for bpic in bpics:
        if bpic == 15:
            graph_name = 'bpic-15.graphset'
            log_name = 'bpic-15' #name of eventlog as dataframe in db
            truth_name = log_name + '_enc'  # name of table that has truth
            mode = 'bpic15'
            log_dir = '../Data/EventLogs/BPIC2015/'
            engine_embedding = create_engine('sqlite:///../Data/Database/embedding_bpic15.db', echo=False)
            # multiple whitelists / o
            run_cols_list = [ #bpic_15:
                [8]  # org:resource,
                , [16]     # CaseType,
                , [8, 16]  # org:resource, CaseType
                , [18]     # case:ResponsibleActor
                , [8, 18]  # org:resource, case:ResponsibleActor
             ]
        elif bpic == 19:
            graph_name = 'bpic-19.graphset'
            log_name = 'bpic-19' #name of eventlog as dataframe in db + 'no-enc'
            truth_name = log_name + '_enc'  # name of table that has truth
            mode = 'bpic19'
            log_dir = '../Data/EventLogs/BPIC2019/'
            engine_embedding = create_engine('sqlite:///../Data/Database/embedding_bpic19.db', echo=False)
            run_cols_list = [  # bpic_19
                [5]  # org:resource,
                , [5, 8]    # org:resource, DocumentType
                , [8]       # DocumentType
                , [5, 14]   # org:resource, ItemCategory
                , [14]      # ItemCategory
                , [5, 12]   # org:resource, Vendor
                , [12]      # Vendor
            ]

        hyperparams = pd.concat([hyperparams_original]*len(run_cols_list), ignore_index=True)
        hyperparams = hyperparams.sort_values('name')
        hyperparams['columns'] = run_cols_list * length * 2 # * 2 bc extended mode is additionally to all hyperparam combinaitions
        graph_dir = '../Data/Graph/'
        print('debug hyperparams: ', type(hyperparams))
        print_prefix = '\t'
        print('creating/importing graph \t', datetime.now().time())
        graphset = import_graph(logdir=log_dir,graphdir=graph_dir,dataframename=log_name,graphname=graph_name, engine_info=engine_info, engine_embedding=engine_embedding, mode=mode, overwrite=overwriteGraph,output_prefix=print_prefix)
        graphset= graphset#[:100]
        graphset_time = datetime.now()

        print('embedding graphs \t\t\t', datetime.now().time())
        if run_cols_list:
            #for counter, cols in enumerate(run_cols_list):
            for counter, hyperparam in hyperparams.iterrows():
                cols = hyperparam['columns']
                start_whitelist_time = datetime.now().time()
                #print(print_prefix+'running {} of {} whitelists with {} scenarios in each.'.format(counter+1, len(run_cols_list), length), start_whitelist_time)
                #print(print_prefix+'\t'+'generating hyperparameter ', datetime.now().time())

                embedding_name = 'embedding_' + str(cols) + '-'
                embedding_start = datetime.now()
                print(print_prefix+'\t'+'embedding graphs in whitelist', embedding_start.time())
                # print('hyperparams:\n',hyperparams)
                embed(hyperparam, graphset, engine_embedding, engine_info, name=embedding_name, graphsetname=mode, bpic=bpic, default_prefix=print_prefix+'\t\t')
                end_whitelist_time = datetime.now().time()
                print(print_prefix+'done embedding graphs in whitelist {} of {}. Started at {} ended at {}' .format(counter+1, len(run_cols_list), start_whitelist_time, end_whitelist_time))

        else:
            print("This should not occur!")

        eval_start = datetime.now()
        print('evaluating embedding ', eval_start.time())
        table_name = 'nmi_' + str(bpic)
        evaluate(engine_embedding, engine_info, table_name, truth_name, output_prefix=print_prefix)
        endtime = datetime.now()

        print('Done run. ', endtime)
        embedding_time = eval_start - embedding_start
        print('Time spent for Embedding: \t', time_formatter(embedding_time))
        total_time = endtime - starttime
        print('Time spent in Total: \t\t', time_formatter(total_time))


if __name__ == '__main__':
    # pandas settings for output
    pd.set_option('display.width', 240)
    pd.set_option('display.max_columns', None)
    main()
