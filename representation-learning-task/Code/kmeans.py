import pandas as pd
import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine, inspect
from sklearn.cluster import KMeans
from sklearn import metrics, preprocessing

class kmeans():
    def __init__(self, engine_embedding, engine_info, output_table_name='nmi', true_dataframe=None, true_table=None, output_prefix=''):
        '''
        Connects to DB and evaluates stored embeddings. Assumed that all tables starting with
        'embedding_' are to be evaluated. Truth-labels can be provided either via dataframe or
        by providing the table name form the db.
        :param engine_embedding: DB Connection
        :param true_dataframe: dataframe with true labels
        :param true_table: table name with truth labels
        '''
        self.output_prefix = output_prefix
        self.engine_source = engine_embedding
        self.engine_target = engine_info
        self.true_dataframe = true_dataframe if true_dataframe else None
        self.true_table = true_table if true_table else None
        self.truth = self.establish_truth()
        self.clusters = self.truth.loc[:,0].nunique() if true_dataframe else self.truth.iloc[:,0].nunique()
        self.scenario_list = self.get_scenarios()
        self.nmi_table_name = output_table_name

    def establish_truth(self):
        if self.true_dataframe:
            truth = self.true_dataframe
            print(self.output_prefix + 'Truth provided by dataframe. {} instances with {} cluster found.'.format(len(truth), truth.loc[:,0].nunique()))
        elif self.true_table:
            query = 'SELECT [case:CaseId], AVG([case:Municipality]) FROM [' + self.true_table + '] GROUP BY [case:CaseId]'
            truth = pd.read_sql(query, index_col=['case:CaseId'], con=self.engine_source)

            print(self.output_prefix + 'Truth provided by table. {} instances with {} cluster found.'.format(len(truth), truth.iloc[:,0].nunique()))
        else:
            print(self.output_prefix + 'Warning: No truth provided!')
            truth = pd.DataFrame()
        return truth

    def get_scenarios(self):
        '''
        build list of table names to evaluate
        :return: list of strings of table names
        '''
        inspector = inspect(self.engine_source)
        list_of_scenarios = [name for name in inspector.get_table_names() if 'embedding_' in name and 'hyperparams_to_embedding_final' not in name]
        print(self.output_prefix + 'Done looking for scenarios: {} found!'.format(len(list_of_scenarios)))
        return list_of_scenarios

    def evaluate_scenario(self, scenario):
        embedding = pd.read_sql(scenario, con=self.engine_source).iloc[:, 1:].to_numpy()
        kmeans = KMeans(n_clusters=self.clusters)
        kmeans.fit(preprocessing.normalize(embedding, norm='l2'))
        label = kmeans.predict(preprocessing.normalize(embedding, norm='l2'))
        true_label = self.truth.to_numpy().flatten()
        if len(label) != len(true_label):
            print(self.output_prefix + 'Warning! Truth ({}) and labels ({}) do not match in evaluation. Assuming Dev mode and trimming truth!'.format(len(true_label), len(label)))
            true_label = true_label[:len(label)]
        nmi = metrics.normalized_mutual_info_score(true_label, label)
        return nmi

    def evaluate(self):
        print(self.output_prefix + 'scenario list: ', self.scenario_list)
        nmi_list = [self.evaluate_scenario(scenario) for scenario in tqdm(self.scenario_list,desc='Evaluating ')]
        nmi_df = pd.DataFrame(zip(nmi_list,self.scenario_list), columns=['nmi','id'])
        nmi_df.set_index('id', inplace=True)
        #stats
        id_min = nmi_df['nmi'].idxmin()
        nmi_min = nmi_df['nmi'].min()
        id_max = nmi_df['nmi'].idxmax()
        nmi_max = nmi_df['nmi'].max()
        nmi_mean = nmi_df['nmi'].mean()
        nmi_median = nmi_df['nmi'].median()
        print(self.output_prefix + 'Best scenario is {} with NMI of {:1.3f}.'.format(id_max, nmi_max))
        print(self.output_prefix + 'Worst scenario is {} with NMI of {:1.3f}.'.format(id_min, nmi_min))
        print(self.output_prefix + 'Average NMI is {:1.3f} with median {:1.3f}'.format(nmi_mean, nmi_median))
        return nmi_df

    def write_to_sql(self, nmi_df):
        nmi_df.to_sql(self.nmi_table_name, con=self.engine_target, if_exists='replace')
