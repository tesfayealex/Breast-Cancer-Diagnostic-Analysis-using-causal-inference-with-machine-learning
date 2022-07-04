import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.discretiser.discretiser_strategy import ( DecisionTreeSupervisedDiscretiserMethod )
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from IPython.display import Markdown, display, Image, display_html
import pandas as pd
from logger import logger
from datetime import date, datetime


class HandleDataFrame:
    def extract_dataframe(self,df: pd.DataFrame , columns: list) -> list:
        try:
            ext_df = df[columns]
            return ext_df
        except Exception as e:
            logger.error(e)
            return pd.DataFrame()
    def scale_dataframe(self,df:pd.DataFrame):
        try:
            minmax_scaler = MinMaxScaler()
            scaled = pd.DataFrame(minmax_scaler.fit_transform(df), columns=df.columns.to_list())
            return scaled
        except Exception as e:
            logger.error(e)
            return pd.DataFrame()
    def normalize_dataframe(self,df:pd.DataFrame):
        try:
            normal = Normalizer()
            normalized = pd.DataFrame(normal.fit_transform(df), columns=df.columns.to_list())
            return normalized
        except Exception as e:
            logger.error(e)
            return pd.DataFrame()
    def create_structural_model(self,df:pd.DataFrame,parent_node):
        try:
            sm = from_pandas(df, tabu_parent_nodes=parent_node,)
            return sm
        except Exception as e:
            logger.error(e)
            return pd.DataFrame() 
    def create_structural_model_with_threshold(self,df:pd.DataFrame ,parent_node, threshold: float):
        try:
            sm = from_pandas(df, tabu_parent_nodes=parent_node,)
            sm.remove_edges_below_threshold(threshold)
            return sm
        except Exception as e:
            logger.error(e)
            return pd.DataFrame()
    def draw_graph_from_model(self,sm):
        try:
            viz = plot_structure( sm, graph_attributes={"scale": "2.5",  "size": 2.5}, all_node_attributes=NODE_STYLE.WEAK,all_edge_attributes=EDGE_STYLE.WEAK)
            Image(viz.draw(format='png'))
            return viz
        except Exception as e:
            logger.error(e)
            return pd.DataFrame()  
    def select_database_percentage(self,df,percentage):
        size = int(df.shape[0] * percentage)
        return df[0:size:]
    def jaccard_similarity(self,g, h):
        i = set(g).intersection(h)
        u = set(g).union(h)
        return len(i)/len(u)
    def getOverview(self,df) -> None:

        _labels = [column for column in df]  # Only numeric columns
        _count = df.count().values
        _unique = [df[column].value_counts().shape[0] for column in df]
        nullSum = df.isna().sum()
        _missing_values =  [col for col in nullSum]

        columns = [
            'label',
            'count',
            'none_count',
            'none_percentage',
            'unique_value_count',
            'unique_percentage',
            'dtype']
        data = zip(
            _labels,
            _count,
            _missing_values,
            [str(round(((value / df.shape[0]) * 100), 2)) + '%' for value in _missing_values],
            _unique,
            [str(round(((value / df.shape[0]) * 100), 2)) + '%' for value in _unique],
            df.dtypes
        )
        new_df = pd.DataFrame(data=data, columns=columns)
        new_df.set_index('label', inplace=True)
        new_df.sort_values(by=["none_count"], inplace=True)
        return new_df
    
    def add_column(self,df, column , column_value):
        df.insert(loc=0, column=column, value=column_value)
        return df 
    def get_features(self,df,diffrence_column):
        features = list(df.columns.difference(['diagnosis']))
        return features
    def change_dataframe_to_discrete(self,main_df , extracted_df , features):
        tree_discretiser = DecisionTreeSupervisedDiscretiserMethod(
            mode='single',
            tree_params={'max_depth': 3, 'random_state': 2021},
        )
        tree_discretiser.fit(
            feat_names=features,
            dataframe=extracted_df,
            target_continuous=True,
            target='diagnosis',
        )
        tree_discretiser
        discretised_data = main_df.copy()
        for col in features:
            discretised_data[col] = tree_discretiser.transform(main_df[[col]])
        
        return discretised_data
    def get_scores(self,true_values,pred_values):
        print('Recall: {:.2f}'.format(recall_score(y_true=true_values, y_pred=pred_values)))
        print('F1: {:.2f} '.format(f1_score(y_true=true_values, y_pred=pred_values)))
        print('Accuracy: {:.2f} '.format(accuracy_score(y_true=true_values, y_pred=pred_values)))
        print('Precision: {:.2f} '.format(precision_score(y_true=true_values, y_pred=pred_values)))

       