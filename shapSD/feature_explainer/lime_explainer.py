"""
provide explanations using LimeExplainer
author: Xiaoqi
date: 2019.07.30
"""

from lime import lime_tabular
from lime import lime_text
from lime import lime_image
from .logging_custom import *
import pandas as pd


class LimeExplainer(object):

    def __init__(self, x_train, model, explainer_type='tabular'):
        self.x_train = x_train
        self.model = model
        self.explainer_type = explainer_type

        if self.explainer_type == 'tabular':
            tabular_explainer = lime_tabular.LimeTabularExplainer
            self.explainer = self.get_tabular_explainer(tabular_explainer)
        elif self.explainer_type == 'text':
            self.explainer = lime_text.LimeTextExplainer
        elif self.explainer_type == 'image':
            self.explainer = lime_image.LimeImageExplainer
        else:
            raise TypeError('Does not support {} explainer'.format(self.explainer_type))

    def get_tabular_explainer(self, tabular_explainer):
        data = self.x_train.copy()
        # check whether contains categorical features
        cat_cols = data.select_dtypes(exclude=['number']).columns
        try:
            # have categorical features
            if len(cat_cols) > 0:
                cat_features = [list(self.x_train.columns).index(col) for col in cat_cols]
                data[cat_cols] = data[cat_cols].astype('category')
                # label encoding
                data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)

                # map dictionary to label encoding
                label_dic = {}
                for i, col in enumerate(cat_cols):
                    label_dic[cat_features[i]] = dict(enumerate(self.x_train[col].astype('category').cat.categories))

                self.x_train = data
                lime_tab_explainer = tabular_explainer(self.x_train.values,
                                                feature_names=self.x_train.columns,
                                                categorical_features=cat_features,
                                                categorical_names=label_dic,
                                                discretize_continuous=True)

                return lime_tab_explainer
            else:
                lime_tab_explainer = tabular_explainer(self.x_train.values,
                                                feature_names=self.x_train.columns,
                                                discretize_continuous=True)
                return lime_tab_explainer
        except Exception as err:
            print('Error: model is not supported by LIME {} Explainer'.format(self.explainer_type))
            err_logging(err)
            raise Exception(err)

    def get_instance_explanation(self, instance_ind, num_features=10):
        if hasattr(self.model, 'predict_proba'):
            predict_f = self.model.predict_proba
        elif hasattr(self.model, 'predict'):
            predict_f = self.model.predict
        else:
            raise AttributeError('Does not support model prediction function')

        exp = self.explainer.explain_instance(self.x_train.iloc[instance_ind], predict_f,
                                                           num_features=num_features)
        return exp

    def lime_explain_instance(self, instance_ind, num_features=10):

        exp = self.get_instance_explanation(instance_ind, num_features)
        return exp.show_in_notebook(show_table=True, show_all=all)

    def get_explanation_as_df(self, instance_ind=None, instance_interval=None):
        col_len = len(self.x_train.columns)
        df_exp = pd.DataFrame(columns=range(col_len))
        start = end = 0
        if instance_ind is not None:
            start = end = instance_ind

        if isinstance(instance_interval, tuple) or isinstance(instance_interval, list):
            start = instance_interval[0]
            end = instance_interval[1]

        for ind in range(start, end+1):
            exp = self.get_instance_explanation(ind)
            exp_map = exp.as_map()[1]
            col_explanation = {k: v for k, v in exp_map}
            explanations = {}
            for i in range(col_len):
                if i in col_explanation:
                    explanations[i] = col_explanation[i]
                else:
                    explanations[i] = 0

            df_exp = pd.concat([df_exp, pd.DataFrame([explanations])])

        df_exp.columns = self.x_train.columns
        df_exp.reset_index(drop=True, inplace=True)
        return df_exp
