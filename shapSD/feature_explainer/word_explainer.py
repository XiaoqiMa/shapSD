"""
show word influence in a sentence; using LIME and SHAP visualizations
author: Xiaoqi
date: 2019.09.12
"""

import re
import pandas as pd
from collections import defaultdict
from collections import Counter
from .shap_explainer import ShapExplainer


class WordExplainer(object):

    def __init__(self, x_train, x_train_transform, model):
        self.x_train = pd.DataFrame(x_train)
        if 'text' not in self.x_train.columns or 'category' not in self.x_train.columns:
            raise KeyError('Input must include "text" and "category" columns')
        self.x_train_transform = pd.DataFrame(x_train_transform)
        self.model = model
        self.index_to_cat, self.common_cat = self.get_most_common_cat()
        self.index_to_cat_vec = self.get_cat_vectors()
        self.shap_values = self.get_shap_values()

    def clean_cat(self, cat_str):
        pattern = r'[\{\}\'\'\"\"]'
        cat_str = re.sub(pattern, '', cat_str)
        cat_list = cat_str.split(',')
        cat_list = [i.strip() for i in cat_list]
        return cat_list

    def get_most_common_cat(self):
        index_to_cat = defaultdict()
        all_cat = []
        for i in self.x_train.index:
            cleand_cat_list = self.clean_cat(self.x_train["category"][i])
            index_to_cat[i] = cleand_cat_list
            all_cat += cleand_cat_list

        common_cat = [i[0] for i in Counter(all_cat).most_common(50)]
        common_cat = sorted(common_cat)
        return index_to_cat, common_cat

    def get_cat_vectors(self):
        index_to_cat_vec = defaultdict()
        for i in self.index_to_cat.keys():
            cat_list = self.index_to_cat[i]
            cat_vec = [0] * len(self.common_cat)
            for ind, cat in enumerate(self.common_cat):
                if cat in cat_list:
                    cat_vec[ind] = 1
                else:
                    cat_vec[ind] = 0

            index_to_cat_vec[i] = cat_vec

        return index_to_cat_vec

    def get_instance_index(self, word):
        instance_ind = []
        for i in self.x_train.index:
            if word in self.x_train['text'][i]:
                instance_ind.append(i)

        return instance_ind

    def get_shap_values(self):
        exp_tree = ShapExplainer(self.x_train_transform, self.model)
        shap_values = exp_tree.get_shap_values_as_df(instance_interval=(0, len(self.x_train)))
        return shap_values

    def word_effect_summary(self):
        effect = pd.DataFrame([self.shap_values.abs().sum()])
        effect = effect.transpose()
        effect.columns = ['abs_shap_value']
        effect = effect.sort_values('abs_shap_value', ascending=False).head(20)
        return effect.plot.bar()

    def get_word_effect(self, word):

        instance_ind = self.get_instance_index(word)
        shap_value = self.shap_values[word]

        word_shap_value = shap_value[instance_ind]
        instance_vectors = []
        for ind in instance_ind:
            instance_vectors.append(self.index_to_cat_vec[ind])

        df_word_effect = pd.DataFrame(instance_vectors, columns=self.common_cat, index=instance_ind)
        df_word_effect['{}_shap_value'.format(word)] = word_shap_value.tolist()
        return df_word_effect
