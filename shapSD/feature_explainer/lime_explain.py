from lime import lime_tabular
from lime import lime_text
from lime import lime_image
from .logging_custom import *


class LimeExplain(object):

    def __init__(self, x_train, model, explainer_type='tabular'):
        self.x_train = x_train
        self.model = model
        self.explainer_type = explainer_type

        if self.explainer_type == 'tabular':
            self.explainer = lime_tabular.LimeTabularExplainer
        elif self.explainer_type == 'text':
            self.explainer = lime_text.LimeTextExplainer
        elif self.explainer_type == 'image':
            self.explainer = lime_image.LimeImageExplainer
        else:
            raise TypeError('Does not support {} explainer'.format(self.explainer_type))

        self.lime_explainer = self.get_lime_explainer()

    def get_lime_explainer(self):
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
                lime_explainer = self.explainer(self.x_train.values,
                                                feature_names=self.x_train.columns,
                                                categorical_features=cat_features,
                                                categorical_names=label_dic,
                                                discretize_continuous=True)

                return lime_explainer
            else:
                lime_explainer = self.explainer(self.x_train.values,
                                                feature_names=self.x_train.columns,
                                                discretize_continuous=True)
                return lime_explainer
        except Exception as err:
            print('Error: model is not supported by LIME {} Explainer'.format(self.explainer_type))
            err_logging(err)
            raise Exception(err)

    def lime_explain_instance(self, instance_ind, num_features=10):

        if hasattr(self.model, 'predict_proba'):
            predict_f = self.model.predict_proba
        elif hasattr(self.model, 'predict'):
            predict_f = self.model.predict
        else:
            raise AttributeError('Does not support model prediction function')

        exp = self.lime_explainer.explain_instance(self.x_train.iloc[instance_ind], predict_f,
                                                   num_features=num_features)
        return exp.show_in_notebook(show_table=True, show_all=all)
