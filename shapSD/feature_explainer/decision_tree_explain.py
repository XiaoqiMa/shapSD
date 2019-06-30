import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from graphviz import Source
from IPython.display import SVG
from IPython.display import display
from subprocess import check_output


class DecisionTreeExplain(object):

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.PROJECT_ROOT_DIR = "."
        self.FILE_FOLDER = 'files'
        file_dir = os.path.join(self.PROJECT_ROOT_DIR, self.FILE_FOLDER)
        os.makedirs(file_dir, exist_ok=True)

    def dtree_clf_model(self, criterion='gini', max_depth=5, min_samples_leaf=50, min_samples_split=100, **kwargs):
        estimator = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                           min_samples_split=min_samples_split, **kwargs)
        return estimator.fit(self.x_train, self.y_train)

    def dtree_reg_model(self, criterion='mse', max_depth=5, min_samples_leaf=50, min_samples_split=100, **kwargs):
        estimator = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          min_samples_split=min_samples_split, **kwargs)
        return estimator.fit(self.x_train, self.y_train)

    def rf_clf_model(self, criterion='gini', max_depth=5, min_samples_leaf=50, min_samples_split=100, **kwargs):
        estimator = RandomForestClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                           min_samples_split=min_samples_split, **kwargs)
        estimator.fit(self.x_train, self.y_train)
        return estimator.estimators_[0]

    def rf_reg_model(self, criterion='mse', max_depth=5, min_samples_leaf=50, min_samples_split=100, **kwargs):
        estimator = RandomForestRegressor(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          min_samples_split=min_samples_split, **kwargs)
        estimator.fit(self.x_train, self.y_train)
        return estimator.estimators_[0]

    def visualize_dtree(self, estimator, file_id):
        dot_file = './files/{}.dot'.format(file_id)
        pdf_file = './files/{}.pdf'.format(file_id)
        try:
            graph = Source(export_graphviz(estimator, out_file=None,
                                           feature_names=list(self.x_train.columns),
                                           class_names=list(map(str, list(self.y_train.unique()))), filled=True))
            display(SVG(graph.pipe(format='svg')))
            with open(dot_file, "w") as f:
                export_graphviz(estimator, out_file=f,
                                feature_names=list(self.x_train.columns),
                                class_names=list(map(str, list(self.y_train.unique()))),
                                filled=True, rounded=True)
            check_output("dot -Tpdf " + dot_file + " -o " + pdf_file, shell=True)
        except Exception as err:
            print('Cannot visualize decision trees')
            print(err)
