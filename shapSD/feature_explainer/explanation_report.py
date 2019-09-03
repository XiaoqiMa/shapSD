"""
Generate explanation report automatically
author: Xiaoqi
date: 2019.09.03
"""
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer, PageBreak, TableStyle, Table
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors

from .feature_importance import FeatureImportance
from .shap_explainer import ShapExplainer
import shapSD.pysubgroup as ssd


class ExplanationReport(object):

    def __init__(self, ori_dataset, x_train, y_train, model, var_name, dataset_name):
        self.ori_dataset = ori_dataset
        self.x_train = x_train
        self.y_train = y_train
        self.target_name = self.y_train.columns.tolist()[0]
        self.model = model
        self.var_name = var_name
        self.dataset_name = dataset_name
        self.features = []
        self.df_effect = None
        self.df_sg_result = None
        self.attr_name = None

    def get_feature_imp_plot(self):
        # construct FeamtureImportance class, to visualize feature importance
        label_feature_imp = FeatureImportance(self.x_train, self.y_train, self.model)
        label_imp = label_feature_imp.permutation_importance()
        self.features = label_imp['Features'].tolist()
        feature_imp_path = label_feature_imp.vis_perm_importance(label_imp)
        return feature_imp_path

    def get_shap_plt(self):
        tree_shap = ShapExplainer(self.x_train, self.model, explainer_type='Tree')
        summary_plt_path = tree_shap.shap_summary_plot()
        dependence_plt_path = tree_shap.shap_dependence_plot(ind=self.var_name, interaction_index=self.var_name)
        return summary_plt_path, dependence_plt_path

    def get_shap_values(self):
        tree_shap = ShapExplainer(self.x_train, self.model, explainer_type='Tree')
        exp, shap_v, expected_v = tree_shap.calc_shap_values(attr=self.var_name)
        self.df_effect = self.ori_dataset.drop(self.target_name, axis=1)
        self.attr_name = '{}_shap_values'.format(self.var_name)
        self.df_effect[self.attr_name] = shap_v
        effect = self.df_effect[self.features[:5]]
        effect[self.attr_name] = shap_v
        return effect

    def get_sg_result(self):
        target = ssd.NumericTarget(self.attr_name)
        search_space = ssd.create_selectors(self.df_effect, ignore=[self.target_name, self.var_name, self.attr_name])
        task = ssd.SubgroupDiscoveryTask(self.df_effect, target, search_space, qf=ssd.StandardQFNumeric(1),
                                         result_set_size=10)
        result = ssd.BeamSearch().execute(task)
        # result = ssd.overlap_filter(result, df_effect, similarity_level=0.85)
        self.df_sg_result = ssd.as_df(self.df_effect, result, statistics_to_show=ssd.all_statistics_numeric)
        self.df_sg_result = self.df_sg_result[['quality', 'subgroup']]
        return self.df_sg_result

    def generate_report(self):

        report_name = '{}_explanation_report.pdf'.format(self.var_name)
        doc = SimpleDocTemplate(report_name, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

        report = []
        title = '<b><font size=12>Inspect influence of variable "{}" \
                            in black box models in "{}" dataset</font></b>'.format(self.var_name, self.dataset_name)
        report.append(Paragraph(title, styles["Justify"]))
        report.append(Spacer(1, 20))

        txt_feature_imp = '<font size=12>Feature Importance ranking from global interpretation perspective</font>'
        report.append(Paragraph(txt_feature_imp, styles["Justify"]))
        report.append(Spacer(1, 20))

        # feature importance ranking image
        report.append(Image(self.get_feature_imp_plot(), 5 * inch, 4 * inch))
        report.append(Spacer(1, 20))

        txt_features = 'Top 5 important features are: <b>{0}, {1}, {2}, {3}, {4}</b>'.format(
            self.features[0], self.features[1], self.features[2], self.features[3], self.features[4])
        report.append(Paragraph(txt_features, styles["Justify"]))

        report.append(PageBreak())

        txt_summary = 'To get an overview of which <b>most important features </b> in a model, we \
                        can plot the SHAP values of every feature for every sample. The plot above \
                        sorts features by the sum of SHAP (absolute) value magnitudes over all samples, \
                        and uses SHAP values to show the distribution of the impacts each feature has \
                        on the model output. The color represents the feature value (red high, blue low)'
        report.append(Paragraph(txt_summary, styles["Normal"]))
        report.append(Spacer(1, 20))

        summary_plt_path, dependence_plt_path = self.get_shap_plt()
        report.append(Image(summary_plt_path, 3 * inch, 3 * inch))
        report.append(Spacer(1, 20))

        txt_dependence = "To understand how a single feature affects the output of the model we can plot \
                            the SHAP value of that feature vs. the value of the feature for all the examples \
                            in a dataset. Since SHAP values represent a feature's responsibility for a change \
                            in the model output, the plot below represents the change in predicted income probability \
                            as <b>{}</b> changes".format(self.var_name)
        report.append(Paragraph(txt_dependence, styles["Normal"]))
        report.append(Spacer(1, 20))

        report.append(Image(dependence_plt_path, 3 * inch, 3 * inch))
        report.append(Spacer(1, 20))

        report.append(PageBreak())
        txt_sg = 'We take the SHAP value of <b>"{}"</b> to represent the contribution of this feature to the prediction \
                    in an individual instance. Further, it is set up as our target concept in subgroup discovery \
                    Results are shown as follows'.format(self.var_name)
        report.append(Paragraph(txt_sg, styles["Normal"]))
        report.append(Spacer(1, 20))

        grid_style = TableStyle(
            [('GRID', (0, 0), (-1, -1), 0.1, colors.black),
             ('ALIGN', (1, 0), (-1, -1), 'RIGHT')])

        tab_df = []
        effect= self.get_shap_values()
        tab_df.append(effect.columns.tolist())
        tab_df.extend(np.array(effect.head()).tolist())
        tab_df = Table(tab_df, style=grid_style)
        report.append(tab_df)
        report.append(Spacer(1, 20))

        txt_int_sg = '<b>Interesting subgroups are listed as follows</b>'
        report.append(Paragraph(txt_int_sg, styles['Normal']))
        report.append(Spacer(1, 20))

        sg_df = []
        df_result = self.get_sg_result()
        sg_df.append(df_result.columns.tolist())
        sg_df.extend(np.array(df_result.head()).tolist())
        sg_df = Table(sg_df, style=grid_style)
        report.append(sg_df)
        doc.build(report)
