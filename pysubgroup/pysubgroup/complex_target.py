import numpy as np
import pysubgroup.pysubgroup as ps
from functools import total_ordering
from scipy.stats import linregress
from scipy.stats import spearmanr

@total_ordering
class ComplexTarget(object):
    def __init__(self, target_variable_set):
        if not isinstance(target_variable_set, tuple):
            raise TypeError('Target should be a set of two numeric variables')

        self.target_variable_set = target_variable_set

    def __repr__(self):
        return "T: " + str(self.target_variable_set)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return str(self) < str(other)

    def get_attributes(self):
        return [var for var in self.target_variable_set]

    def get_corr_statistics(self, data, subgroup):

        sg_instances = subgroup.subgroup_description.covers(data)

        all_target_values_x1 = data[self.target_variable_set[0]]
        all_target_values_x2 = data[self.target_variable_set[1]]
        sg_target_values_x1 = all_target_values_x1[sg_instances]
        sg_target_values_x2 = all_target_values_x2[sg_instances]
        sg_complement_x1 = all_target_values_x1[~all_target_values_x1.index.isin(sg_instances)]
        sg_complement_x2 = all_target_values_x2[~all_target_values_x2.index.isin(sg_instances)]

        instances_dataset_len = len(data)
        instances_sg_len = np.sum(sg_instances)
        complement_sg_len = instances_dataset_len - instances_sg_len
        try:
            # sg_corr = np.corrcoef(sg_target_values_x1, sg_target_values_x2)[0, 1]
            # dataset_corr = np.corrcoef(all_target_values_x1, all_target_values_x2)[0, 1]
            # sg_complement_corr = np.corrcoef(sg_complement_x1, sg_complement_x2)[0, 1]

            sg_corr = spearmanr(sg_target_values_x1, sg_target_values_x2).correlation
            dataset_corr = spearmanr(all_target_values_x1, all_target_values_x2).correlation
            sg_complement_corr = spearmanr(sg_complement_x1, sg_complement_x2).correlation
        except Exception as err:
            sg_corr = dataset_corr = sg_complement_corr = 0.5
            print(err)
            # raise Exception('Cannot get correlation statistics')

        # sg_slope = linregress(sg_target_values_x1, sg_target_values_x2).slope
        # dataset_slope = linregress(all_target_values_x1, all_target_values_x2).slope
        # sg_complement_slope = linregress(sg_complement_x1, sg_complement_x2).slope

        return (instances_dataset_len, dataset_corr,
                instances_sg_len, sg_corr,
                complement_sg_len, sg_complement_corr)


    def calculate_corr_statistics(self, data, subgroup):

        args = self.get_corr_statistics(data, subgroup)
        instances_dataset_len = args[0]
        dataset_corr = args[1]
        instances_sg_len = args[2]
        sg_corr = args[3]
        complement_sg_len = args[4]
        sg_complement_corr = args[5]

        subgroup.statistics['sg_size'] = instances_sg_len
        subgroup.statistics['dataset_size'] = instances_dataset_len
        subgroup.statistics['complement_sg_size'] = complement_sg_len
        subgroup.statistics['sg_corr'] = sg_corr
        subgroup.statistics['dataset_corr'] = dataset_corr
        subgroup.statistics['complement_sg_corr'] = sg_complement_corr
        # subgroup.statistics['sg_slope'] = sg_slope
        # subgroup.statistics['dataset_slope'] = dataset_slope
        # subgroup.statistics['complement_sg_slope'] = sg_complement_slope
        subgroup.statistics['corr_lift'] = subgroup.statistics['sg_corr'] / \
                                           subgroup.statistics['dataset_corr']
        # print('sub sta: ', subgroup.statistics)
        # print('type sta: ', type(subgroup.statistics))
        return subgroup.statistics


class CorrelationQF(ps.CorrelationModelMeasure):

    def __init__(self, measure='abs_diff'):
        """
        correlation model quality measurement
        :param measure:
        'abs_diff': absolute difference of corr between sg and it's complement subgroup
        'entropy': considering the entropy of two subgroups H(p)*abs_diff
        'significance_test': quality defined by 1-p_value
        """
        self.measure = measure

    def corr_qf(self, **statistics):
        complement_sg_size = statistics['complement_sg_size'],
        complement_sg_corr = statistics['complement_sg_corr']
        sg_size = statistics['sg_size']
        sg_corr = statistics['sg_corr']

        if sg_size == 0:
            return 0
        if self.measure == 'abs_diff':
            return np.abs(sg_corr - complement_sg_corr)
        elif self.measure == 'entropy':
            n = sg_size / (sg_size + complement_sg_size)
            n_bar = 1 - n
            entropy = -(np.log2(n) * n + np.log2(n_bar) * n_bar)
            return entropy * np.abs(sg_corr - complement_sg_corr)
        elif self.measure == 'significance_test':
            pass
        else:
            raise BaseException("measurement is constrained to "
                                "['abs_diff', 'entropy', 'significance_test']")

    def evaluate_from_dataset(self, data, subgroup):
        if not self.is_applicable(subgroup):
            raise BaseException('Correlation model Quality measure can not be used')
        statistics = subgroup.calculate_corr_statistics(data)
        # print(type(subgroup.calculate_corr_statistics(data)))
        return self.evaluate_from_statistics(**statistics)

    def evaluate_from_statistics(self, **statistics):
        return self.corr_qf(**statistics)

    def is_applicable(self, subgroup):
        return isinstance(subgroup.target, ComplexTarget)
