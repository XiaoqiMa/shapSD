import pysubgroup.pysubgroup as ps
import pandas as pd
# import numpy as np
# # np.warnings.filterwarnings('ignore')
#
data = pd.read_csv("../../data/adult.csv", index_col=0)
data = data[:(len(data)//2)]
target = ps.ComplexTarget(('age', 'education-num'))
search_space = ps.create_nominal_selectors(data, ignore=['age', 'education-num'])
# selector = ps.NominalSelector('sex', ' Male')
# subgroup = ps.Subgroup(target, selector)
# res = ps.CorrelationQF().evaluate_from_dataset(data, subgroup)
# print('type: ', type(res))
# print(res)
task = ps.SubgroupDiscoveryTask(data, target, search_space, qf=ps.CorrelationQF('entropy'))
# task = ps.SubgroupDiscoveryTask(data, target, search_space, qf=ps.CorrelationQF())
result = ps.BeamSearch().execute(task)
result = ps.overlap_filter(result, data, similarity_level=0.7)
#
for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))

df = ps.results_as_df(data, result, statistics_to_show=ps.complex_statistics, complex_target=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)
