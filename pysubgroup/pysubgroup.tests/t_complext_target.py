import pysubgroup.pysubgroup as ps
import pandas as pd
import numpy as np

data = pd.read_csv("../../data/adult.csv", index_col=0)
data = data[:100]
# print(data.head())
target = ps.ComplexTarget(('hours-per-week', 'fnlwgt'))
search_space = ps.create_nominal_selectors(data, ignore=['fnlwgt', 'hours-per-week'])
selector = ps.NominalSelector('sex', ' Male')
subgroup = ps.Subgroup(target, selector)
res = ps.CorrelationQF().evaluate_from_dataset(data, subgroup)
# print('type: ', type(res))
# print(res)
task = ps.SubgroupDiscoveryTask(data, target, search_space, qf=ps.CorrelationQF())
result = ps.BeamSearch().execute(task)
result = ps.overlap_filter(result, data, similarity_level=0.7)

for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))