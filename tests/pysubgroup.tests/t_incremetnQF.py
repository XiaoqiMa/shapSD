import shapSD as spd
import pandas as pd

adult = pd.read_csv('../../notebooks/adult_age_test.csv', sep='\t', index_col=0)
new_adult = adult[:4000]
effect = 'age_prediction_change'

target = spd.NumericTarget(effect)
search_space = spd.create_selectors(new_adult, ignore=[effect, 'income'])
task = spd.SubgroupDiscoveryTask(new_adult, target, search_space, qf=spd.IncrementalQFNumeric(1), min_quality=0)
result = spd.BeamSearch().execute(task)
for (q, sg) in result:
    print(str(q) + ":\t" + str(sg.subgroup_description))
