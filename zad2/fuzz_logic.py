import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


ph = ctrl.Antecedent(np.arange(0, 15, 1), 'ph')
temp = ctrl.Antecedent(np.arange(0, 31, 1), 'temp')
humidity = ctrl.Antecedent(np.arange(0, 11, 1), 'hum')
growth = ctrl.Consequent(np.arange(0, 8, 1), 'growth')

ph['alkaline'] = fuzz.trimf(ph.universe, [0, 0, 6])
ph['average'] = fuzz.trimf(ph.universe, [6, 7, 8])
ph['acidic'] = fuzz.trimf(ph.universe, [8, 14, 14])

temp['low'] = fuzz.trimf(temp.universe, [0, 0, 17])
temp['medium'] = fuzz.trimf(temp.universe, [17, 23, 25])
temp['high'] = fuzz.trimf(temp.universe, [25, 35, 35])

humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 5])
humidity['medium'] = fuzz.trimf(humidity.universe, [5, 6, 7])
humidity['high'] = fuzz.trimf(humidity.universe, [7, 10, 10])

growth['fast'] = fuzz.trimf(growth.universe, [0, 0, 3])
growth['medium'] = fuzz.trimf(growth.universe, [3, 5, 6])
growth['long'] = fuzz.trimf(growth.universe, [6, 8, 8])

humidity.view()
temp.view()
growth.view()
ph.view()

rule1 = ctrl.Rule(ph['acidic'] | temp['low'] | humidity['low'] | ph['alkaline'], growth['long'])
rule2 = ctrl.Rule(ph['average'] & temp['high'] & humidity['medium'], growth['medium'])
rule3 = ctrl.Rule(ph['average'] & temp['high'] & humidity['high'], growth['fast'])
rule4 = ctrl.Rule(ph['average'] & temp['medium'] & humidity['medium'], growth['fast'])

result_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
result = ctrl.ControlSystemSimulation(result_ctrl)

result.input['hum'] = 6
result.input['ph'] = 7
result.input['temp'] = 20

result.compute()

print(result.output['growth'])
growth.view(sim=result)








