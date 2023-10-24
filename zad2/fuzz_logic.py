import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

"""

Bean growth infos: https://www.pphu-ogrodnik.pl/content/17-jak-dlugo-trwa-proces-wzrostu-nasion-fasoli

Example of Fuzzy Logic for determinning the time of bean growth depending on temperature, soil ph and humidity.

Authors:
    Oliwier Kossak s22018
    Daniel Klimowski S18504
    
Preparing the environment:
    to install numpy use command: "pip install numpy"    
    to install skfuzzy use command: "pip install skfuzzy"    

"""

"""
Initialization of antecedents with labels and variable scope - arguments of growth function:
soil ph (0 - 14), air temperature (0 - 30) and humidity (0 - 10)
"""
ph = ctrl.Antecedent(np.arange(0, 15, 1), 'ph')
temp = ctrl.Antecedent(np.arange(0, 31, 1), 'temp')
humidity = ctrl.Antecedent(np.arange(0, 11, 1), 'hum')

"""
Initialization of consequet - result argument of bean growth (0 - 8 weeks)
"""
growth = ctrl.Consequent(np.arange(0, 8, 1), 'growth')

"""
initialization of arguments assesment as labels 
"""
ph['alkaline'] = fuzz.trimf(ph.universe, [0, 0, 7])
ph['average'] = fuzz.trimf(ph.universe, [6, 7, 8])
ph['acidic'] = fuzz.trimf(ph.universe, [7, 14, 14])

temp['low'] = fuzz.trimf(temp.universe, [0, 0, 20])
temp['medium'] = fuzz.trimf(temp.universe, [17, 23, 26])
temp['high'] = fuzz.trimf(temp.universe, [24, 35, 35])

humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 6])
humidity['medium'] = fuzz.trimf(humidity.universe, [4, 6, 8])
humidity['high'] = fuzz.trimf(humidity.universe, [7, 10, 10])

growth['fast'] = fuzz.trimf(growth.universe, [0, 0, 3])
growth['medium'] = fuzz.trimf(growth.universe, [2, 5, 6])
growth['long'] = fuzz.trimf(growth.universe, [4, 8, 8])

"""
Print graph for each argument scale and assesment 
"""
humidity.view()
temp.view()
growth.view()
ph.view()

"""
Fuzzy rules
Crucial fuzzy logic function - declaring the rules to specify relation between arguments (Antecedents to Consequences)
These rules determine the program assessment (bean growth).

Example: if: temperature is medium (17 do 25 Celsius degree), soil ph is average (6 - 8) and humidity is medium (5 - 7)
         then: bean gonna grow faster (0 to 3 weeks) 
"""
rule1 = ctrl.Rule(ph['acidic'] | temp['low'] | humidity['low'] | ph['alkaline'], growth['long'])
rule2 = ctrl.Rule(ph['average'] & temp['high'] & humidity['medium'], growth['medium'])
rule3 = ctrl.Rule(ph['average'] & temp['high'] & humidity['high'], growth['fast'])
rule4 = ctrl.Rule(ph['average'] & temp['medium'] & humidity['medium'], growth['fast'])

"""
Creation of control system due to defined rules
"""
result_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

"""
Initialization of assessment by control system
"""
result = ctrl.ControlSystemSimulation(result_ctrl)

"""
Providing the inputs for growth function
"""
result.input['hum'] = 6
result.input['ph'] = 7
result.input['temp'] = 20

"""
Calculation of result
"""
result.compute()

"""
Print result growth with center of gravity marked 
"""
print(result.output['growth'])
growth.view(sim=result)


e260e623318140a186f1d033dbd2bac4








