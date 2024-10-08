import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

"""# keep the input & output labels unchanged """
waitingTraffic  = ctrl.Antecedent(np.arange(1, 100, 1), 'waiting')
incomingTraffic = ctrl.Antecedent(np.arange(0, 100, 1), 'incoming')
waitingDuration = ctrl.Consequent(np.arange(0, 120, 1), 'wait duration')
openDuration    = ctrl.Consequent(np.arange(0, 120, 1), 'open duration')




# You can change the design of your FIS components
fuzzySetNamesTraffic= ['light', 'average', 'heavy']
waitingTraffic.automf(names=fuzzySetNamesTraffic)
incomingTraffic.automf(names=fuzzySetNamesTraffic)
waitingTraffic.view()
incomingTraffic.view()

fuzzySetNamesDuration = ['short', 'medium', 'long']
waitingDuration.automf(names=fuzzySetNamesDuration)
openDuration.automf(names=fuzzySetNamesDuration)
waitingDuration.view()
openDuration.view()

rule1 = ctrl.Rule(incomingTraffic['light'] & waitingTraffic['light'], [waitingDuration['short'], openDuration['short']])
rule2 = ctrl.Rule(incomingTraffic['light'] & waitingTraffic['average'], [waitingDuration['short'], openDuration['medium']])
rule3 = ctrl.Rule(incomingTraffic['light'] & waitingTraffic['heavy'], [waitingDuration['short'], openDuration['long']])

rule4 = ctrl.Rule(incomingTraffic['average'] & waitingTraffic['light'], [waitingDuration['medium'], openDuration['short']])
rule5 = ctrl.Rule(incomingTraffic['average'] & waitingTraffic['average'], [waitingDuration['medium'], openDuration['medium']])
rule6 = ctrl.Rule(incomingTraffic['average'] & waitingTraffic['heavy'], [waitingDuration['medium'], openDuration['long']])

rule7 = ctrl.Rule(incomingTraffic['heavy'] & waitingTraffic['light'], [waitingDuration['long'], openDuration['short']])
rule8 = ctrl.Rule(incomingTraffic['heavy'] & waitingTraffic['average'], [waitingDuration['long'], openDuration['medium']])
rule9 = ctrl.Rule(incomingTraffic['heavy'] & waitingTraffic['heavy'], [waitingDuration['long'], openDuration['long']])

waitingDuration.defuzzify_method = 'centroid'
openDuration.defuzzify_method = 'centroid'

fis = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
fis_simulator = ctrl.ControlSystemSimulation(fis)





"""# This is how yo can test your FIS"""

import trafficSimulator
num_cars_on_main, num_cars_on_side, wait_times_main, wait_times_side = trafficSimulator.simulate(fis_simulator, verbose = False)

print("Mean waiting time - main street", np.mean(wait_times_main))
print("Mean waiting time - side street", np.mean(wait_times_side))
plt.plot(num_cars_on_main)
plt.plot(num_cars_on_side)
plt.legend(["Main", "Side"], loc="lower right")
plt.title("# cars waiting to cross over time")
plt.show()