from Environment import Evaluator
from configuration import configuration


evaluation = Evaluator(configuration)
evaluation.start()
evaluation.plotResults(0)
