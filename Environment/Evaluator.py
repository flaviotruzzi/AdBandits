import numpy as np
from .MAB import MAB
from joblib import *
import copy as cp
from .Result import Result
import matplotlib.pyplot as plt

class Evaluator:
    """docstring for Evaluator"""

    def __init__(self, configuration):
        self.cfg = configuration
        self.envs = []
        self.policies = []
        self.__initEnvironments__()
        print (len(self.cfg['policies']), len(self.envs), self.cfg['horizon'])
        self.rewards = np.zeros((len(self.cfg['policies']),
                                 len(self.envs), self.cfg['horizon']))
        self.pulls = {}
        for env in xrange(len(self.envs)):
            self.pulls[env] = np.zeros((len(self.cfg['policies']), self.envs[env].nbArms))

    def __initEnvironments__(self):
        for armType in self.cfg['environment']:
            self.envs.append(MAB(armType))

    def __initPolicies__(self, env):
        for policy in self.cfg['policies']:
            self.policies.append(policy['archtype'](env.nbArms,
                                                    **policy['params']))

    def start(self):
        for envId, env in enumerate(self.envs):
            print "Evaluating environment: " + repr(env)
            self.policies = []
            self.__initPolicies__(env)
            for polId, policy in enumerate(self.policies):
                print "+Evaluating: " + policy.__class__.__name__ + ' (' + \
                      policy.params + ") ..."
                results = Parallel(n_jobs=self.cfg['n_jobs'], verbose=self.cfg['verbosity'])(
                    delayed(play)(env, policy, self.cfg['horizon'])
                    for _ in xrange(self.cfg['repetitions']))
                for result in results:
                    self.rewards[polId, envId, :] += np.cumsum(result.rewards)
                    self.pulls[envId][polId, :] += result.pulls

    def getReward(self, policyId, environmentId):
        return self.rewards[policyId, environmentId, :] / self.cfg['repetitions']

    def getRegret(self, policyId, environmentId):
        horizon = np.arange(self.cfg['horizon'])
        return horizon * self.envs[environmentId].maxArm - self.getReward(policyId, environmentId)

    def plotResults(self, environment):
        figure = plt.figure()
        for i, policy in enumerate(self.policies):
            plt.plot(self.getRegret(i, environment), label=str(policy))
        plt.show()

def play(env, policy, horizon):
    env = cp.deepcopy(env)
    policy = cp.deepcopy(policy)
    horizon = cp.deepcopy(horizon)

    policy.startGame()
    result = Result(env.nbArms, horizon)
    for t in xrange(horizon):
        choice = policy.choice()
        reward = env.arms[choice].draw(t)
        policy.getReward(choice, reward)
        result.store(t, choice, reward)
    return result
