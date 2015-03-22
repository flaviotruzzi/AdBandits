# -*- coding: utf-8 -*-
from Arms.Bernoulli import Bernoulli
from Policies import UCB
from Policies import Thompson
from Policies import klUCB
from Policies import AdBandit

configuration = {
    "horizon": 10000,
    "repetitions": 100,
    "n_jobs": 4,
    "verbosity": 5,
    "environment": [
        {
            "arm_type": Bernoulli,
            "probabilities": [0.02, 0.02, 0.02, 0.10, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01]
        }
    ],
    "policies": [
        {
            "archtype": UCB,
            "params": {}
        },
        {
            "archtype": Thompson,
            "params": {}
        },
        {
            "archtype": klUCB,
            "params": {}
        },
        {
            "archtype": AdBandit,
            "params": {
                "alpha": 0.5,
                "horizon": 10000
            }
        }
    ]
}
