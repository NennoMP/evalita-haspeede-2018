import random

import optuna
import numpy as np

from itertools import product
from math import log10

from training.callbacks import CustomPrintCallback, CustomBestEarlyStopping
from training.solver import Solver



ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


def sample_hparams_spaces(search_spaces, trial=None):
    """"
    Takes search spaces for random search or bayesian optimization as input; samples 
    accordingly from these spaces and returns the sampled hyper-params as a config-object,
    which will be used to construct solver and/or model.
    
    Args:
    - search_spaces: hparams intervals for tuning
    
    Optional:
    - trial: if provided, use Bayesian suggestions, otherwise sample randomly
    """
    
    config = {}
    for key, (values, mode) in search_spaces.items():
        if mode == "float":
            config[key] = (
                trial.suggest_float(key, values[0], values[1]) if trial
                else random.uniform(values[0], values[1])
            )
        elif mode == "int":
            config[key] = (
                trial.suggest_int(key, values[0], values[1]) if trial
                else np.random.randint(values[0], values[1])
            )
        elif mode == "item":
            config[key] = (
                trial.suggest_categorical(key, values) if trial
                else np.random.choice(values)
            )
        elif mode == "log":
            if trial:
                config[key] = trial.suggest_float(key, values[0], values[1], log=True)
            else:
                log_min, log_max = np.log(values)
                config[key] = np.exp(np.random.uniform(log_min, log_max))

    return config


######################################
# BAYESIAN OPTIMIZATION
######################################
def bayesian_optimization(model_fn, train_data, train_labels, val_data, val_labels, 
                          bayesian_optimization_spaces, TARGET='val_avg_f1', N_TRIALS=50, EPOCHS=30, PATIENCE=5):
    
    def objective(trial):
        config = sample_hparams_spaces(bayesian_optimization_spaces, trial)
        model = model_fn(config)
        
        print("\nEvaluating Config #{} [of {}]:\n".format((trial.number), N_TRIALS-1), config)
        history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), batch_size=config['batch_size'], epochs=EPOCHS, verbose=1)
        return max(history.history[TARGET])
    
    if TARGET == 'val_avg_f1':
        direction = 'maximize'
    else:
        raise ValueError(f"Unsupported TARGET value: {TARGET}. Try 'val_loss' or 'val_avg_f1!'")
    
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=N_TRIALS)
    
    print(f"\nSearch done. Best {TARGET} = {study.best_value}")
    print("Best Config:", study.best_params)
    return study.best_value, study.best_params


######################################
# RANDOM SEARCH
######################################
def random_search(model_fn, train_data, train_labels, val_data, val_labels, 
                  random_search_spaces, TARGET='val_loss', NUM_SEARCH=20, EPOCHS=30, PATIENCE=5):
    """
    Samples NUM_SEARCH hyper parameter sets within the provided search spaces
    and returns the best model.
    
    Required arguments:
        - model_fn: a function returning a model object

        - train_data: training data
        - train_labels: training gtc labels
        
        - val_data: validation data
        - val_labels: validation gtc labels
            
        - random_search_spaces: a dictionary where every key corresponds to a
            to-tune-hyperparameter and every value contains an interval or a list of possible
            values to test.
    
    Optional arguments:
        - TARGET: target metric for optimization. Allowed choices are 'val_loss' or 'val_avg_f1'
        - NUM_SEARCH: number of configurations to test
        - EPOCHS: number of epochs each model will be trained on
        - PATIENCE: when to stop early the training process
    """
    
    configs = []
    for _ in range(NUM_SEARCH):
        configs.append(sample_hparams_spaces(random_search_spaces))

    return findBestConfig(model_fn, train_data, train_labels, val_data, val_labels, configs, TARGET, EPOCHS, PATIENCE)


def findBestConfig(model_fn, train_data, train_labels, val_data, val_labels, configs, TARGET, EPOCHS, PATIENCE):
    """
    Get a list of hyperparameter configs for random search, trains a model on all configs 
    and returns the one performing best on validation set according to specific TARGET metric.
    """

    if TARGET == 'val_avg_f1':
        best_target = float('-inf')
    else:
        raise ValueError(f"Unsupported TARGET value: {TARGET}. Try 'val_loss' or 'val_avg_f1!'")
    
    best_config = None
    best_model = None
    results = []

    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format((i+1), len(configs)), configs[i])

        model = model_fn(configs[i])
        solver = Solver(model, train_data, train_labels, val_data, val_labels, TARGET, **configs[i])
        solver.train(epochs=EPOCHS, patience=PATIENCE, batch_size=configs[i]['batch_size'])
        results.append(solver.best_model_stats)

        if solver.best_model_stats[TARGET] > best_target:
            best_target, best_model, best_config = solver.best_model_stats[TARGET], model, configs[i]

    print(f"\nSearch done. Best {TARGET} = {best_target}")
    print("Best Config:", best_config)
    return best_model, best_config, list(zip(configs, results))