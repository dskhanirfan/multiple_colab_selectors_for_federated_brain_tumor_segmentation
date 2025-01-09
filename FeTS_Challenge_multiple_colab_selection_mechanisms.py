#!/usr/bin/env python
# coding: utf-8

# # FeTS Challenge
# 
# Contributing Authors (alphabetical order):
# - Brandon Edwards (Intel)
# - Patrick Foley (Intel)
# - Alexey Gruzdev (Intel)
# - Sarthak Pati (University of Pennsylvania)
# - Micah Sheller (Intel)
# - Ilya Trushkin (Intel)


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # edit according to your system's configuration


import numpy as np
import random

from fets_challenge import run_challenge_experiment
from scipy.optimize import linprog
#from multiarmbandit_reinforcement import rl_collaborator_selector, get_reward, get_state
#from custom_greedy import greedy_epsilon_rl_collaborator_selector
#from new_ucb import new_ucb_rl_collaborator_selector
#from april_ucb import ucb_rl_collaborator_selector
#from new_kl_ucb_rl import kl_ucb_rl_collaborator_selector
#from regularized_kl_ucb import regularized_kl_ucb_rl_collaborator_selector
#from fixed_kl_ucb_rl import custom_ucb_rl_collaborator_selector
#from fixed_april_my_rl import custom_ucb_rl_collaborator_selector
#from latest_ucb_frequency import frequency_ucb_rl_collaborator_selector
#from cmdstanpy import CmdStanModel
import stan
import arviz as az
import networkx as nx
import numpy as np
import pandas as pd
#from collections import Counter
import random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

#import ctypes
#libgcc_s = ctypes.CDLL('libgcc_s.so.1')
#from Qlearning_reinforcement import rl_collaborator_selector, calculate_reward
#from recommender_engine_colab import recommender_all_collaborators_train
#from my_bayesian import Bayesian_SimAgg
#from final_bayesian import Bayesian_SimAgg
# # Adding custom functionality to the experiment
# Within this notebook there are **four** functional areas that you can adjust to improve upon the challenge reference code:
# 
# - [Custom aggregation logic](#Custom-Aggregation-Functions)
# - [Selection of training hyperparameters by round](#Custom-hyperparameters-for-training)
# - [Collaborator training selection by round](#Custom-Collaborator-Training-Selection)
# 

# ## Experiment logger for your functions
# The following import allows you to use the same logger used by the experiment framework. This lets you include logging in your functions.

from fets_challenge.experiment import logger


# # Getting access to historical weights, metrics, and more
# The **db_iterator** parameter gives full access to all of the tensors and metrics stored by the aggregator. Participants can access these records to create advanced aggregation methods, hyperparameters for training, and novel selection logic for which collaborators should participant in a given training round. See below for details about how data is stored internally and a comprehensive set of examples. 
# 
# ## Basic Form
# Each record yielded by the `db_iterator` contains the following fields:
# 
# |                      TensorKey                     |   Tensor  |
# |:--------------------------------------------------:|:---------:|
# | 'tensor_name', 'origin', 'round', 'report', 'tags' | 'nparray' |
# 
# All records are internally stored as a numpy array: model weights, metrics, as well as hyperparameters. 
# 
# Detailed field explanation:
# - **'tensor_name'** (str): The `'tensor_name'` field corresponds to the model layer name (i.e. `'conv2d'`), or the name of the metric that has been reported by a collaborator (i.e. `'accuracy'`). The built-in validation functions used for evaluation of the challenge will be given a prefix of `'challenge_metric_\*'`. The names that you provide in conjunction with a custom validation metrics to the ```run_challenge_experiment``` function will remain unchanged.  
# - **'origin'** (str): The origin denotes where the numpy array came from. Possible values are any of the collaborator names (i.e. `'col1'`), or the aggregator.
# - **'round'** (int): The round that produced the tensor. If your experiment has `N` rounds, possible values are `0->N-1`
# - **'report'** (boolean): This field is one of the ways that a metric can be denoted; For the purpose of aggregation, this field can be ignored.
# - **'tags'** (tuple(str)): The tags include unstructured information that can be used to create complex data flows. For example, model layer weights will have the same `'tensor_name'` and `'round'` before and after training, so a tag of `'trained'` is used to denote that the numpy array corresponds to the layer of a locally trained model. This field is also used to capture metric information. For example, `aggregated_model_validation` assigns tags of `'metric'` and `'validate_agg'` to reflect that the metric reported corresponds to the validation score of the latest aggregated model, whereas the tags of `'metric'` and `'validate_local'` are used for metrics produced through validation after training on a collaborator's local data.   
# - **'nparray'** (numpy array) : This contains the value of the tensor. May contain the model weights, metrics, or hyperparameters as a numpy array.
# 

# ### Note about OpenFL "tensors"
# In order to be ML framework agnostic, OpenFL represents tensors as numpy arrays. Throughout this code, tensor data is represented as numpy arrays (as opposed to torch tensors, for example).

# # Custom Collaborator Training Selection
# By default, all collaborators will be selected for training each round, but you can easily add your own logic to select a different set of collaborators based on custom criteria. An example is provided below for selecting a single collaborator on odd rounds that had the fastest training time (`one_collaborator_on_odd_rounds`).

#import random
#import numpy as np
def my_custom_ucb_rl_collaborator_selector(collaborators, db_iterator, fl_round, collaborators_chosen_each_round,
                                        collaborator_times_per_round):
    """Selects collaborators for the given round using the KL-UCB algorithm.

    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}.
            Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    #print('collaborators_chosen_each_round', collaborators_chosen_each_round)
    #print('collaborator_times_per_round', collaborator_times_per_round)

    num_collaborators = len(collaborators)
    print('num_collaborators', num_collaborators)

    if fl_round > 0:
        print('custom_ucb_rl_collaborator_selector called')

        AGG_RESULTS = []
        for record in db_iterator:
            if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3 and \
                    record['tensor_name'] == 'valid_dice':
                AGG_RESULTS.append((record['tags'][0], record['nparray']))
        print('AGG_RESULTS', AGG_RESULTS)

        colab_numbers = [t[0] for t in AGG_RESULTS]
        agg_valid_dice = [t[1] for t in AGG_RESULTS]
        array = np.array(list(zip(colab_numbers, agg_valid_dice)))
        print('iftikharray -> ', array)
        total_reward = np.sum(np.array(agg_valid_dice).astype(float)) / num_collaborators  

        total_scores = []
        for i in range(num_collaborators):
            collab_id = array[i][0]
            if collab_id not in collaborator_times_per_round[fl_round - 1]:
                total_scores.append(1e10)
            else:
                estimated_reward = float(array[i][1])
                individual_score = abs(estimated_reward - total_reward)
                total_scores.append(individual_score)

        print('total_scores', total_scores)

        k = int(num_collaborators * 0.2)
        if fl_round % 2 == 0:
            selected_collaborators = array[np.argsort(total_scores)[-k:], 0]
        else:
            selected_collaborators = array[np.argsort(total_scores)[:k], 0]
        print('selected_collaborators', selected_collaborators)

        return selected_collaborators

    else:
        print('else of greater than 0 ')
        # Return the selected collaborators
        #custom_percentage_collaborator_selector(collaborators, db_iterator, fl_round, collaborators_chosen_each_round,collaborator_times_per_round)
        
        num_to_select = int(num_collaborators * 0.2)
        selected_collaborators = random.sample(collaborators, num_to_select)
        print("selected_collaborators", selected_collaborators)
        return selected_collaborators
        
# a very simple function. Everyone trains every round.
def all_collaborators_train(collaborators,
                            db_iterator,
                            fl_round,
                            collaborators_chosen_each_round,
                            collaborator_times_per_round):
    """Chooses which collaborators will train for a given round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    """
    return collaborators

# this is not a good algorithm, but we include it to demonstrate the following:
    # simple use of the logger and of fl_round
    # you can search through the "collaborator_times_per_round" dictionary to see how long collaborators have been taking
    # you can have a subset of collaborators train in a given round
def one_collaborator_on_odd_rounds(collaborators,
                                   db_iterator,
                                   fl_round,
                                   collaborators_chosen_each_round,
                                   collaborator_times_per_round):
    """Chooses which collaborators will train for a given round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    """
    logger.info("one_collaborator_on_odd_rounds called!")
    # on odd rounds, choose the fastest from the previous round
    if fl_round % 2 == 1:
        training_collaborators = None
        fastest_time = np.inf
        
        # the previous round information will be index [fl_round - 1]
        # this information is itself a dictionary of {col: time}
        for col, t in collaborator_times_per_round[fl_round - 1].items():
            if t < fastest_time:
                fastest_time = t
                training_collaborators = [col]
    else:
        training_collaborators = collaborators
    return training_collaborators


# # Custom hyperparameters for training

# You can customize the hyper-parameters for the training collaborators at each round. At the start of the round, the experiment loop will invoke your function and set the hyper-parameters for that round based on what you return.
# 
# The training hyper-parameters for a round are:
# - **`learning_rate`**: the learning rate value set for the Adam optimizer
# - **`batches_per_round`**: a flat number of batches each training collaborator will train. Must be an integer or None
# - **`epochs_per_round`**: the number of epochs each training collaborator will train. Must be a float or None. Partial epochs are allowed, such as 0.5 epochs.
# 
# Note that exactly one of **`epochs_per_round`** and **`batches_per_round`** must be `"None"`. You will get an error message and the experiment will terminate if this is not the case to remind you of this requirement.
# 
# Your function will receive the typical aggregator state/history information that it can use to make its determination. The function must return a tuple of (`learning_rate`, `epochs_per_round`, `batches_per_round`). For example, if you return:
# 
# `(1e-4, 2.5, None)`
# 
# then all collaborators selected based on the [collaborator training selection criteria](#Custom-Collaborator-Training-Selection) will train for `2.5` epochs with a learning rate of `1e-4`.
# 
# Different hyperparameters can be specified for collaborators for different rounds but they remain the same for all the collaborators that are chosen for that particular round. In simpler words, collaborators can not have different hyperparameters for the same round.

# This simple example uses constant hyper-parameters through the experiment
def constant_hyper_parameters(collaborators,
                              db_iterator,
                              fl_round,
                              collaborators_chosen_each_round,
                              collaborator_times_per_round):
    """Set the training hyper-parameters for the round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    Returns:
        tuple of (learning_rate, epochs_per_round, batches_per_round). One of epochs_per_round and batches_per_round must be None.
    """
    # these are the hyperparameters used in the May 2021 recent training of the actual FeTS Initiative
    # they were tuned using a set of data that UPenn had access to, not on the federation itself
    # they worked pretty well for us, but we think you can do better :)
    epochs_per_round = 1.0
    batches_per_round = None
    learning_rate = 5e-5
    return (learning_rate, epochs_per_round, batches_per_round)


# this example trains less at each round
def train_less_each_round(collaborators,
                          db_iterator,
                          fl_round,
                          collaborators_chosen_each_round,
                          collaborator_times_per_round):
    """Set the training hyper-parameters for the round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    Returns:
        tuple of (learning_rate, epochs_per_round, batches_per_round). One of epochs_per_round and batches_per_round must be None.
    """

    # we'll have a constant learning_rate
    learning_rate = 5e-5
    
    # our epochs per round will start at 1.0 and decay by 0.9 for the first 10 rounds
    epochs_per_round = 1.0
    decay = min(fl_round, 10)
    decay = 0.9 ** decay
    epochs_per_round *= decay    
    
    return (learning_rate, epochs_per_round, None)


# this example has each institution train the same number of batches
def fixed_number_of_batches(collaborators,
                            db_iterator,
                            fl_round,
                            collaborators_chosen_each_round,
                            collaborator_times_per_round):
    """Set the training hyper-parameters for the round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    Returns:
        tuple of (learning_rate, epochs_per_round, batches_per_round). One of epochs_per_round and batches_per_round must be None.
    """

    # we'll have a constant learning_rate
    learning_rate = 5e-5
    
    # instead of a number of epochs, collaborators will train for a number of batches
    # this means the number of training batches is irrespective of the data sizes at the institutions
    # if the institution has less data than this, they will loop on their data until they have trained
    # the correct number of batches
    batches_per_round = 16
    
    # Note that the middle element (epochs_per_round) is now None
    return (learning_rate, None, batches_per_round)


# # Custom Aggregation Functions
# Standard aggregation methods allow for simple layer-wise combination (via weighted_mean, mean, median, etc.); 
# however, more complex aggregation methods can be supported by evaluating collaborator metrics, weights from prior rounds, etc. 
# OpenFL enables custom aggregation functions via the 
# [**AggregationFunctionInterface**](https://github.com/intel/openfl/blob/fets/openfl/component/aggregation_functions/interface.py). 
# For the challenge, we wrap this interface so we can pass additional simulation state, such as simulated time.
# 
# [**LocalTensors**](https://github.com/intel/openfl/blob/fets/openfl/utilities/types.py#L13) are named tuples of the form 
# `('collaborator_name', 'tensor', 'collaborator_weight')`. 
# Your custom aggregation function will be passed a list of LocalTensors, 
# which will contain an entry for each collaborator who participated in the prior training round. 
# The [**`tensor_db`**](#Getting-access-to-historical-weights,-metrics,-and-more) gives direct access to the aggregator's tensor_db dataframe 
# and includes all tensors / metrics reported by collaborators. Using the passed tensor_db reference, 
# participants may even store custom information by using in-place write operations. A few examples are included below.
# 
# We also provide a number of convenience functions to be used in conjunction with the TensorDB for those who are less familiar with pandas. 
# These are added directly to the dataframe object that gets passed to the aggregation function to make it easier to *store* , *retrieve*, and *search* 
# through the database so that participants can focus on algorithms instead of infrastructure / framework details.
#
# tensor_db.store:
#
#        Convenience method to store a new tensor in the dataframe.
#        Args:
#            tensor_name [ optional ] : The name of the tensor (or metric) to be saved
#            origin      [ optional ] : Origin of the tensor
#            fl_round    [ optional ] : Round the tensor is associated with
#            metric:     [ optional ] : Is the tensor a metric?
#            tags:       [ optional ] : Tuple of unstructured tags associated with the tensor
#            np.array    [ required ] : Value to store associated with the other included information (i.e. TensorKey info)
#            overwrite:  [ optional ] : If the tensor is already present in the dataframe
#                                       should it be overwritten?
#        Returns:
#            None
#
#
# tensor_db.retrieve
# 
#        Convenience method to retrieve tensor from the dataframe.
#        Args:
#            tensor_name [ optional ] : The name of the tensor (or metric) to retrieve
#            origin      [ optional ] : Origin of the tensor
#            fl_round    [ optional ] : Round the tensor is associated with
#            metric:     [ optional ] : Is the tensor a metric?
#            tags:       [ optional ] : Tuple of unstructured tags associated with the tensor
#                                       should it be overwritten?
#        Returns:
#            Optional[ np.array ]     : If there is a match, return the first row
#
# tensor_db.search
#
#        Search the tensor_db dataframe based on:
#            - tensor_name
#            - origin
#            - fl_round
#            - metric
#            -tags
#        Returns a new dataframe that matched the query
#        Args:
#            tensor_name: The name of the tensor (or metric) to be searched
#            origin:      Origin of the tensor
#            fl_round:    Round the tensor is associated with
#            metric:      Is the tensor a metric?
#            tags:        Tuple of unstructured tags associated with the tensor
#        Returns:
#            pd.DataFrame : New dataframe that matches the search query from 
#                           the tensor_db dataframe
#
# ## Converting the tensor_db to a db_iterator (to reuse aggregation methods from last year's competition)
# ### Using prior layer weights
# Here is an example of how to extract layer weights from prior round. The tag is `'aggregated'` indicates this : 
#     
#     for _, record in tensor_db.iterrows():
#             if (
#                 record['round'] == (fl_round - 1)
#                 and record['tensor_name'] == tensor_name
#                 and 'aggregated' in record['tags']
#                 and 'delta' not in record['tags']
#                ):
#                 previous_tensor_value = record['nparray']
#                 break
# 
# ### Using validation metrics for filtering
# 
#     threshold = fl_round * 0.3 + 0.5
#     metric_name = 'acc'
#     tags = ('metric','validate_agg')
#     selected_tensors = []
#     selected_weights = []
#     for _, record in tensor_db.iterrows():
#         for local_tensor in local_tensors:
#             tags = set(tags + [local_tensor.col_name])
#             if (
#                 tags <= set(record['tags']) 
#                 and record['round'] == fl_round
#                 and record['tensor_name'] == metric_name
#                 and record['nparray'] >= threshold
#             ):
#                 selected_tensors.append(local_tensor.tensor)
#                 selected_weights.append(local_tensor.weight)
# 
# ### A Note about true OpenFL deployments
# The OpenFL custom aggregation interface does not currently provide timing information, 
# so please note that any solutions that make use of simulated time will need to be adapted to be truly OpenFL compatible in a real federation by using actual `time.time()` 
# calls (or similar) instead of the simulated time.
# 
# Solutions that use neither **`collaborators_chosen_each_round`** or **`collaborator_times_per_round`** will match the existing OpenFL aggregation customization interface, 
# thus could be used in a real federated deployment using OpenFL.

# here we will clip outliers by clipping deltas to the Nth percentile (e.g. 80th percentile)
def clipped_aggregation(local_tensors,
                        db_iterator,
                        tensor_name,
                        fl_round,
                        collaborators_chosen_each_round,
                        collaborator_times_per_round):
    """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    # the percentile we will clip to
    clip_to_percentile = 80
    
    # first, we need to determine how much each local update has changed the tensor from the previous value
    # we'll use the db_iterator to find the previous round's value for this tensor
    previous_tensor_value = None
    for record in db_iterator:
        if (
            record['round'] == (fl_round - 1)
            and record['tensor_name'] == tensor_name
            and 'aggregated' in record['tags']
            and 'delta' not in record['tags']
           ):
            previous_tensor_value = record['nparray']
            break
       
    # if we have no previous tensor_value, we won't actually clip
    if previous_tensor_value is None:
        clipped_tensors = [t.tensor for t in local_tensors]
    # otherwise, we will use clipped deltas
    else:
        # compute the deltas
        deltas = [t.tensor - previous_tensor_value for t in local_tensors]
    
        # concatenate all the deltas
        all_deltas = np.concatenate(deltas)
        
        # take the absolute value
        abs_deltas = np.abs(all_deltas)
        
        # finally, get the 80th percentile
        clip_value = np.percentile(abs_deltas, clip_to_percentile)
        
        # let's log what we're clipping to
        logger.info("Clipping tensor {} to value {}".format(tensor_name, clip_value))
    
        # now we can compute our clipped tensors
        clipped_tensors = []
        for delta, t in zip(deltas, local_tensors):
            new_tensor = previous_tensor_value + np.clip(delta, -1 * clip_value, clip_value)
            clipped_tensors.append(new_tensor)
        
    # get an array of weight values for the weighted average
    weights = [t.weight for t in local_tensors]

    # return the weighted average of the clipped tensors
    return np.average(clipped_tensors, weights=weights, axis=0)

# the simple example of weighted FedAVG
def weighted_average_aggregation(local_tensors,
                                 tensor_db,
                                 tensor_name,
                                 fl_round,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
    """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        tensor_db: pd.DataFrame that contains global tensors / metrics.
            Columns: ['tensor_name', 'origin', 'round', 'report',  'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    # basic weighted fedavg

    # here are the tensor values themselves
    tensor_values = [t.tensor for t in local_tensors]
    
    # and the weights (i.e. data sizes)
    weight_values = [t.weight for t in local_tensors]
    
    # so we can just use numpy.average
    return np.average(tensor_values, weights=weight_values, axis=0)

# here we will clip outliers by clipping deltas to the Nth percentile (e.g. 80th percentile)
def clipped_aggregation(local_tensors,
                        tensor_db,
                        tensor_name,
                        fl_round,
                        collaborators_chosen_each_round,
                        collaborator_times_per_round):
    """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        tensor_db: pd.DataFrame that contains global tensors / metrics.
            Columns: ['tensor_name', 'origin', 'round', 'report',  'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    # the percentile we will clip to
    clip_to_percentile = 80
    
    # first, we need to determine how much each local update has changed the tensor from the previous value
    # we'll use the tensor_db search function to find the 
    previous_tensor_value = tensor_db.search(tensor_name=tensor_name, fl_round=fl_round, tags=('model',), origin='aggregator')

    if previous_tensor_value.shape[0] > 1:
        print(previous_tensor_value)
        raise ValueError(f'found multiple matching tensors for {tensor_name}, tags=(model,), origin=aggregator')

    if previous_tensor_value.shape[0] < 1:
        # no previous tensor, so just return the weighted average
        return weighted_average_aggregation(local_tensors,
                                            tensor_db,
                                            tensor_name,
                                            fl_round,
                                            collaborators_chosen_each_round,
                                            collaborator_times_per_round)

    previous_tensor_value = previous_tensor_value.nparray.iloc[0]

    # compute the deltas for each collaborator
    deltas = [t.tensor - previous_tensor_value for t in local_tensors]

    # get the target percentile using the absolute values of the deltas
    clip_value = np.percentile(np.abs(deltas), clip_to_percentile)
        
    # let's log what we're clipping to
    logger.info("Clipping tensor {} to value {}".format(tensor_name, clip_value))
    
    # now we can compute our clipped tensors
    clipped_tensors = []
    for delta, t in zip(deltas, local_tensors):
        new_tensor = previous_tensor_value + np.clip(delta, -1 * clip_value, clip_value)
        clipped_tensors.append(new_tensor)
        
    # get an array of weight values for the weighted average
    weights = [t.weight for t in local_tensors]

    # return the weighted average of the clipped tensors
    return np.average(clipped_tensors, weights=weights, axis=0)

# Adapted from FeTS Challenge 2021
# Federated Brain Tumor Segmentation:Multi-Institutional Privacy-Preserving Collaborative Learning
# Ece Isik-Polat, Gorkem Polat,Altan Kocyigit1, and Alptekin Temizel1
def FedAvgM_Selection(local_tensors,
                      tensor_db,
                      tensor_name,
                      fl_round,
                      collaborators_chosen_each_round,
                      collaborator_times_per_round):
    
        """Aggregate tensors.

        Args:
            local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
            tensor_db: Aggregator's TensorDB [writable]. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                - 'round': 0-based number of round corresponding to this tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training result.
                        These tensors are passed to the aggregator node after local learning.
                    - 'aggregated' indicates that tensor is a result of aggregation.
                        These tensors are sent to collaborators for the next round.
                    - 'delta' indicates that value is a difference between rounds
                        for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
        Returns:
            np.ndarray: aggregated tensor
        """
        #momentum
        tensor_db.store(tensor_name='momentum',nparray=0.9,overwrite=False)
        #aggregator_lr
        tensor_db.store(tensor_name='aggregator_lr',nparray=1.0,overwrite=False)

        if fl_round == 0:
            # Just apply FedAvg

            tensor_values = [t.tensor for t in local_tensors]
            weight_values = [t.weight for t in local_tensors]               
            new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        

            #if not (tensor_name in weight_speeds):
            if tensor_name not in tensor_db.search(tags=('weight_speeds',))['tensor_name']:    
                #weight_speeds[tensor_name] = np.zeros_like(local_tensors[0].tensor) # weight_speeds[tensor_name] = np.zeros(local_tensors[0].tensor.shape)
                tensor_db.store(
                    tensor_name=tensor_name, 
                    tags=('weight_speeds',), 
                    nparray=np.zeros_like(local_tensors[0].tensor),
                )
            return new_tensor_weight        
        else:
            if tensor_name.endswith("weight") or tensor_name.endswith("bias"):
                # Calculate aggregator's last value
                previous_tensor_value = None
                for _, record in tensor_db.iterrows():
                    if (record['round'] == fl_round 
                        and record["tensor_name"] == tensor_name
                        and record["tags"] == ("aggregated",)): 
                        previous_tensor_value = record['nparray']
                        break

                if previous_tensor_value is None:
                    logger.warning("Error in fedAvgM: previous_tensor_value is None")
                    logger.warning("Tensor: " + tensor_name)

                    # Just apply FedAvg       
                    tensor_values = [t.tensor for t in local_tensors]
                    weight_values = [t.weight for t in local_tensors]               
                    new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        
                    
                    if tensor_name not in tensor_db.search(tags=('weight_speeds',))['tensor_name']:    
                        tensor_db.store(
                            tensor_name=tensor_name, 
                            tags=('weight_speeds',), 
                            nparray=np.zeros_like(local_tensors[0].tensor),
                        )

                    return new_tensor_weight
                else:
                    # compute the average delta for that layer
                    deltas = [previous_tensor_value - t.tensor for t in local_tensors]
                    weight_values = [t.weight for t in local_tensors]
                    average_deltas = np.average(deltas, weights=weight_values, axis=0) 

                    # V_(t+1) = momentum*V_t + Average_Delta_t
                    tensor_weight_speed = tensor_db.retrieve(
                        tensor_name=tensor_name,
                        tags=('weight_speeds',)
                    )
                    
                    momentum = float(tensor_db.retrieve(tensor_name='momentum'))
                    aggregator_lr = float(tensor_db.retrieve(tensor_name='aggregator_lr'))
                    
                    new_tensor_weight_speed = momentum * tensor_weight_speed + average_deltas # fix delete (1-momentum)
                    
                    tensor_db.store(
                        tensor_name=tensor_name, 
                        tags=('weight_speeds',), 
                        nparray=new_tensor_weight_speed
                    )
                    # W_(t+1) = W_t-lr*V_(t+1)
                    new_tensor_weight = previous_tensor_value - aggregator_lr*new_tensor_weight_speed

                    return new_tensor_weight
            else:
                # Just apply FedAvg       
                tensor_values = [t.tensor for t in local_tensors]
                weight_values = [t.weight for t in local_tensors]               
                new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)

                return new_tensor_weight

def my_sum(input_array):
    #result = 0
    if bool(input_array.ndim > 1):#2 in itertools.chain.from_iterable(input_array, dtype="float32"):#any(isinstance(input_array, (list, tuple))) == True:#any(isinsta$
        print("YUP NESTED")
        means = np.sum(input_array,axis=1)
        result = np.sum(means,axis=0)
        if bool(result.ndim > 1):#2 in itertools.chain.from_iterable(result, dtype="float32"):#isinstance(result, (list, tuple)):
            result = my_sum(result)
            #result=np.sum(means, axis=0)
    else:
        print('NOT NESTED')
        result=input_array.copy()#np.sum(input_array,axis=0)#input_array.copy()

    if np.isscalar(result):
        return result
    else:
        return np.sum(result)

def original_simagg_hmean(local_tensors,
                                     db_iterator,
                                     tensor_name,
                                     fl_round,
                                     collaborators_chosen_each_round,
                                     collaborator_times_per_round):
    """Aggregate tensors using a similarity-weighted harmonic mean.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    epsilon = 1e-5
    tensor_values = [t.tensor for t in local_tensors]

    if not any([bool(value.tolist()) for value in tensor_values]):
        print('local_tensors')
        print(local_tensors)

    weight_values = [t.weight for t in local_tensors]
    total_average_of_tensors = np.average(tensor_values, axis=0)

    distances = []
    for tv in tensor_values:
        temp_abs = abs(my_sum(total_average_of_tensors) - my_sum(tv))
        distances.append(temp_abs)

    print(distances)
    weights = []
    total_distance = np.sum(distances)

    for i, dist in enumerate(distances):
        weights.append(total_distance / (epsilon + dist))

    weight_sum = np.sum(weights)
    weights_norm = []
    for i in range(len(weights)):
        weights_norm.append(weight_values[i] + weights[i] / (epsilon + weight_sum))

    weight_sum_1 = np.sum(weights_norm)
    weights_norm_1 = []
    for i in range(len(weights_norm)):
        weights_norm_1.append(weights_norm[i] / (epsilon + weight_sum_1))

    try:
        return hmean(tensor_values, weights=weights_norm_1, axis=0)
    except:
        print('FEDAVG CALLED')
        return np.average(tensor_values, weights=weight_values, axis=0)

def sim_agg_weight_bias_hmean(local_tensors,
                              tensor_db,
                              tensor_name,
                              fl_round,
                              collaborators_chosen_each_round,
                              collaborator_times_per_round):
    """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    if tensor_name.endswith("weight") or tensor_name.endswith("bias"):
        print('sim_agg_weight_bias_hmean CALLED')
        #print(f"collaborator_times_per_round: {collaborator_times_per_round}")
        epsilon = 1e-5
        tensor_values = [t.tensor for t in local_tensors]

        # Uncomment if needed
        # if not any([bool(value.tolist()) for value in tensor_values]):
        #     print('local_tensors')
        #     print(local_tensors)

        weight_values = [t.weight for t in local_tensors]
        total_average_of_tensors = np.average(tensor_values, axis=0)

        distances = []
        for tv in tensor_values:
            temp_abs = abs(my_sum(total_average_of_tensors) - my_sum(tv))
            distances.append(temp_abs)

        print(distances)
        weights = []
        total_distance = np.sum(distances)

        for i, dist in enumerate(distances):
            weights.append(total_distance / (epsilon + dist))

        weight_sum = np.sum(weights)

        weights_norm = []
        for i in range(len(weights)):
            weights_norm.append(weight_values[i] + weights[i] / (weight_sum + epsilon))  # + weights[i]/(weight_sum + epsilon)) #yaha pr add ki jgha multiply lhaya ha bs

        weight_sum_1 = np.sum(weights_norm)
        weights_norm_1 = []
        for i in range(len(weights_norm)):
            weights_norm_1.append(weights_norm[i] / (weight_sum_1 + epsilon))

        try:
            return hmean(tensor_values, weights=weights_norm_1, axis=0)  # hmean
        except:
            # print('FEDAVG CALLED')
            return np.average(tensor_values, weights=weight_values, axis=0)
    else:
        # print('FEDAVG CALLED')
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]
        return np.average(tensor_values, weights=weight_values, axis=0)
def olympic_mean(values, weights, trim_percent=0.2):
    """Calculate the Olympic mean by trimming the extremes and then averaging."""
    # Sort values and weights together based on values
    sorted_indices = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, sorted_indices, axis=0)
    sorted_weights = np.take_along_axis(weights, sorted_indices, axis=0)

    # Calculate the number of elements to trim
    trim_count = int(trim_percent * len(values))

    # Trim the sorted values and weights
    trimmed_values = sorted_values[trim_count: -trim_count]
    trimmed_weights = sorted_weights[trim_count: -trim_count]

    # Return the weighted average of the trimmed values
    return np.average(trimmed_values, weights=trimmed_weights, axis=0)

def sim_agg_weight_bias_olympic_mean(local_tensors, tensor_db, tensor_name, fl_round, collaborators_chosen_each_round, collaborator_times_per_round):
    """Aggregate tensors using the Olympic mean to avoid extreme changes."""
    if tensor_name.endswith("weight") or tensor_name.endswith("bias"):
        epsilon = 1e-5
        tensor_values = [t.tensor for t in local_tensors]

        weight_values = [t.weight for t in local_tensors]
        total_average_of_tensors = np.average(tensor_values, axis=0)

        distances = []
        for tv in tensor_values:
            temp_abs = abs(np.sum(total_average_of_tensors) - np.sum(tv))
            distances.append(temp_abs)

        weights = []
        total_distance = np.sum(distances)

        for i, dist in enumerate(distances):
            weights.append(total_distance / (epsilon + dist))

        weight_sum = np.sum(weights)
        weights_norm = [(weight_values[i] + weights[i] / (weight_sum + epsilon)) for i in range(len(weights))]

        weight_sum_1 = np.sum(weights_norm)
        weights_norm_1 = [(w / (weight_sum_1 + epsilon)) for w in weights_norm]

        try:
            # Use the Olympic mean instead of the harmonic mean
            return olympic_mean(np.array(tensor_values), np.array(weights_norm_1), trim_percent=0.2)
        except:
            return np.average(tensor_values, weights=weight_values, axis=0)
    else:
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]
        return np.average(tensor_values, weights=weight_values, axis=0)

def sim_weighted_average_aggregation(local_tensors,
                                 tensor_db,
                                 tensor_name,
                                 fl_round,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
    """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    print(f"collaborator_times_per_round: {collaborator_times_per_round}")
    epsilon = 1e-5
    tensor_values = [t.tensor for t in local_tensors]

#    if not any([bool(value.tolist()) for value in tensor_values]):
#        print('local_tensors')
#        print(local_tensors)

    weight_values = [t.weight for t in local_tensors]
    total_average_of_tensors = np.average(tensor_values, axis=0)

    distances = []
    for tv in tensor_values:
        temp_abs = abs(my_sum(total_average_of_tensors) - my_sum(tv))
        distances.append(temp_abs)

    print(distances)
    weights = []
    total_distance = np.sum(distances)

    for i, dist in enumerate(distances):
        weights.append(total_distance/(epsilon + dist))

    weight_sum = np.sum(weights)

    weights_norm = []
    for i in range(len(weights)):
        weights_norm.append(weight_values[i] + weights[i]/(weight_sum + epsilon)) # + weights[i]/(weight_sum + epsilon)) #yaha pr add ki jgha multiply lhaya ha bs

    weight_sum_1 = np.sum(weights_norm)
    weights_norm_1 = []
    for i in range(len(weights_norm)):
        weights_norm_1.append(weights_norm[i]/(weight_sum_1 + epsilon))

    try:
        #return np.average(tensor_values, weights=weights_norm_1, axis=0)
        return hmean(tensor_values, weights=weights_norm_1, axis=0) #hmean
        #return np.average(tensor_values, weights=updated_noised_weights, axis=0)
    except:
        print('FEDAVG CALLED')
        return np.average(tensor_values, weights=weight_values, axis=0)

def reg_sim_weighted_average_aggregation(local_tensors,
                                 tensor_db,
                                 tensor_name,
                                 fl_round,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
    """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    print(f"collaborator_times_per_round: {collaborator_times_per_round}")
    epsilon = 1e-5
    tensor_values = [t.tensor for t in local_tensors]

#    if not any([bool(value.tolist()) for value in tensor_values]):
#        print('local_tensors')
#        print(local_tensors)

    weight_values = [t.weight for t in local_tensors]
    total_average_of_tensors = np.average(tensor_values, axis=0)

    distances = []
    for tv in tensor_values:
        temp_abs = abs(my_sum(total_average_of_tensors) - my_sum(tv))
        distances.append(temp_abs)

    print(distances)
    weights = []
    total_distance = np.sum(distances)

    for i, dist in enumerate(distances):
        weights.append(total_distance/(epsilon + dist))

    weight_sum = np.sum(weights)
    weights_norm = []
    for i in range(len(weights)):
        weights_norm.append(weight_values[i] * weights[i]/(weight_sum + epsilon)) # + weights[i]/(weight_sum + epsilon)) #yaha pr add ki jgha multiply lhaya ha bs

    weight_sum_1 = np.sum(weights_norm)
    weights_norm_1 = []
    for i in range(len(weights_norm)):
        weights_norm_1.append(weights_norm[i]/(weight_sum_1 + epsilon))


    if fl_round > 10:
        for _, record in tensor_db.iterrows():
            if (record['round'] == (fl_round - 1)
                and record['tensor_name'] == tensor_name
                and 'aggregated' in record['tags']
                and 'delta' not in record['tags']):
                previous_tensor_value = record['nparray']
                break

        deltas = [previous_tensor_value - t.tensor for t in local_tensors]
        average_deltas = np.average(deltas, weights=weight_values, axis=0)
        weights_norm_2 = []
        for i in range(len(weights_norm_1)):
            weights_norm_2.append(weights_norm_1[i]/(average_deltas + epsilon))
        
        try:
            return np.round(np.average(tensor_values, weights=weights_norm_2, axis=0),4)
        except: 
            print('FEDAVG CALLED')
            return np.round(np.average(tensor_values, weights=weight_values, axis=0),4)
    else: 
        try:
            return np.round(np.average(tensor_values, weights=weights_norm_1, axis=0),4)
        #return np.average(tensor_values, weights=updated_noised_weights, axis=0)
        except:
            print('FEDAVG CALLED')
            return np.round(np.average(tensor_values, weights=weight_values, axis=0),4)

all_time_colab_list = []

def custom_percentage_collaborator_selector_without_repetition(collaborators,
                                   db_iterator,
                                   fl_round,
                                   collaborators_chosen_each_round,
                                   collaborator_times_per_round):
    """Chooses which collaborators will train for a given round.

    Args:
	collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """

    logger.info("custom_percentage_collaborator_selector_without_repetition called!")
    global all_time_colab_list
    original_array=collaborators
    #print('custom_percentage_collaborator_selector function called')
    percentage = 0.2
    windowsize = (len(collaborators)*percentage)

    collaborators=equal_partitions(collaborators,windowsize)

    pieces=np.ceil(len(collaborators)/windowsize) # how many pieces of collaborators
    print('pieces: ',pieces)

    if fl_round>=pieces: # fl_round > number of peices
        print('in if fl_round>=pieces',fl_round,'>=',pieces,' is TRUE')
        print('fl_round-int(fl_round/pieces)*pieces:',fl_round-(int(fl_round/pieces)*pieces))
        if (fl_round-(int(fl_round/pieces)*pieces)) == 0: #(c-int(c/l)*l) # to enable shuffle
           print('********************SHUFFLE YES********************')
           collaborators=original_array # reassigning to original array
           np.random.shuffle(collaborators) # shuffle
           #custom_percentage_collaborator_selector_without_repetition(collaborators,db_iterator,fl_round,collaborators_chosen_each_round,collaborator_times_per_round)
          # the array is saved in the file temp_colab.npy 
           np.save('temp_colab', collaborators)
        else:
            # to load from temp_colab.npy file
            # the array is loaded into b
            print('loading colabs')
            temp_collaborators = np.load('temp_colab.npy')
            collaborators = np.asarray(temp_collaborators)
    else:
        print('in else fl_round>=pieces',fl_round,'>=',pieces,' is false')
        
    #print(f"collaborators: {collaborators}")
    #print(f"type(collaborators): {type(collaborators)}")

    for i in range(fl_round*int(windowsize)%len(collaborators), len(collaborators), int(windowsize)): # 
    #################################################LOGIC TO CHECK IF COLAB EXISTS IN LISTS OF LIST####################### 
        #selected_colabs_to_check = collaborators[i: i+int(windowsize)]
        if type(collaborators[i: i+int(windowsize)]) is np.ndarray:#!= 'list':
            #print("point 1000: collaborators not a list")
            selected_colabs_to_check = collaborators[i: i+int(windowsize)]
        else:
            #print("point 1001: collaborators is a list")
            selected_colabs_to_check = np.asarray(collaborators[i: i+int(windowsize)])
        #print(f"type(selected_colabs_to_check): {type(selected_colabs_to_check)}")
        #print(f"type(all_time_colab_list): {type(all_time_colab_list)}")
        #print(f"selected_colabs_to_check: {selected_colabs_to_check}")
        #print(f"selected_colabs_to_check.tolist(): {selected_colabs_to_check.tolist()}")
        #print(f"len(all_time_colab_list): {len(all_time_colab_list)}")
        if len(all_time_colab_list) == 0:# if global array is not null
            #print("if len(all_time_colab_list) == 0:")
            all_time_colab_list.append(selected_colabs_to_check.tolist())
            #print(f"all_time_colab_list: {all_time_colab_list}")
            return selected_colabs_to_check
        else:
            if selected_colabs_to_check.tolist() in all_time_colab_list:
                #print("else of len(all_time_colab_list) == 0:")
                #print("if collaborators[i+int(windowsize)] in all_time_colab_list:")
                collaborators=original_array
                np.random.shuffle(collaborators)
                #print(f"all_time_colab_list: {all_time_colab_list}")
                collaborators=equal_partitions(collaborators,windowsize)
                np.save('temp_colab', collaborators)
                custom_percentage_collaborator_selector_without_repetition(collaborators,db_iterator,fl_round,collaborators_chosen_each_round,collaborator_times_per_round)
                #goto beginning of function 
            else:
                #print("else of len(all_time_colab_list) == 0:")
                #print("else of collaborators[i+int(windowsize)] in all_time_colab_list:")
                all_time_colab_list.append(selected_colabs_to_check.tolist())
                #print(f"all_time_colab_list: {all_time_colab_list}")
                return selected_colabs_to_check

#Particle Swarm Optimization (PSO) Collaborator Selector 
#THIS METHOD SOMETIMES BREAKS SO I RUN IT AGAIN AND AGAIN :)
def pso_collaborator_selector(collaborators,
                              db_iterator,
                              fl_round,
                              collaborators_chosen_each_round,
                              collaborator_times_per_round):
    """
    Selects collaborators for the given round using Particle Swarm Optimization (PSO).

    Args:
        collaborators (list): List of collaborator names.
        db_iterator (iterator): Iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray'].
        fl_round (int): Current round number.
        collaborators_chosen_each_round (dict): Dictionary of {round: list of collaborators}.
        collaborator_times_per_round (dict): Dictionary of {round: {collaborator: total_time_taken_in_round}}.
        swarm_size (int): Number of particles in the swarm.
        iterations (int): Number of iterations to run the optimization.
        inertia (float): Inertia weight for velocity update.
        cognitive_coeff (float): Cognitive coefficient for personal best influence.
        social_coeff (float): Social coefficient for global best influence.

    Returns:
        list: Selected collaborators for the current round.
    """
    print('pso_collaborator_selector called')

    swarm_size=20
    iterations=50
    inertia=0.5
    cognitive_coeff=1.5
    social_coeff=1.5

    # Step 1: Collect metrics
    agg_dice_results = []
    agg_loss_results = []

    # Collect dice and loss results from the previous round
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    # Convert results to dictionaries
    dice_scores = dict(agg_dice_results)
    loss_scores = dict(agg_loss_results)

    # Sum up times for each collaborator across all rounds
    total_times = Counter()
    for round_data in collaborator_times_per_round.values():
        total_times.update(round_data)

    # Collect all selections and count frequencies
    all_selections = [col for values in collaborators_chosen_each_round.values() for col in values]
    frequency_counter = Counter(all_selections)

    # Prepare collaborator data
    collaborator_data = {}
    for collaborator in collaborators:
        dice_score = float(dice_scores.get(collaborator, 0))
        loss_score = float(loss_scores.get(collaborator, 0))
        total_time = float(total_times.get(collaborator, 0))
        frequency = float(frequency_counter.get(collaborator, 0))

        collaborator_data[collaborator] = {
            'dice': dice_score,
            'loss': loss_score,
            'time': total_time,
            'frequency': frequency
        }

    # Normalize the attributes for all collaborators
    def normalize_attribute(attr_name):
        attr_values = [data[attr_name] for data in collaborator_data.values()]
        max_value = max(attr_values) if attr_values else 1
        for data in collaborator_data.values():
            data[attr_name] = data[attr_name] / max_value if max_value else 0

    normalize_attribute('dice')
    normalize_attribute('loss')
    normalize_attribute('time')
    normalize_attribute('frequency')

    # Step 2: Particle Swarm Optimization
    num_to_select = max(1, int(len(collaborators) * 0.2))  # Select 20% of collaborators
    particles = [random.sample(collaborators, num_to_select) for _ in range(swarm_size)]
    velocities = [np.zeros(len(collaborators)) for _ in range(swarm_size)]  # Velocity for each particle

    def fitness(particle):
        # Calculate fitness based on collaborator attributes
        total_dice = sum(collaborator_data[col]['dice'] for col in particle)
        total_loss = sum(collaborator_data[col]['loss'] for col in particle)
        total_time = sum(collaborator_data[col]['time'] for col in particle)
        total_frequency = sum(collaborator_data[col]['frequency'] for col in particle)

        # Higher dice and lower loss are better; lower time and frequency promote efficiency and fairness
        return (total_dice - total_loss) / (1 + total_time + total_frequency)

    # Initialize personal bests and global best
    personal_best_positions = particles.copy()
    personal_best_scores = [fitness(p) for p in particles]
    global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
    global_best_score = max(personal_best_scores)

    for iteration in range(iterations):
        for i, particle in enumerate(particles):
            # Update velocity
            velocity_update = (
                inertia * velocities[i]
                + cognitive_coeff * random.random() * np.array([1 if col in personal_best_positions[i] else 0 for col in collaborators])
                + social_coeff * random.random() * np.array([1 if col in global_best_position else 0 for col in collaborators])
            )
            velocities[i] = velocity_update

            # Update particle position
            probabilities = np.clip(velocities[i], 0, 1)
            new_particle = [collaborators[idx] for idx, prob in enumerate(probabilities) if prob > 0.5]
            if len(new_particle) > num_to_select:
                particles[i] = random.sample(new_particle, num_to_select)
            else:
                particles[i] = new_particle

            # Update personal best
            current_fitness = fitness(particles[i])
            if current_fitness > personal_best_scores[i]:
                personal_best_positions[i] = particles[i]
                personal_best_scores[i] = current_fitness

        # Update global best
        best_particle_idx = np.argmax(personal_best_scores)
        if personal_best_scores[best_particle_idx] > global_best_score:
            global_best_position = personal_best_positions[best_particle_idx]
            global_best_score = personal_best_scores[best_particle_idx]

    print("Selected Collaborators:", global_best_position)
    return global_best_position


def genetic_algorithm_collaborator_selector(collaborators, 
                                             db_iterator, 
                                             fl_round, 
                                             collaborators_chosen_each_round, 
                                             collaborator_times_per_round):
    """
    Selects collaborators for the given round using a Genetic Algorithm approach.

    Args:
        collaborators: list of strings of collaborator names.
        db_iterator: iterator over history of all tensors.
        fl_round: round number.
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
        population_size (int): Number of candidate solutions in each generation.
        num_generations (int): Number of iterations to evolve the solutions.
        mutation_rate (float): Probability of mutation in the genetic algorithm.
        selection_ratio (float): Proportion of top-performing solutions to select for reproduction.

    Returns:
        list: Selected collaborators for the round.
    """
    logger.info("genetic_algorithm_collaborator_selector called!")
    population_size=20
    num_generations=50
    mutation_rate=0.1
    selection_ratio=0.2
    # Step 1: Gather performance metrics from the previous round
    agg_dice_results, agg_loss_results = [], []
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    colab_numbers_dice, agg_valid_dice = zip(*agg_dice_results) if agg_dice_results else ([], [])
    colab_numbers_loss, agg_valid_loss = zip(*agg_loss_results) if agg_loss_results else ([], [])

    total_times = {col: sum(round_data.get(col, 0) for round_data in collaborator_times_per_round.values())
                   for col in collaborators}
    frequency_counter = {col: sum(col in values for values in collaborators_chosen_each_round.values())
                         for col in collaborators}

    # Step 2: Initialize a population of random subsets of collaborators
    num_to_select = max(1, int(len(collaborators) * 0.2))  # Top 20% collaborators
    population = [random.sample(collaborators, num_to_select) for _ in range(population_size)]

    # Step 3: Fitness function
    def fitness(solution):
        # Calculate average performance metrics for the solution
        dice_scores = [dict(zip(colab_numbers_dice, agg_valid_dice)).get(c, 0) for c in solution]
        loss_scores = [dict(zip(colab_numbers_loss, agg_valid_loss)).get(c, 0) for c in solution]
        times = [total_times.get(c, 0) for c in solution]
        frequencies = [frequency_counter.get(c, 0) for c in solution]

        # Maximize DICE, minimize loss, minimize total time, and balance participation frequency
        dice_avg = np.mean(dice_scores) if dice_scores else 0
        loss_avg = np.mean(loss_scores) if loss_scores else float('inf')
        time_total = np.sum(times)
        frequency_balance = -np.std(frequencies)  # Lower variance in frequency is better

        # Combine into a weighted score
        return dice_avg - loss_avg - 0.1 * time_total + frequency_balance

    # Step 4: Genetic Algorithm loop
    for _ in range(num_generations):
        # Evaluate fitness for the population
        fitness_scores = [fitness(solution) for solution in population]
        sorted_population = [solution for _, solution in sorted(zip(fitness_scores, population), reverse=True)]

        # Select top solutions for reproduction
        num_selected = max(1, int(selection_ratio * population_size))
        selected_population = sorted_population[:num_selected]

        # Generate new population through crossover
        new_population = selected_population[:]
        while len(new_population) < population_size:
            # Select two parents
            parent1, parent2 = random.sample(selected_population, 2)
            # Perform crossover
            crossover_point = random.randint(1, num_to_select - 1)
            child = list(set(parent1[:crossover_point] + parent2[crossover_point:]))
            # Ensure the child has the correct size
            while len(child) < num_to_select:
                child.append(random.choice(collaborators))
            while len(child) > num_to_select:
                child.pop()
            new_population.append(child)

        # Apply mutation
        for solution in new_population:
            if random.random() < mutation_rate:
                replace_idx = random.randint(0, num_to_select - 1)
                solution[replace_idx] = random.choice(collaborators)

        # Update population
        population = new_population

    # Step 5: Select the best solution from the final generation
    best_solution = max(population, key=fitness)
    print("Selected Collaborators:", best_solution)
    return best_solution


def memetic_algorithm_collaborator_selector(collaborators,
                                             db_iterator,
                                             fl_round,
                                             collaborators_chosen_each_round,
                                             collaborator_times_per_round):
    """
    Selects collaborators for the given round using a Memetic Algorithm (MA).

    Args:
        collaborators (list): List of collaborator names.
        db_iterator (iterator): Iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray'].
        fl_round (int): Current round number.
        collaborators_chosen_each_round (dict): Dictionary of {round: list of collaborators}.
        collaborator_times_per_round (dict): Dictionary of {round: {collaborator: total_time_taken_in_round}}.
        population_size (int): Number of individuals in the population.
        generations (int): Number of generations to evolve.
        mutation_rate (float): Probability of mutation in offspring.
        local_search_iterations (int): Number of iterations for local optimization.

    Returns:
        list: Selected collaborators for the current round.
    """
    print('memetic_algorithm_collaborator_selector called')

    population_size=20
    generations=50
    mutation_rate=0.1
    local_search_iterations=5
    # Step 1: Collect metrics
    agg_dice_results = []
    agg_loss_results = []

    # Collect dice and loss results from the previous round
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    # Convert results to dictionaries
    dice_scores = dict(agg_dice_results)
    loss_scores = dict(agg_loss_results)

    # Sum up times for each collaborator across all rounds
    total_times = Counter()
    for round_data in collaborator_times_per_round.values():
        total_times.update(round_data)

    # Collect all selections and count frequencies
    all_selections = [col for values in collaborators_chosen_each_round.values() for col in values]
    frequency_counter = Counter(all_selections)

    # Prepare collaborator data
    collaborator_data = {}
    for collaborator in collaborators:
        dice_score = float(dice_scores.get(collaborator, 0))
        loss_score = float(loss_scores.get(collaborator, 0))
        total_time = float(total_times.get(collaborator, 0))
        frequency = float(frequency_counter.get(collaborator, 0))

        collaborator_data[collaborator] = {
            'dice': dice_score,
            'loss': loss_score,
            'time': total_time,
            'frequency': frequency
        }

    # Normalize the attributes for all collaborators
    def normalize_attribute(attr_name):
        attr_values = [data[attr_name] for data in collaborator_data.values()]
        max_value = max(attr_values) if attr_values else 1
        for data in collaborator_data.values():
            data[attr_name] = data[attr_name] / max_value if max_value else 0

    normalize_attribute('dice')
    normalize_attribute('loss')
    normalize_attribute('time')
    normalize_attribute('frequency')

    # Step 2: Memetic Algorithm
    num_to_select = max(1, int(len(collaborators) * 0.2))  # Select 20% of collaborators

    # Initialize population
    population = [random.sample(collaborators, num_to_select) for _ in range(population_size)]

    def fitness(individual):
        # Calculate fitness based on collaborator attributes
        total_dice = sum(collaborator_data[col]['dice'] for col in individual)
        total_loss = sum(collaborator_data[col]['loss'] for col in individual)
        total_time = sum(collaborator_data[col]['time'] for col in individual)
        total_frequency = sum(collaborator_data[col]['frequency'] for col in individual)

        # Higher dice and lower loss are better; lower time and frequency promote efficiency and fairness
        return (total_dice - total_loss) / (1 + total_time + total_frequency)

    def local_search(individual):
        # Apply local optimization to refine the individual
        for _ in range(local_search_iterations):
            # Randomly replace one collaborator with a new one and check if fitness improves
            new_collaborator = random.choice(collaborators)
            if new_collaborator not in individual:
                replaced = random.choice(individual)
                candidate = individual.copy()
                candidate.remove(replaced)
                candidate.append(new_collaborator)

                if fitness(candidate) > fitness(individual):
                    individual = candidate
        return individual

    def mutate(individual):
        # Mutate the individual by replacing a random collaborator
        if random.random() < mutation_rate:
            collaborator_to_replace = random.choice(individual)
            available_collaborators = list(set(collaborators) - set(individual))
            if available_collaborators:
                new_collaborator = random.choice(available_collaborators)
                idx = individual.index(collaborator_to_replace)
                individual[idx] = new_collaborator
        return individual

    for generation in range(generations):
        # Evaluate fitness of the population
        fitness_scores = [fitness(ind) for ind in population]

        # Select top individuals for breeding
        selected_parents = [
            population[idx]
            for idx in np.argsort(fitness_scores)[-population_size // 2:]
        ]

        # Create next generation through crossover
        offspring = []
        while len(offspring) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = list(set(parent1 + parent2))
            if len(child) > num_to_select:
                child = random.sample(child, num_to_select)
            offspring.append(child)

        # Apply mutation to offspring
        offspring = [mutate(child) for child in offspring]

        # Apply local search to offspring
        offspring = [local_search(child) for child in offspring]

        # Update population
        population = offspring

    # Select the best individual from the final population
    best_individual = max(population, key=fitness)

    print("Selected Collaborators:", best_individual)
    return best_individual



def simulated_annealing_collaborator_selector(collaborators,
                                              db_iterator,
                                              fl_round,
                                              collaborators_chosen_each_round,
                                              collaborator_times_per_round):
    """
    Selects collaborators for the given round using Simulated Annealing (SA).

    Args:
        collaborators (list): List of collaborator names.
        db_iterator (iterator): Iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray'].
        fl_round (int): Current round number.
        collaborators_chosen_each_round (dict): Dictionary of {round: list of collaborators}.
        collaborator_times_per_round (dict): Dictionary of {round: {collaborator: total_time_taken_in_round}}.
        initial_temperature (float): Starting temperature for simulated annealing.
        cooling_rate (float): Rate at which the temperature decreases (0 < cooling_rate < 1).
        max_iterations (int): Maximum number of iterations.

    Returns:
        list: Selected collaborators for the current round.
    """
    print('simulated_annealing_collaborator_selector called')
    initial_temperature=100
    cooling_rate=0.95
    max_iterations=100
    # Step 1: Collect metrics
    agg_dice_results = []
    agg_loss_results = []

    # Collect dice and loss results from the previous round
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    # Convert results to dictionaries
    dice_scores = dict(agg_dice_results)
    loss_scores = dict(agg_loss_results)

    # Sum up times for each collaborator across all rounds
    total_times = Counter()
    for round_data in collaborator_times_per_round.values():
        total_times.update(round_data)

    # Collect all selections and count frequencies
    all_selections = [col for values in collaborators_chosen_each_round.values() for col in values]
    frequency_counter = Counter(all_selections)

    # Prepare collaborator data
    collaborator_data = {}
    for collaborator in collaborators:
        dice_score = float(dice_scores.get(collaborator, 0))
        loss_score = float(loss_scores.get(collaborator, 0))
        total_time = float(total_times.get(collaborator, 0))
        frequency = float(frequency_counter.get(collaborator, 0))

        collaborator_data[collaborator] = {
            'dice': dice_score,
            'loss': loss_score,
            'time': total_time,
            'frequency': frequency
        }

    # Normalize the attributes for all collaborators
    def normalize_attribute(attr_name):
        attr_values = [data[attr_name] for data in collaborator_data.values()]
        max_value = max(attr_values) if attr_values else 1
        for data in collaborator_data.values():
            data[attr_name] = data[attr_name] / max_value if max_value else 0

    normalize_attribute('dice')
    normalize_attribute('loss')
    normalize_attribute('time')
    normalize_attribute('frequency')

    # Step 2: Simulated Annealing
    num_to_select = max(1, int(len(collaborators) * 0.2))  # Select 20% of collaborators

    # Fitness function
    def fitness(solution):
        total_dice = sum(collaborator_data[col]['dice'] for col in solution)
        total_loss = sum(collaborator_data[col]['loss'] for col in solution)
        total_time = sum(collaborator_data[col]['time'] for col in solution)
        total_frequency = sum(collaborator_data[col]['frequency'] for col in solution)
        return (total_dice - total_loss) / (1 + total_time + total_frequency)

    # Generate initial solution
    current_solution = random.sample(collaborators, num_to_select)
    current_fitness = fitness(current_solution)

    # Set initial best solution
    best_solution = current_solution
    best_fitness = current_fitness

    temperature = initial_temperature

    for iteration in range(max_iterations):
        # Generate a neighboring solution by swapping one collaborator
        neighbor_solution = current_solution.copy()
        replaced_collaborator = random.choice(neighbor_solution)
        available_collaborators = list(set(collaborators) - set(current_solution))
        if available_collaborators:
            new_collaborator = random.choice(available_collaborators)
            neighbor_solution.remove(replaced_collaborator)
            neighbor_solution.append(new_collaborator)

        # Evaluate the neighboring solution
        neighbor_fitness = fitness(neighbor_solution)

        # Accept the neighbor with a probability based on the temperature
        if (neighbor_fitness > current_fitness) or \
                (random.random() < np.exp((neighbor_fitness - current_fitness) / temperature)):
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness

        # Update the best solution found
        if current_fitness > best_fitness:
            best_solution = current_solution
            best_fitness = current_fitness

        # Decrease the temperature
        temperature *= cooling_rate

        # Early stopping if temperature is too low
        if temperature < 1e-5:
            break

    print("Selected Collaborators:", best_solution)
    return best_solution


def fuzzy_logic_collaborator_selector(collaborators,
                                       db_iterator,
                                       fl_round,
                                       collaborators_chosen_each_round,
                                       collaborator_times_per_round):
    """
    Selects collaborators for the given round using Fuzzy Logic.

    Args:
        collaborators (list): List of collaborator names.
        db_iterator (iterator): Iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray'].
        fl_round (int): Current round number.
        collaborators_chosen_each_round (dict): Dictionary of {round: list of collaborators}.
        collaborator_times_per_round (dict): Dictionary of {round: {collaborator: total_time_taken_in_round}}.

    Returns:
        list: Selected collaborators for the current round.
    """
    print('fuzzy_logic_collaborator_selector called')

    # Step 1: Collect metrics
    agg_dice_results = []
    agg_loss_results = []

    # Collect dice and loss results from the previous round
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    # Convert results to dictionaries
    dice_scores = dict(agg_dice_results)
    loss_scores = dict(agg_loss_results)

    # Sum up times for each collaborator across all rounds
    total_times = Counter()
    for round_data in collaborator_times_per_round.values():
        total_times.update(round_data)

    # Collect all selections and count frequencies
    all_selections = [col for values in collaborators_chosen_each_round.values() for col in values]
    frequency_counter = Counter(all_selections)

    # Prepare collaborator data
    collaborator_data = {}
    for collaborator in collaborators:
        dice_score = float(dice_scores.get(collaborator, 0))
        loss_score = float(loss_scores.get(collaborator, 0))
        total_time = float(total_times.get(collaborator, 0))
        frequency = float(frequency_counter.get(collaborator, 0))

        collaborator_data[collaborator] = {
            'dice': dice_score,
            'loss': loss_score,
            'time': total_time,
            'frequency': frequency
        }

    # Normalize the attributes for all collaborators
    def normalize_attribute(attr_name):
        attr_values = [data[attr_name] for data in collaborator_data.values()]
        max_value = max(attr_values) if attr_values else 1
        for data in collaborator_data.values():
            data[attr_name] = data[attr_name] / max_value if max_value else 0

    normalize_attribute('dice')
    normalize_attribute('loss')
    normalize_attribute('time')
    normalize_attribute('frequency')

    # Step 2: Apply Fuzzy Logic
    def fuzzify(collaborator):
        """Assign fuzzy membership values to the collaborator based on attributes."""
        data = collaborator_data[collaborator]
        
        # Membership values for "High Performance" (based on DICE and loss)
        performance_high = min(data['dice'], 1 - data['loss'])  # Higher dice, lower loss is better

        # Membership values for "Low Resource Usage" (based on time)
        resource_low = 1 - data['time']  # Lower time is better

        # Membership values for "Fair Participation" (based on frequency)
        participation_fair = 1 - data['frequency']  # Less frequent participation is better

        return {
            'performance_high': performance_high,
            'resource_low': resource_low,
            'participation_fair': participation_fair
        }

    def defuzzify(fuzzy_values):
        """Combine fuzzy values to get a single score for ranking."""
        # Weighted sum of fuzzy values (adjust weights as needed)
        return (0.5 * fuzzy_values['performance_high'] +
                0.3 * fuzzy_values['resource_low'] +
                0.2 * fuzzy_values['participation_fair'])

    # Calculate fuzzy scores for all collaborators
    collaborator_scores = {}
    for collaborator in collaborators:
        fuzzy_values = fuzzify(collaborator)
        collaborator_scores[collaborator] = defuzzify(fuzzy_values)

    # Rank collaborators by fuzzy score
    ranked_collaborators = sorted(collaborator_scores.items(), key=lambda x: x[1], reverse=True)

    # Select the top 20% of collaborators
    num_to_select = max(1, int(len(collaborators) * 0.2))
    selected_collaborators = [collaborator for collaborator, score in ranked_collaborators[:num_to_select]]

    print("Selected Collaborators:", selected_collaborators)
    return selected_collaborators




def ant_colony_optimization_collaborator_selector(collaborators,
                                                   db_iterator,
                                                   fl_round,
                                                   collaborators_chosen_each_round,
                                                   collaborator_times_per_round):
    """
    Selects collaborators for the given round using Ant Colony Optimization (ACO).

    Args:
        collaborators (list): List of collaborator names.
        db_iterator (iterator): Iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray'].
        fl_round (int): Current round number.
        collaborators_chosen_each_round (dict): Dictionary of {round: list of collaborators}.
        collaborator_times_per_round (dict): Dictionary of {round: {collaborator: total_time_taken_in_round}}.
        num_ants (int): Number of ants in each iteration.
        num_iterations (int): Number of iterations to run the optimization.
        pheromone_decay (float): Decay factor for pheromone trails (0 < pheromone_decay < 1).
        alpha (float): Importance of pheromone strength.
        beta (float): Importance of collaborator attributes.

    Returns:
        list: Selected collaborators for the current round.
    """
    print('ant_colony_optimization_collaborator_selector called')

    num_ants=50
    num_iterations=100
    pheromone_decay=0.5
    alpha=1
    beta=2
    # Step 1: Collect metrics
    agg_dice_results = []
    agg_loss_results = []

    # Collect dice and loss results from the previous round
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    # Convert results to dictionaries
    dice_scores = dict(agg_dice_results)
    loss_scores = dict(agg_loss_results)

    # Sum up times for each collaborator across all rounds
    total_times = Counter()
    for round_data in collaborator_times_per_round.values():
        total_times.update(round_data)

    # Collect all selections and count frequencies
    all_selections = [col for values in collaborators_chosen_each_round.values() for col in values]
    frequency_counter = Counter(all_selections)

    # Prepare collaborator data
    collaborator_data = {}
    for collaborator in collaborators:
        dice_score = float(dice_scores.get(collaborator, 0))
        loss_score = float(loss_scores.get(collaborator, 0))
        total_time = float(total_times.get(collaborator, 0))
        frequency = float(frequency_counter.get(collaborator, 0))

        collaborator_data[collaborator] = {
            'dice': dice_score,
            'loss': loss_score,
            'time': total_time,
            'frequency': frequency
        }

    # Normalize the attributes for all collaborators
    def normalize_attribute(attr_name):
        attr_values = [data[attr_name] for data in collaborator_data.values()]
        max_value = max(attr_values) if attr_values else 1
        for data in collaborator_data.values():
            data[attr_name] = data[attr_name] / max_value if max_value else 0

    normalize_attribute('dice')
    normalize_attribute('loss')
    normalize_attribute('time')
    normalize_attribute('frequency')

    # Step 2: Initialize ACO Parameters
    num_to_select = max(1, int(len(collaborators) * 0.2))  # Select 20% of collaborators
    pheromone = {collaborator: 1.0 for collaborator in collaborators}  # Initial pheromone levels

    def heuristic_value(collaborator):
        """Compute heuristic value based on collaborator attributes."""
        data = collaborator_data[collaborator]
        return data['dice'] - data['loss'] + (1 - data['time']) + (1 - data['frequency'])

    def calculate_probabilities(ant_solution):
        """Calculate probabilities of selecting the next collaborator."""
        probabilities = {}
        for collaborator in collaborators:
            if collaborator not in ant_solution:
                pheromone_level = pheromone[collaborator] ** alpha
                heuristic = heuristic_value(collaborator) ** beta
                probabilities[collaborator] = pheromone_level * heuristic
        total = sum(probabilities.values())
        return {k: v / total for k, v in probabilities.items()} if total > 0 else {}

    # Step 3: Run ACO
    best_solution = None
    best_fitness = -float('inf')

    for iteration in range(num_iterations):
        all_solutions = []
        solution_fitness = []

        for ant in range(num_ants):
            ant_solution = []
            while len(ant_solution) < num_to_select:
                probabilities = calculate_probabilities(ant_solution)
                if probabilities:
                    next_collaborator = random.choices(
                        population=list(probabilities.keys()),
                        weights=list(probabilities.values())
                    )[0]
                    ant_solution.append(next_collaborator)
                else:
                    break

            # Calculate fitness of the ant's solution
            fitness = sum(heuristic_value(c) for c in ant_solution)
            all_solutions.append(ant_solution)
            solution_fitness.append(fitness)

            # Update the global best solution
            if fitness > best_fitness:
                best_solution = ant_solution
                best_fitness = fitness

        # Update pheromone levels
        for collaborator in pheromone:
            pheromone[collaborator] *= (1 - pheromone_decay)  # Apply pheromone decay
        for solution, fitness in zip(all_solutions, solution_fitness):
            for collaborator in solution:
                pheromone[collaborator] += fitness  # Add pheromone based on fitness

    print("Selected Collaborators:", best_solution)
    return best_solution


def imperialist_competitive_algorithm_collaborator_selector(collaborators,
                                                            db_iterator,
                                                            fl_round,
                                                            collaborators_chosen_each_round,
                                                            collaborator_times_per_round):
    """
    Selects collaborators for the given round using Imperialist Competitive Algorithm (ICA).

    Args:
        collaborators (list): List of collaborator names.
        db_iterator (iterator): Iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray'].
        fl_round (int): Current round number.
        collaborators_chosen_each_round (dict): Dictionary of {round: list of collaborators}.
        collaborator_times_per_round (dict): Dictionary of {round: {collaborator: total_time_taken_in_round}}.
        num_empires (int): Number of initial empires.
        max_generations (int): Maximum number of generations to run the optimization.
        assimilation_factor (float): Proportion of assimilation towards imperialist solutions (0 < assimilation_factor < 1).
        revolution_probability (float): Probability of revolutionary changes in colonies (0 < revolution_probability < 1).
        competition_rate (float): Rate at which weaker empires lose colonies (0 < competition_rate < 1).

    Returns:
        list: Selected collaborators for the current round.
    """
    print('imperialist_competitive_algorithm_collaborator_selector called')
    num_empires=5
    max_generations=50
    assimilation_factor=0.1
    revolution_probability=0.2
    competition_rate=0.1
    # Step 1: Collect metrics
    agg_dice_results = []
    agg_loss_results = []

    # Collect dice and loss results from the previous round
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    # Convert results to dictionaries
    dice_scores = dict(agg_dice_results)
    loss_scores = dict(agg_loss_results)

    # Sum up times for each collaborator across all rounds
    total_times = Counter()
    for round_data in collaborator_times_per_round.values():
        total_times.update(round_data)

    # Collect all selections and count frequencies
    all_selections = [col for values in collaborators_chosen_each_round.values() for col in values]
    frequency_counter = Counter(all_selections)

    # Prepare collaborator data
    collaborator_data = {}
    for collaborator in collaborators:
        dice_score = float(dice_scores.get(collaborator, 0))
        loss_score = float(loss_scores.get(collaborator, 0))
        total_time = float(total_times.get(collaborator, 0))
        frequency = float(frequency_counter.get(collaborator, 0))

        collaborator_data[collaborator] = {
            'dice': dice_score,
            'loss': loss_score,
            'time': total_time,
            'frequency': frequency
        }

    # Normalize the attributes for all collaborators
    def normalize_attribute(attr_name):
        attr_values = [data[attr_name] for data in collaborator_data.values()]
        max_value = max(attr_values) if attr_values else 1
        for data in collaborator_data.values():
            data[attr_name] = data[attr_name] / max_value if max_value else 0

    normalize_attribute('dice')
    normalize_attribute('loss')
    normalize_attribute('time')
    normalize_attribute('frequency')

    # Step 2: Fitness Function
    def fitness(solution):
        """Compute fitness for a subset of collaborators."""
        total_dice = sum(collaborator_data[col]['dice'] for col in solution)
        total_loss = sum(collaborator_data[col]['loss'] for col in solution)
        total_time = sum(collaborator_data[col]['time'] for col in solution)
        total_frequency = sum(collaborator_data[col]['frequency'] for col in solution)
        return (total_dice - total_loss) / (1 + total_time + total_frequency)

    # Step 3: Initialize Empires
    num_to_select = max(1, int(len(collaborators) * 0.2))  # Select 20% of collaborators
    initial_colonies = [random.sample(collaborators, num_to_select) for _ in range(num_empires * 3)]
    empires = []

    for i in range(num_empires):
        if not initial_colonies:  # Safeguard against empty initial_colonies
            break
        imperialist = initial_colonies.pop(random.randint(0, len(initial_colonies) - 1))
        num_colonies = min(3, len(initial_colonies))  # Ensure we don't exceed available colonies
        colonies = [initial_colonies.pop(random.randint(0, len(initial_colonies) - 1)) for _ in range(num_colonies)]
        empires.append({
            'imperialist': imperialist,
            'colonies': colonies,
            'fitness': fitness(imperialist)
        })

    # Step 4: Evolve Empires
    for generation in range(max_generations):
        for empire in empires:
            # Assimilation: Move colonies towards imperialist
            for colony in empire['colonies']:
                for i in range(len(colony)):
                    if random.random() < assimilation_factor:
                        colony[i] = empire['imperialist'][i]

            # Revolution: Random changes in colonies
            for colony in empire['colonies']:
                if random.random() < revolution_probability:
                    replaced = random.choice(colony)
                    available = list(set(collaborators) - set(colony))
                    if available:
                        colony[colony.index(replaced)] = random.choice(available)

            # Update imperialist if a colony outperforms it
            for colony in empire['colonies']:
                if fitness(colony) > empire['fitness']:
                    empire['imperialist'] = colony
                    empire['fitness'] = fitness(colony)

        # Imperialistic Competition: Weak empires lose colonies
        weakest_empire = min(empires, key=lambda x: x['fitness'])
        strongest_empire = max(empires, key=lambda x: x['fitness'])
        if random.random() < competition_rate and weakest_empire['colonies']:
            lost_colony = weakest_empire['colonies'].pop(random.randint(0, len(weakest_empire['colonies']) - 1))
            strongest_empire['colonies'].append(lost_colony)

        # Remove weak empires with no colonies
        empires = [empire for empire in empires if empire['colonies']]

    # Step 5: Select Best Imperialist
    best_empire = max(empires, key=lambda x: x['fitness'])
    print("Selected Collaborators:", best_empire['imperialist'])
    return best_empire['imperialist']



def stochastic_diffusion_search_collaborator_selector(collaborators,
                                                       db_iterator,
                                                       fl_round,
                                                       collaborators_chosen_each_round,
                                                       collaborator_times_per_round):
    """
    Selects collaborators for the given round using Stochastic Diffusion Search (SDS).

    Args:
        collaborators (list): List of collaborator names.
        db_iterator (iterator): Iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray'].
        fl_round (int): Current round number.
        collaborators_chosen_each_round (dict): Dictionary of {round: list of collaborators}.
        collaborator_times_per_round (dict): Dictionary of {round: {collaborator: total_time_taken_in_round}}.
        num_agents (int): Number of agents in the population.
        max_iterations (int): Maximum number of iterations to run the search.
        test_probability (float): Probability of performing an independent evaluation (0 < test_probability < 1).

    Returns:
        list: Selected collaborators for the current round.
    """
    print('stochastic_diffusion_search_collaborator_selector called')

    num_agents=50
    max_iterations=100
    test_probability=0.2

    # Step 1: Collect metrics
    agg_dice_results = []
    agg_loss_results = []

    # Collect dice and loss results from the previous round
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    # Convert results to dictionaries
    dice_scores = dict(agg_dice_results)
    loss_scores = dict(agg_loss_results)

    # Sum up times for each collaborator across all rounds
    total_times = Counter()
    for round_data in collaborator_times_per_round.values():
        total_times.update(round_data)

    # Collect all selections and count frequencies
    all_selections = [col for values in collaborators_chosen_each_round.values() for col in values]
    frequency_counter = Counter(all_selections)

    # Prepare collaborator data
    collaborator_data = {}
    for collaborator in collaborators:
        dice_score = float(dice_scores.get(collaborator, 0))
        loss_score = float(loss_scores.get(collaborator, 0))
        total_time = float(total_times.get(collaborator, 0))
        frequency = float(frequency_counter.get(collaborator, 0))

        collaborator_data[collaborator] = {
            'dice': dice_score,
            'loss': loss_score,
            'time': total_time,
            'frequency': frequency
        }

    # Normalize the attributes for all collaborators
    def normalize_attribute(attr_name):
        attr_values = [data[attr_name] for data in collaborator_data.values()]
        max_value = max(attr_values) if attr_values else 1
        for data in collaborator_data.values():
            data[attr_name] = data[attr_name] / max_value if max_value else 0

    normalize_attribute('dice')
    normalize_attribute('loss')
    normalize_attribute('time')
    normalize_attribute('frequency')

    # Step 2: Fitness Function
    def fitness(solution):
        """Compute fitness for a subset of collaborators."""
        total_dice = sum(collaborator_data[col]['dice'] for col in solution)
        total_loss = sum(collaborator_data[col]['loss'] for col in solution)
        total_time = sum(collaborator_data[col]['time'] for col in solution)
        total_frequency = sum(collaborator_data[col]['frequency'] for col in solution)
        return (total_dice - total_loss) / (1 + total_time + total_frequency)

    # Step 3: Initialize Agents
    num_to_select = max(1, int(len(collaborators) * 0.2))  # Select 20% of collaborators
    agents = [random.sample(collaborators, num_to_select) for _ in range(num_agents)]
    agent_scores = [fitness(agent) for agent in agents]

    # Step 4: Iterative Search
    for iteration in range(max_iterations):
        for i in range(num_agents):
            if random.random() < test_probability:
                # Perform independent evaluation
                new_solution = random.sample(collaborators, num_to_select)
                new_score = fitness(new_solution)
                if new_score > agent_scores[i]:
                    agents[i] = new_solution
                    agent_scores[i] = new_score
            else:
                # Imitate another agent
                peer_idx = random.randint(0, num_agents - 1)
                if agent_scores[peer_idx] > agent_scores[i]:
                    agents[i] = agents[peer_idx]
                    agent_scores[i] = agent_scores[peer_idx]

    # Step 5: Select Best Agent
    best_agent_idx = np.argmax(agent_scores)
    best_solution = agents[best_agent_idx]

    print("Selected Collaborators:", best_solution)
    return best_solution



def q_learning_collaborator_selector(collaborators,
                                      db_iterator,
                                      fl_round,
                                      collaborators_chosen_each_round,
                                      collaborator_times_per_round):
    """
    Selects collaborators for the given round using Q-learning.

    Args:
        collaborators (list): List of collaborator names.
        db_iterator (iterator): Iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray'].
        fl_round (int): Current round number.
        collaborators_chosen_each_round (dict): Dictionary of {round: list of collaborators}.
        collaborator_times_per_round (dict): Dictionary of {round: {collaborator: total_time_taken_in_round}}.
        num_episodes (int): Number of episodes for training.
        learning_rate (float): Learning rate for Q-value updates.
        discount_factor (float): Discount factor for future rewards.
        exploration_rate (float): Probability of exploring random actions.

    Returns:
        list: Selected collaborators for the current round.
    """
    print('q_learning_collaborator_selector called')

    # Step 1: Collect metrics
    agg_dice_results = []
    agg_loss_results = []
    num_episodes=100
    learning_rate=0.1
    discount_factor=0.9
    exploration_rate=0.2
    # Step 1: Collect metrics
    agg_dice_results = []
    agg_loss_results = []

    # Collect dice and loss results from the previous round
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    # Convert results to dictionaries
    dice_scores = dict(agg_dice_results)
    loss_scores = dict(agg_loss_results)

    # Sum up times for each collaborator across all rounds
    total_times = Counter()
    for round_data in collaborator_times_per_round.values():
        total_times.update(round_data)

    # Collect all selections and count frequencies
    all_selections = [col for values in collaborators_chosen_each_round.values() for col in values]
    frequency_counter = Counter(all_selections)

    # Prepare collaborator data
    collaborator_data = {}
    for collaborator in collaborators:
        dice_score = float(dice_scores.get(collaborator, 0))
        loss_score = float(loss_scores.get(collaborator, 0))
        total_time = float(total_times.get(collaborator, 0))
        frequency = float(frequency_counter.get(collaborator, 0))

        collaborator_data[collaborator] = {
            'dice': dice_score,
            'loss': loss_score,
            'time': total_time,
            'frequency': frequency
        }

    # Normalize the attributes for all collaborators
    def normalize_attribute(attr_name):
        attr_values = [data[attr_name] for data in collaborator_data.values()]
        max_value = max(attr_values) if attr_values else 1
        for data in collaborator_data.values():
            data[attr_name] = data[attr_name] / max_value if max_value else 0

    normalize_attribute('dice')
    normalize_attribute('loss')
    normalize_attribute('time')
    normalize_attribute('frequency')

    # Step 2: Fitness Function
    def fitness(solution):
        """Compute fitness for a subset of collaborators."""
        total_dice = sum(collaborator_data[col]['dice'] for col in solution)
        total_loss = sum(collaborator_data[col]['loss'] for col in solution)
        total_time = sum(collaborator_data[col]['time'] for col in solution)
        total_frequency = sum(collaborator_data[col]['frequency'] for col in solution)
        return (total_dice - total_loss) / (1 + total_time + total_frequency)

    # Step 3: Initialize Q-learning
    num_to_select = max(1, int(len(collaborators) * 0.2))  # Select 20% of collaborators
    q_table = defaultdict(lambda: defaultdict(float))  # Q-values for state-action pairs
    actions = ["add", "replace"]  # Possible actions

    # Helper functions for Q-learning
    def get_action(current_solution):
        """Choose an action based on exploration or exploitation."""
        if random.random() < exploration_rate:
            return random.choice(actions)
        else:
            action_values = {act: q_table[str(current_solution)][act] for act in actions}
            return max(action_values, key=action_values.get)

    def perform_action(current_solution, action):
        """Perform the chosen action and return the new solution."""
        if action == "add" and len(current_solution) < num_to_select:
            available = list(set(collaborators) - set(current_solution))
            if available:
                current_solution.append(random.choice(available))
        elif action == "replace" and len(current_solution) > 0:
            replaced = random.choice(current_solution)
            available = list(set(collaborators) - set(current_solution))
            if available:
                current_solution[current_solution.index(replaced)] = random.choice(available)
        return current_solution

    # Training the Q-learning agent
    best_solution = None
    best_fitness = -float("inf")

    for episode in range(num_episodes):
        current_solution = random.sample(collaborators, num_to_select)
        for step in range(num_to_select):
            action = get_action(current_solution)
            next_solution = perform_action(current_solution[:], action)

            # Calculate reward and update Q-value
            reward = fitness(next_solution)
            old_q_value = q_table[str(current_solution)][action]
            best_future_q = max(q_table[str(next_solution)].values(), default=0)
            q_table[str(current_solution)][action] = old_q_value + learning_rate * (reward + discount_factor * best_future_q - old_q_value)

            # Update best solution if needed
            if reward > best_fitness:
                best_solution = next_solution
                best_fitness = reward

            current_solution = next_solution

    print("Selected Collaborators:", best_solution)
    return best_solution


def kwta_collaborator_selector(collaborators,
                               db_iterator,
                               fl_round,
                               collaborators_chosen_each_round,
                               collaborator_times_per_round):
    """
    Selects collaborators for the given round using k-Winner-Take-All (kWTA).

    Args:
        collaborators (list): List of collaborator names.
        db_iterator (iterator): Iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray'].
        fl_round (int): Current round number.
        collaborators_chosen_each_round (dict): Dictionary of {round: list of collaborators}.
        collaborator_times_per_round (dict): Dictionary of {round: {collaborator: total_time_taken_in_round}}.
        k (int): Number of top collaborators to select.
        max_iterations (int): Maximum number of iterations for refinement.

    Returns:
        list: Selected collaborators for the current round.
    """
    print('kwta_collaborator_selector called')
    k=5
    max_iterations=100
    # Step 1: Collect metrics
    agg_dice_results = []
    agg_loss_results = []

    # Collect dice and loss results from the previous round
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    # Convert results to dictionaries
    dice_scores = dict(agg_dice_results)
    loss_scores = dict(agg_loss_results)

    # Sum up times for each collaborator across all rounds
    total_times = Counter()
    for round_data in collaborator_times_per_round.values():
        total_times.update(round_data)

    # Collect all selections and count frequencies
    all_selections = [col for values in collaborators_chosen_each_round.values() for col in values]
    frequency_counter = Counter(all_selections)

    # Prepare collaborator data
    collaborator_data = {}
    for collaborator in collaborators:
        dice_score = float(dice_scores.get(collaborator, 0))
        loss_score = float(loss_scores.get(collaborator, 0))
        total_time = float(total_times.get(collaborator, 0))
        frequency = float(frequency_counter.get(collaborator, 0))

        collaborator_data[collaborator] = {
            'dice': dice_score,
            'loss': loss_score,
            'time': total_time,
            'frequency': frequency
        }

    # Normalize the attributes for all collaborators
    def normalize_attribute(attr_name):
        attr_values = [data[attr_name] for data in collaborator_data.values()]
        max_value = max(attr_values) if attr_values else 1
        for data in collaborator_data.values():
            data[attr_name] = data[attr_name] / max_value if max_value else 0

    normalize_attribute('dice')
    normalize_attribute('loss')
    normalize_attribute('time')
    normalize_attribute('frequency')

    # Step 2: Scoring Function
    def score(collaborator):
        """Compute a score for a single collaborator."""
        data = collaborator_data[collaborator]
        return data['dice'] - data['loss'] + (1 - data['time']) + (1 - data['frequency'])

    # Step 3: Initialize kWTA Selection
    scores = {collaborator: score(collaborator) for collaborator in collaborators}
    selected_collaborators = sorted(scores, key=scores.get, reverse=True)[:k]  # Initial top-k selection

    # Step 4: Iterative Refinement
    for iteration in range(max_iterations):
        # Explore by replacing a collaborator with a new candidate
        for i in range(k):
            new_candidate = random.choice(list(set(collaborators) - set(selected_collaborators)))
            temp_selection = selected_collaborators[:]
            temp_selection[i] = new_candidate

            # Compute the score for the new subset
            temp_score = sum(scores[collab] for collab in temp_selection)
            current_score = sum(scores[collab] for collab in selected_collaborators)

            # Accept the new subset if it performs better
            if temp_score > current_score:
                selected_collaborators = temp_selection

    print("Selected Collaborators:", selected_collaborators)
    return selected_collaborators


def knowledge_graph_collaborator_selector(collaborators, db_iterator, fl_round, collaborators_chosen_each_round,
                                          collaborator_times_per_round):
    """Selects collaborators for the given round using a knowledge graph approach.

    Args:
        collaborators: list of strings of collaborator names.
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number.
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}.
            Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    print('knowledge_graph_collaborator_selector called')

    # Initialize result containers
    agg_dice_results = []
    agg_loss_results = []

    # Collect dice and loss results from previous round
    for record in db_iterator:
        if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3:
            if record['tensor_name'] == 'valid_dice':
                agg_dice_results.append((record['tags'][0], record['nparray']))
            elif record['tensor_name'] == 'valid_loss':
                agg_loss_results.append((record['tags'][0], record['nparray']))

    # Convert results to NumPy arrays
    colab_numbers_dice, agg_valid_dice = zip(*agg_dice_results) if agg_dice_results else ([], [])
    colab_numbers_loss, agg_valid_loss = zip(*agg_loss_results) if agg_loss_results else ([], [])

    # Sum up times for each collaborator across all rounds
    total_times = Counter()
    for round_data in collaborator_times_per_round.values():
        total_times.update(round_data)

    # Collect all selections and count frequencies
    all_selections = [col for values in collaborators_chosen_each_round.values() for col in values]
    frequency_counter = Counter(all_selections)

    # Initialize a graph
    G = nx.Graph()

    # Add nodes for each collaborator with attributes
    for collaborator in collaborators:
        dice_score = float(dict(zip(colab_numbers_dice, agg_valid_dice)).get(collaborator, 0))
        loss_score = float(dict(zip(colab_numbers_loss, agg_valid_loss)).get(collaborator, 0))
        total_time = float(total_times.get(collaborator, 0))
        frequency = float(frequency_counter.get(collaborator, 0))

        # Normalize the attributes (for simplicity)
        normalized_dice = dice_score / max(agg_valid_dice) if agg_valid_dice else 0
        normalized_loss = loss_score / max(agg_valid_loss) if agg_valid_loss else 0
        normalized_time = total_time / max(total_times.values()) if total_times else 0
        normalized_frequency = frequency / max(frequency_counter.values()) if frequency_counter else 0

        # Add the collaborator as a node with attributes
        G.add_node(collaborator, dice=normalized_dice, loss=normalized_loss,
                   time=normalized_time, frequency=normalized_frequency)

    # Add edges based on similarity in attributes (e.g., similar dice and loss)
    for col1 in collaborators:
        for col2 in collaborators:
            if col1 != col2:
                # Calculate similarity as a simple inverse Euclidean distance between attributes
                similarity = np.linalg.norm(
                    np.array([G.nodes[col1]['dice'], G.nodes[col1]['loss'], G.nodes[col1]['time'], G.nodes[col1]['frequency']]) -
                    np.array([G.nodes[col2]['dice'], G.nodes[col2]['loss'], G.nodes[col2]['time'], G.nodes[col2]['frequency']])
                )
                # Add an edge if the similarity is within a threshold
                if similarity < 0.5:  # Arbitrary threshold for adding an edge
                    G.add_edge(col1, col2, weight=1 - similarity)

    # Rank nodes based on centrality (more central nodes are more important)
    centrality = nx.degree_centrality(G)
    ranked_collaborators = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    # Select top 20% of collaborators based on centrality
    num_to_select = int(len(collaborators) * 0.2)
    selected_collaborators = [collaborator for collaborator, score in ranked_collaborators[:num_to_select]]

    # Fallback: random selection if no data to process
    if not selected_collaborators:
        selected_collaborators = random.sample(collaborators, num_to_select)

    print("Selected Collaborators:", selected_collaborators)
###########
    # Visualize the Knowledge Graph and save as PDF
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout for better visualization
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), alpha=0.3, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Draw nodes with different sizes based on centrality
    node_sizes = [centrality[node] * 2000 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='orange', alpha=0.7)

    plt.title(f'Knowledge Graph of Collaborators for Round {fl_round}')
    pdf_filename = f'knowledge_graph_round_{fl_round}.pdf'
    plt.savefig(f'/scratch/project_2005952/DISH/Challenge/Task_1/{pdf_filename}', format='pdf')
    plt.close()

##########
    return selected_collaborators

def lp_collaborator_selector(collaborators, db_iterator, fl_round, collaborators_chosen_each_round, collaborator_times_per_round):
    """
    Selects collaborators for the given round using Linear Programming (LP).

    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}.
            Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    print('collaborators_chosen_each_round', collaborators_chosen_each_round)
    print('collaborator_times_per_round', collaborator_times_per_round)
    
    num_collaborators = len(collaborators)
    print('num_collaborators', num_collaborators)
    
    # If this is not the initial round or a special round
    if fl_round > 0 or fl_round % 5 != 0:
        print('lp_collaborator_selector called')
        
        # Aggregate results for the previous round
        AGG_RESULTS = []
        for record in db_iterator:
            if record['round'] == fl_round - 1 and 'validate_agg' in record['tags'] and len(record['tags']) == 3 and record['tensor_name'] == 'valid_dice':
                AGG_RESULTS.append((record['tags'][0], record['nparray']))
        print('AGG_RESULTS', AGG_RESULTS)
        
        # Extract collaborator numbers and their associated rewards
        colab_numbers = [t[0] for t in AGG_RESULTS]
        agg_valid_dice = [t[1] for t in AGG_RESULTS]
        array = np.array(list(zip(colab_numbers, agg_valid_dice)))
        
        # Define rewards as the valid_dice values (objective function)
        rewards = np.array(agg_valid_dice).astype(float)

        # Setup linear programming problem
        #c = -rewards  # Coefficients for the objective function (maximize reward, hence minimize -reward)
        # Determine if it's an even or odd round
        if fl_round % 2 == 0:
            print('Even round: Selecting best collaborators')
            # Maximize reward: Leave rewards as-is (maximize the positive values)
            c = -rewards  # Coefficients for the objective function (maximize reward, hence minimize -reward)
        else:
            print('Odd round: Selecting worst collaborators')
            # Minimize reward: Flip the rewards to prioritize the worst ones
            c = rewards  # Coefficients for the objective function (minimize reward directly)

        A_eq = np.ones((1, num_collaborators))  # Only one constraint: sum(x) == number of collaborators to select
        b_eq = [int(num_collaborators * 0.2)]  # Select 20% of the collaborators
        
        # Bounds for each collaborator selection variable (binary: either selected (1) or not (0))
        x_bounds = [(0, 1) for _ in range(num_collaborators)]

        # Solve the LP problem
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')

        # Output the results
        print('Optimal value:', result.fun)
        print('Optimal variables:', result.x)
        print('Solver status:', result.message)

        if result.success:
            selected_indices = np.where(result.x > 0.5)[0]  # LP may return fractional values; threshold at 0.5
            selected_collaborators = array[selected_indices, 0]
        else:
            # If optimization fails, fall back to random selection
            selected_collaborators = random.sample(collaborators, int(num_collaborators * 0.2))
        
        print('selected_collaborators', selected_collaborators)
        return selected_collaborators

    else:
        # If round is special (e.g., fl_round % 5 == 0), do random selection
        print('Random selection in special rounds')
        num_to_select = int(num_collaborators * 0.2)
        selected_collaborators = random.sample(collaborators, num_to_select)
        print("selected_collaborators", selected_collaborators)
        return selected_collaborators


#import numpy as np
#from scipy.stats import hmean

def quantum_amplitude_aggregation(tensor_values, weights):
    """
    Quantum-inspired amplitude-based aggregation.

    Args:
        tensor_values (list): List of tensor values (treated as quantum states).
        weights (list): Weights associated with each tensor.

    Returns:
        Aggregated tensor.
    """
    # Normalize weights to represent quantum probabilities
    total_weight = np.sum(weights)
    probabilities = np.array(weights) / (total_weight + 1e-5)

    # Create "amplitude states" from probabilities
    amplitudes = np.sqrt(probabilities)

    # Compute the superposition state
    superposition_state = np.sum([amp * tensor for amp, tensor in zip(amplitudes, tensor_values)], axis=0)

    # Interference-based adjustment (weighted sum with interference pattern)
    interference_pattern = np.abs(superposition_state)**2  # Mimic quantum measurement probabilities
    adjusted_tensor = np.average(tensor_values, axis=0, weights=interference_pattern)

    return adjusted_tensor

def sim_agg_weight_bias_hmean_quantum(local_tensors, tensor_db, tensor_name, fl_round, collaborators_chosen_each_round, collaborator_times_per_round):
    """
    Quantum-inspired aggregation of tensors using harmonic mean.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        tensor_db: Aggregator's TensorDB [writable]. Columns:
            ['tensor_name', 'origin', 'round', 'report', 'tags', 'nparray']
        tensor_name: Name of the tensor.
        fl_round: Round number.
        collaborators_chosen_each_round: Dictionary of {round: list of collaborators}.
        collaborator_times_per_round: Dictionary of {round: {collaborator: time}}.

    Returns:
        Aggregated tensor using quantum-inspired principles.
    """
    epsilon = 1e-5  # Small constant to prevent division by zero

    if tensor_name.endswith("weight") or tensor_name.endswith("bias"):
        tensor_values = np.array([t.tensor for t in local_tensors])
        weight_values = np.array([t.weight for t in local_tensors])

        # Compute the average of tensors
        total_average_of_tensors = np.average(tensor_values, axis=0)

        # Compute distances from the average tensor
        distances = []
        for tv in tensor_values:
            temp_abs = abs(my_sum(total_average_of_tensors) - my_sum(tv)) #np.linalg.norm(total_average_of_tensors - tv)
            distances.append(temp_abs)

        # Compute quantum-inspired weights based on distances
        weights = []
        total_distance = np.sum(distances)
        for i, dist in enumerate(distances):
            weights.append(total_distance / (epsilon + dist)) 
        #for dist in distances:
        #    weights.append(total_distance / (epsilon + dist))

        # Normalize weights
        weight_sum = np.sum(weights)
        weights_norm = []
        for i in range(len(weights)):
            weights_norm.append(weight_values[i] + weights[i] / (weight_sum + epsilon))

        # Normalize further to ensure proper distribution
        weight_sum_1 = np.sum(weights_norm)
        weights_norm_1 = []
        for i in range(len(weights_norm)):
            weights_norm_1.append(weights_norm[i] / (weight_sum_1 + epsilon))
        #weights_norm_1 = [w / (weight_sum_1 + epsilon) for w in weights_norm]

        # Quantum-inspired aggregation
        try:
            aggregated_tensor = quantum_amplitude_aggregation(tensor_values, weights_norm_1)
        except Exception as e:
            print(f"Error in quantum aggregation: {e}")
            try:
                # Fallback to harmonic mean
                aggregated_tensor = hmean(tensor_values, weights=weights_norm_1, axis=0)
            except Exception as e_inner:
                print(f"Error in harmonic mean aggregation: {e_inner}")
                # Final fallback to weighted average
                aggregated_tensor = np.average(tensor_values, weights=weights_norm_1, axis=0)

        return aggregated_tensor
    else:
        # Regular averaging for non-weight/bias tensors
        tensor_values = np.array([t.tensor for t in local_tensors])
        weight_values = np.array([t.weight for t in local_tensors])
        return np.average(tensor_values, weights=weight_values, axis=0)


def custom_percentage_collaborator_selector(collaborators,
                                   db_iterator,
                                   fl_round,
                                   collaborators_chosen_each_round,
                                   collaborator_times_per_round):
    """Chooses which collaborators will train for a given round.

    Args:
	collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    logger.info("one_collaborator_on_odd_rounds called!")
    original_array=collaborators
    #print('custom_percentage_collaborator_selector function called')
    percentage = 0.2
    windowsize = (len(collaborators)*percentage)

    collaborators=equal_partitions(collaborators,windowsize)

    pieces=np.ceil(len(collaborators)/windowsize) # how many pieces of collaborators
    print('pieces: ',pieces)

    if fl_round>=pieces: # fl_round > number of peices
        print('in if fl_round>=pieces',fl_round,'>=',pieces,' is TRUE')
        print('fl_round-int(fl_round/pieces)*pieces:',fl_round-(int(fl_round/pieces)*pieces))
        if (fl_round-(int(fl_round/pieces)*pieces)) == 0: #(c-int(c/l)*l) # to enable shuffle
            print('********************SHUFFLE YES********************')
            collaborators=original_array # reassigning to original array
            np.random.shuffle(collaborators) # shuffle
            collaborators=equal_partitions(collaborators,windowsize)
            # the array is saved in the file temp_colab.npy 
            np.save('temp_colab', collaborators)
        else:
            # to load from temp_colab.npy file
            # the array is loaded into b
            collaborators = np.load('temp_colab.npy')
    else:
        print('in else fl_round>=pieces',fl_round,'>=',pieces,' is false')

    print(collaborators)

    for i in range(fl_round*int(windowsize)%len(collaborators), len(collaborators), int(windowsize)): # 
        #print(i)
        #print(collaborators[i: i+int(windowsize)])
        return collaborators[i: i+int(windowsize)]

def equal_partitions(collaborators,windowsize):
  """
  function name: equal_partitions
  use:  to check if all collaborators have equal members
  """
  if len(collaborators)%int(windowsize)==0: # if all partitions have equal elements
      pass
  else: # if all partitions do not have equal elements, add elements from the start
      np_arr2=collaborators[0:int(windowsize)-(len(collaborators)%int(windowsize))]
      collaborators=np.append(collaborators, np_arr2)
  return collaborators



# # Running the Experiment
# 
# ```run_challenge_experiment``` is singular interface where your custom methods can be passed.
# 
# - ```aggregation_function```, ```choose_training_collaborators```, and ```training_hyper_parameters_for_round``` correspond to the [this list]
# (#Custom-hyperparameters-for-training) of configurable functions 
# described within this notebook.
# - ```institution_split_csv_filename``` : Describes how the data should be split between all collaborators. 
# Extended documentation about configuring the splits in the ```institution_split_csv_filename``` parameter can be found in the [README.md]
# (https://github.com/FETS-AI/Challenge/blob/main/Task_1/README.md). 
# - ```db_store_rounds``` : This parameter determines how long metrics and weights should be stored by the aggregator before being deleted. 
# Providing a value of `-1` will result in all historical data being retained, but memory usage will likely increase.
# - ```rounds_to_train``` : Defines how many rounds will occur in the experiment
# - ```device``` : Which device to use for training and validation

# ## Setting up the experiment
# Now that we've defined our custom functions, the last thing to do is to configure the experiment. The following cell shows the various settings you can change 
# in your experiment.
# 
# Note that ```rounds_to_train``` can be set as high as you want. However, the experiment will exit once the simulated time value exceeds 1 week of simulated time, 
# or if the specified number of rounds has completed.


# change any of these you wish to your custom functions. You may leave defaults if you wish.
aggregation_function = original_simagg_hmean #sim_agg_weight_bias_hmean_quantum #sim_agg_weight_bias_olympic_mean #sim_agg_weight_bias_hmean #B_sim_agg_suleimanbhai #sim_weighted_average_aggregation #Bayesian_SimAgg #weighted_average_aggregation#clipped_aggregation#weighted_average_aggregation#reg_sim_weighted_average_aggregation#weighted_average_aggregation#reg_sim_weighted_average_aggregation#weighted_average_aggregation
choose_training_collaborators = memetic_algorithm_collaborator_selector #kwta_collaborator_selector #q_learning_collaborator_selector #stochastic_diffusion_search_collaborator_selector #imperialist_competitive_algorithm_collaborator_selector #ant_colony_optimization_collaborator_selector #fuzzy_logic_collaborator_selector #simulated_annealing_collaborator_selector #memetic_algorithm_collaborator_selector #pso_collaborator_selector #genetic_algorithm_collaborator_selector #lp_collaborator_selector #knowledge_graph_collaborator_selector #lp_collaborator_selector #my_custom_ucb_rl_collaborator_selector #frequency_ucb_rl_collaborator_selector #my_custom_ucb_rl_collaborator_selector #greedy_epsilon_rl_collaborator_selector #ucb_rl_collaborator_selector #greedy_epsilon_rl_collaborator_selector #rl_collaborator_selector #custom_percentage_collaborator_selector_without_repetition #rl_collaborator_selector#custom_percentage_collaborator_selector_without_repetition#custom_percentage_collaborator_selector#all_collaborators_train
training_hyper_parameters_for_round = constant_hyper_parameters

# As mentioned in the 'Custom Aggregation Functions' section (above), six 
# perfomance evaluation metrics are included by default for validation outputs in addition 
# to those you specify immediately above. Changing the below value to False will change 
# this fact, excluding the three hausdorff measurements. As hausdorff distance is 
# expensive to compute, excluding them will speed up your experiments.
include_validation_with_hausdorff=False

# We encourage participants to experiment with partitioning_1 and partitioning_2, as well as to create
# other partitionings to test your changes for generalization to multiple partitionings.
#institution_split_csv_filename = 'partitioning_1.csv'
institution_split_csv_filename = '/scratch/project_2005952/MICCAI_FeTS2022_TrainingData/partitioning_2.csv' #'small_split.csv'

# change this to point to the parent directory of the data
brats_training_data_parent_dir = '/scratch/project_2005952/MICCAI_FeTS2022_TrainingData'

# increase this if you need a longer history for your algorithms
# decrease this if you need to reduce system RAM consumption
db_store_rounds = 20

# this is passed to PyTorch, so set it accordingly for your system
device = 'cuda' #'cpu'

# you'll want to increase this most likely. You can set it as high as you like, 
# however, the experiment will exit once the simulated time exceeds one week. 
rounds_to_train = 20

# (bool) Determines whether checkpoints should be saved during the experiment. 
# The checkpoints can grow quite large (5-10GB) so only the latest will be saved when this parameter is enabled
save_checkpoints = True

# path to previous checkpoint folder for experiment that was stopped before completion. 
# Checkpoints are stored in ~/.local/workspace/checkpoint, and you should provide the experiment directory 
# relative to this path (i.e. 'experiment_1'). Please note that if you restore from a checkpoint, 
# and save checkpoint is set to True, then the checkpoint you restore from will be subsequently overwritten.
# restore_from_checkpoint_folder = 'experiment_1'
restore_from_checkpoint_folder = None#'experiment_285'#None


# the scores are returned in a Pandas dataframe
scores_dataframe, checkpoint_folder = run_challenge_experiment(
    aggregation_function=aggregation_function,
    choose_training_collaborators=choose_training_collaborators,
    training_hyper_parameters_for_round=training_hyper_parameters_for_round,
    include_validation_with_hausdorff=include_validation_with_hausdorff,
    institution_split_csv_filename=institution_split_csv_filename,
    brats_training_data_parent_dir=brats_training_data_parent_dir,
    db_store_rounds=db_store_rounds,
    rounds_to_train=rounds_to_train,
    device=device,
    save_checkpoints=save_checkpoints,
    restore_from_checkpoint_folder = restore_from_checkpoint_folder)


scores_dataframe


# ## Produce NIfTI files for best model outputs on the validation set
# Now we will produce model outputs to submit to the leader board.
# 
# At the end of every experiment, the best model (according to average ET, TC, WT DICE) 
# is saved to disk at: ~/.local/workspace/checkpoint/\<checkpoint folder\>/best_model.pkl,
# where \<checkpoint folder\> is the one printed to stdout during the start of the 
# experiment (look for the log entry: "Created experiment folder experiment_##..." above).


from fets_challenge import model_outputs_to_disc
from pathlib import Path

# infer participant home folder
home = str(Path.home())

# you will need to specify the correct experiment folder and the parent directory for
# the data you want to run inference over (assumed to be the experiment that just completed)

#checkpoint_folder='experiment_1'
#data_path = </PATH/TO/CHALLENGE_VALIDATION_DATA>
data_path = '/scratch/project_2005952/MICCAI_FeTS2022_ValidationData'
validation_csv_filename = 'validation.csv'

# you can keep these the same if you wish
final_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'best_model.pkl')

# If the experiment is only run for a single round, use the temp model instead
if not Path(final_model_path).exists():
   final_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'temp_model.pkl')

outputs_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'model_outputs')


# Using this best model, we can now produce NIfTI files for model outputs 
# using a provided data directory

model_outputs_to_disc(data_path=data_path, 
                      validation_csv=validation_csv_filename,
                      output_path=outputs_path, 
                      native_model_path=final_model_path,
                      outputtag='',
                      device=device)
