# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:00:09 2017

@author: user
"""
import os

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer

import neat
from neat.reporting import ReporterSet

class NEATClassifier(BaseEstimator, ClassifierMixin):
    
    """
    Wrapper for neat library that can be used with sciki-learn pipelines
    
    """   
    
    def __init__(self, generations = 100, population_size = 100,
                 scoring='accuracy', n_jobs=1, warm_start=False,
                 max_time_msec=45, verbosity=2,
                 config_filename='config-iznn.txt',
                 use_scoop = False):
        #Scoring function dictionary initializaion
        self.scoring_fcns = {'accuracy': self.__acc}
        
        self.generations = generations
        self.population_size = population_size
        self.scoring_fcn = self.scoring_fcns[scoring]
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.max_time_msec = max_time_msec
        self.best_genome = []
        self.config_filename=config_filename
        self.use_scoop = use_scoop
        
        self.warm_start=warm_start
        self.winner = None
        self.p = None
        
    def predict(self, X):
        if not self.winner:
            raise RuntimeError('Estimator not fitted, call `fit` before exploiting the model')
        prediction = self.__simulate(self.winner, X)
        return prediction
    
    def fit(self, X, y):
        self.inputs=X
        self.targets=y
        self.__prepare_data()
        if not self.warm_start or self.p==None:
            self.__initialize_population()
        def eval_fcn(genomes, config):
            return self.__eval_genomes(genomes)
        
        if self.n_jobs>1:
            pe = neat.ParallelEvaluator(self.n_jobs, eval_fcn, use_scoop=self.use_scoop)
            self.winner = self.p.run(pe.evaluate, self.generations)
        else:            
            self.winner = self.p.run(eval_fcn, self.generations)

        self.score = self.winner.fitness
    
    def score(self, X, y):
        if not self.winner:
            raise RuntimeError('Estimator not fitted, call `fit` before exploiting the model')
        return self.__test_genome(self.winner, X, y)
    
    def __prepare_data(self):
        #Classes should be whole numbers    
        self.targets=np.round(self.targets).astype(int)
        #Check if shape of y indicate that output is binarized or vector of class numbers
        if (len(self.targets.shape) == 1):
            lb = LabelBinarizer(neg_label=0, pos_label=1)
            self.targets=lb.fit_transform(self.targets)
    
    def __initialize_population(self):
        local_dir = os.path.dirname(__file__)
        config_file = os.path.join(local_dir, self.config_filename)
        
        self.__modify_config(config_file, {'num_inputs':self.inputs.shape[-1],
                                           'num_outputs':self.targets.shape[-1]+1}) 
        self.config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
        self.p = neat.Population(self.config)
        self.p.reporters = ReporterSet()
        self.p.add_reporter(neat.StdOutReporter(True))
        self.p.add_reporter(neat.StatisticsReporter())
    
    def __modify_config(self, config_file, parameters):
        with open(config_file, 'r') as f:
            config_content=f.readlines()
        for i, l in enumerate(config_content):
            for k,p in parameters.items():
                try:
                    #extract parameter name from file and remove white signs
                    param_name=l.split('=')[0].split()[0]
                except:
                    continue
                if  param_name == k:
                    config_content[i] = ('%s\t= %s\n'%(k,p))
        with open(config_file, 'w') as f:
            f.write(''.join(config_content))
    
    def __test_genome(self, genome, X, y):
        y_pred = self.__simulate(genome,X)
        score = self.scoring_fcn(y, y_pred)
        return score
    
 
    def __eval_genomes(self, genomes):
        for genome_id, genome in genomes:
            y_pred = self.__simulate(genome, self.inputs)
            fitness = self.scoring_fcn(self.targets, y_pred)
            genome.fitness = fitness
            
    def __compute_output(self, tn, vn):
        '''Compute the network's output based on the "time to first spike" of the two output neurons.'''
        if np.any(np.array(tn)==None)  or np.any(np.array(vn)==None):
            return [0.0 for n in tn]
        else:
            #will return 0 or 1 for each otuput
            t0=tn[0]
            response = (t0-np.array(tn[1:]))>0
            return response.astype(int)
            
    def __simulate(self, genome, inputs):
        # Create a network of "fast spiking" Izhikevich neurons.
        net = neat.iznn.IZNN.create(genome, self.config)
        dt = net.get_time_step_msec()
        y_pred=[]
        for idata in inputs:
            neuron_data = {}
            for i, n in net.neurons.items():
                neuron_data[i] = []
            tn = [None]*len(neuron_data)
            vn =[None]*len(neuron_data)
    
            # Reset the network, apply the inputs, and run for the maximum allowed time.
            net.reset()
            net.set_inputs(idata)
            num_steps = int(self.max_time_msec / dt)
            net.set_inputs(idata)
            for j in range(num_steps):
                t = dt * j
                output = net.advance(dt)
                for i, n in net.neurons.items():
                    neuron_data[i].append((t, n.current, n.v, n.u, n.fired))
                    if tn[i]==None and output[i]>0:
                        tn[i]=neuron_data[net.outputs[0]][-2][0]
                        vn[i]=neuron_data[net.outputs[0]][-2][2]
                        
            response = self.__compute_output(tn, vn)
            y_pred.append(response)    
        return np.array(y_pred)
    
    def __acc(self, y, y_pred):
        return np.mean(y==y_pred)
        