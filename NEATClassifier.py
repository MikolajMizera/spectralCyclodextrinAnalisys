# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:00:09 2017

@author: user
"""
import os

import numpy as np
from sklearn.model_selection import LeaveOneOut

import neat
from neat.reporting import ReporterSet

class NEATClassifier:
    
    """
    Wrapper for neat library that can be used with sciki-learn pipelines
    
    """
    
    def __init__(self, generations = 100, population_size = 100,
                 scoring='accuracy', cv=5, subsample=1.0, n_jobs=1,
                 random_state=None, max_time_msec=45, verbosity=2):
        self.generations = generations
        self.population_size = population_size
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbosity = verbosity
        self.max_time_msec = max_time_msec
        self.best_genome = []
        local_dir = os.path.dirname(__file__)
        config_file = os.path.join(local_dir, 'config-neat.txt')
        self.config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
        self.p = neat.Population(self.config)
        self.p.reporters = ReporterSet()
        self.p.add_reporter(neat.StdOutReporter(True))
        self.p.add_reporter(neat.StatisticsReporter())
        
    def predict(self, X):
        pass
    
    def fit(self, X, y):
        loo = LeaveOneOut()
        loo_scores=[]
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index] 
            self.inputs=X_train
            self.targets=y_train        
            winner = self.p.run(self.__eval_genomes, self.generations)
            score = self.__test_genome(winner, X_test, y_test)
            loo_scores.append((winner, score))
        self.score=np.mean(loo_scores)    
        return self.score
    
    def __test_genome(self, genome, X, y):
        results = self.__simulate(genome,X)
        return np.mean(y==results)
        
    def __eval_genomes(self, genomes):
        for genome_id, genome in genomes.items():
            results = self.__simulate(genome, self.inputs)
            genome.fitness = np.mean(self.targets==results)
            
    def __compute_output(self, t0, t1, v0, v1):
        '''Compute the network's output based on the "time to first spike" of the two output neurons.'''
        if t0 is None or t1 is None or v0 is None or v1 is None:
            return 0.0
        else:
            #will return -1 or 1
            return np.sign(t0-t1)
    
    def __simulate(self, genome, inputs):
        # Create a network of "fast spiking" Izhikevich neurons.
        net = neat.iznn.IZNN.create(genome, self.config)
        dt = net.get_time_step_msec()
        y_pred=[]
        for idata in inputs:
            neuron_data = {}
            for i, n in net.neurons.items():
                neuron_data[i] = []
    
            # Reset the network, apply the XOR inputs, and run for the maximum allowed time.
            net.reset()
            net.set_inputs(idata)
            t0 = None
            t1 = None
            v0 = None
            v1 = None
            num_steps = int(self.max_time_msec / dt)
            net.set_inputs(idata)
            for j in range(num_steps):
                t = dt * j
                output = net.advance(dt)
                for i, n in net.neurons.items():
                    neuron_data[i].append((t, n.current, n.v, n.u, n.fired))
                if t0 is None and output[0] > 0:
                    t0, I0, v0, u0, f0 = neuron_data[net.outputs[0]][-2]
    
                if t1 is None and output[1] > 0:
                    t1, I1, v1, u1, f0 = neuron_data[net.outputs[1]][-2]
            response = self.__compute_output(t0, t1, v0, v1)
            y_pred.append(response)    
        return np.array(y_pred)
    
        