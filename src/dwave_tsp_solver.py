from dwave.system.samplers import DWaveSampler           # Library to interact with the QPU
from dwave.system.composites import EmbeddingComposite   # Library to embed our problem onto the QPU physical graph
from dwave_qbsolv import QBSolv
import dwave_networkx.algorithms.tsp
import dwave.inspector
import itertools
import scipy.optimize
import TSP_utilities
import numpy as np

class DWaveTSPSolver(object):
    """
    Class for solving Travelling Salesman Problem using DWave.
    Specifying starting point is not implemented.
    """
    def __init__(self, distance_matrix, sapi_token=None, url=None):

        max_distance = np.max(np.array(distance_matrix))
        scaled_distance_matrix = distance_matrix / max_distance
        self.distance_matrix = scaled_distance_matrix
        self.constraint_constant = 400
        self.cost_constant = 10
        
        #Estos parametros son para hacer el QUBO
#        max_distance = np.max(np.array(distance_matrix))
#        self.distance_matrix = distance_matrix
#        self.constraint_constant = len(distance_matrix)*max_distance
#        self.cost_constant = 1
        
        #Estos parametros son para configurar el DWAVE
        self.chainstrength = 1000
        self.annealing_time = 400
        self.numruns = 2000
        self.sapi_token = sapi_token
        self.url = url
        
        #Estas funciones se ejecutan para llenar el QUBO
        self.qubo_dict = {}
        self.add_cost_objective()
        self.add_time_constraints()
        self.add_position_constraints()
        


#    def add_cost_objective(self):
#        n = len(self.distance_matrix)
#        for t in range(n):
#            for i in range(n):
#                for j in range(n):
#                    if i == j:
#                        continue
#                    qubit_a = t * n + i
#                    qubit_b = (t + 1)%n * n + j
#                    if qubit_b > qubit_a and (abs(i-j)==1 or (i==0 and j==n-1) or (i==n-1 and j==0)):                       
#                        for k in range(n-1-t):
#                            self.qubo_dict[(qubit_a, qubit_b+(k*n))] = self.cost_constant * self.distance_matrix[t][(t + 1)%n+k]
        
        
        
#### FUNCIONES PARA CALCULAR LA FUNCION QUBO ####

    def add_cost_objective(self):
        n = len(self.distance_matrix)
        for t in range(n):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    qubit_a = t * n + i
                    qubit_b = (t + 1)%n * n + j
                    self.qubo_dict[(qubit_a, qubit_b)] = self.cost_constant * self.distance_matrix[i][j]
                    
    def add_time_constraints(self):
        n = len(self.distance_matrix)
        for t in range(n):
            for i in range(n):
                qubit_a = t * n + i
                if (qubit_a, qubit_a) not in self.qubo_dict.keys():
                    self.qubo_dict[(qubit_a, qubit_a)] = -self.constraint_constant
                else:
                    self.qubo_dict[(qubit_a, qubit_a)] += -self.constraint_constant
                for j in range(n):
                    qubit_b = t * n + j
                    if i!=j and qubit_b > qubit_a:
                        self.qubo_dict[(qubit_a, qubit_b)] = 2 * self.constraint_constant


    def add_position_constraints(self):
        n = len(self.distance_matrix)
        for i in range(n):
            for t1 in range(n):
                qubit_a = t1 * n + i
                if (qubit_a, qubit_a) not in self.qubo_dict.keys():
                    self.qubo_dict[(qubit_a, qubit_a)] = -self.constraint_constant
                else:
                    self.qubo_dict[(qubit_a, qubit_a)] += -self.constraint_constant
                for t2 in range(n):
                    qubit_b = t2 * n + i
                    if t1!=t2 and qubit_b > qubit_a:
                        self.qubo_dict[(qubit_a, qubit_b)] = 2 * self.constraint_constant
                        
#### FUNCIONES PARA RESOLVER EL PROBLEMA QUBO ####

    def solve_tsp_QBSolv_Tabu(self):
        
        response = QBSolv().sample_qubo(self.qubo_dict, chain_strength=self.chainstrength, num_reads=self.numruns)             
       
        self.decode_solution(response)
        return self.solution, self.distribution
    
    def solve_tsp_QBSolv_DWAVE(self):
        
        sampler = EmbeddingComposite(DWaveSampler(token=self.sapi_token, endpoint=self.url, solver='DW_2000Q_6'))
        response = QBSolv().sample_qubo(self.qubo_dict, chain_strength=self.chainstrength, num_reads=self.numruns, solver=sampler)             
       
        self.decode_solution(response)
        return self.solution, self.distribution
    
    def solve_tsp_DWAVE(self):
        
        response = EmbeddingComposite(DWaveSampler(token=self.sapi_token, endpoint=self.url, solver='DW_2000Q_6')).sample_qubo(self.qubo_dict, chain_strength=self.chainstrength, num_reads=self.numruns)             
        
        #dwave.inspector.show(response)
        
        self.decode_solution(response)
        return self.solution, self.distribution
    
#### FUNCIONES DE POSTPROCESADO, PARA CALCULAR LA RUTA UNA VEZ DEVUELTA POR DWAVE ####

    def decode_solution(self, response):
        n = len(self.distance_matrix)
        distribution = {}
        min_energy = response.record[0].energy

        for record in response.record:
            sample = record[0]
            solution_binary = [node for node in sample] 
            solution = TSP_utilities.binary_state_to_points_order(solution_binary)
            distribution[tuple(solution)] = (record.energy, record.num_occurrences)
            if record.energy <= min_energy:
                self.solution = solution
        self.distribution = distribution


    def calculate_solution(self):
        """
        Samples the QVM for the results of the algorithm 
        and returns a list containing the order of nodes.
        """
        most_frequent_string, sampling_results = self.qaoa_inst.get_string(self.betas, self.gammas, samples=10000)
        reduced_solution = TSP_utilities.binary_state_to_points_order(most_frequent_string)
        full_solution = self.get_solution_for_full_array(reduced_solution)
        self.solution = full_solution
        
        all_solutions = sampling_results.keys()
        distribution = {}
        for sol in all_solutions:
            reduced_sol = TSP_utilities.binary_state_to_points_order(sol)
            full_sol = self.get_solution_for_full_array(reduced_sol)
            distribution[tuple(full_sol)] = sampling_results[sol]
        self.distribution = distribution
