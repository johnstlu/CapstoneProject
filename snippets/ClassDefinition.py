#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:24:00 2025

@author: lukejohnston
"""

class QuditChain:
    def __init__(self, length, number_of_levels, density=1, N_neighbors=2, save_data=False, savepath=None, epsilon = 0):
        """
        Initializes an instance of the QuditChain class.
        
        Parameters:
            length (int): Length of the quantum chain.
            number_of_levels (int): Number of energy levels per qudit.
            density (float): Density of the random Hermitian matrices.
            N_neighbors (int): Number of nearest neighbors for interactions.
                               (2 for the XX Hamiltonian, 3 for the XXX Hamiltonian)
            save_data (bool): If True, save data to disk.
            savepath (str): Path to save data. (Default: './data')
            perturbation: strength of unprojected randomness
            
        """
        self.n = int(length)
        self.d = int(number_of_levels)
        self.N_neighbors = N_neighbors
        self.density = density
        self.save_data = save_data
        self.savepath = savepath if savepath is not None else './data'
        self.epsilon = epsilon
        
        # Coordinates for pseudo-Pauli operators (assume |0> and |1> are of interest)
        self.coords = [0, 1]
        # Initialize the basis states for a d-dimensional Hilbert space.
        self.basis = [basis(self.d, i) for i in range(self.d)]
        
        # Set interaction strengths and magnetic fields.
        self.j = np.array([np.sqrt(n * (self.n - n)) for n in range(1, self.n + 1)])
        self.j[0] *= 1+epsilon
        self.h = np.zeros(self.n)
        
        # Construct the base Hamiltonian:
        # - If self.N_neighbors == 2, an XX Hamiltonian is constructed.
        # - If self.N_neighbors == 3, an XXX Hamiltonian is constructed.
        self.base_hamiltonian = self.construct_hamiltonian()
        
        # Generate the chaotic Hamiltonians.
        self.projected_chaotic_hamiltonian, self.full_chaotic_hamiltonian = self.chaotic_hamiltonian(density=self.density)
        #self.full_hamiltonian = self.base_hamiltonian + self.projected_chaotic_hamiltonian + self.epsilon*self.full_chaotic_hamiltonian
        self.full_hamiltonian = self.construct_hamiltonian() + self.projected_chaotic_hamiltonian #+ self.epsilon*self.full_chaotic_hamiltonian
        
        # Create an initial state; here we simply assign a nonzero coefficient to the |1> state.
        coefficients = np.zeros(self.d)
        coefficients[1] = 1
        self.create_state(coefficients)