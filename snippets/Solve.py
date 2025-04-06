#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:27:27 2025

@author: lukejohnston
"""

def solve(self, times=None, solve_full=True, solve_xx=False, 
          solve_proj_chaotic=False, solve_full_chaotic=False,
          op_list=None, c_op_list=None, options=None, initial_state=None):
    """
    Solves the time evolution of the system for different Hamiltonians.
    
    Parameters:
        times (np.ndarray): Array of time points. Defaults to np.linspace(0, 10, 200) if not provided.
        solve_full (bool): If True, solve for the full Hamiltonian.
        solve_xx (bool): If True, solve for the XX Hamiltonian.
        solve_proj_chaotic (bool): If True, solve for the projected chaotic Hamiltonian.
        solve_full_chaotic (bool): If True, solve for the full chaotic Hamiltonian.
        op_list (list): List of operators for which expectation values are to be computed.
        c_op_list (list): List of collapse operators.
        options (dict): Solver options for mesolve.
        initial_state (Qobj or 'default'): The initial state for the time evolution. 
                                             If 'default' or None, self.initial_state is used.
    
    Returns:
        tuple: (times, results) where results is a dictionary mapping Hamiltonian type 
               (keys: 'full', 'xx', 'proj_chaotic', 'full_chaotic') to the corresponding 
               mesolve result.
    """
    # Set defaults for mutable arguments.
    if times is None:
        times = np.linspace(0, 10, 200)
    if op_list is None:
        op_list = []
    if c_op_list is None:
        c_op_list = []
    if options is None:
        options = {}
    if initial_state is None or initial_state == 'default':
        initial_state = self.initial_state

    results = {}
    # Map the Hamiltonian types to their corresponding objects based on flags.
    hams = {}
    if solve_full:
        hams['full'] = self.full_hamiltonian
    if solve_xx:
        hams['xx'] = self.base_hamiltonian
    if solve_proj_chaotic:
        hams['proj_chaotic'] = self.projected_chaotic_hamiltonian
    if solve_full_chaotic:
        hams['full_chaotic'] = self.full_chaotic_hamiltonian

    # Solve the time evolution for each Hamiltonian using QuTiP's mesolve.
    for key, H in hams.items():
        results[key] = mesolve(H, initial_state, times, c_op_list, op_list, options=options)
    
    # Save results to disk if required.
    if self.save_data:
        base_savename = f'{self.savepath}/Data/Solutions/N{self.n}_D{self.d}'
        for key, result in results.items():
            qsave(result, f'{base_savename}_{key}')
    
    return times, results
