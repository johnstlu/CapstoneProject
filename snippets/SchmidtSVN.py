#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:27:01 2025

@author: lukejohnston
"""

def schmidt_vn_entropy(self, eigenstates, compute_full=True, compute_xx=False,
                       compute_proj_chaotic=False, compute_full_chaotic=False, split_index='middle'):
    """
    Computes the Von Neumann entropy of eigenstates via Schmidt decomposition.
    
    For each eigenstate in the provided dictionary, this method computes the singular values
    using schmidt_SVD, squares them to obtain the Schmidt coefficients, and then computes the
    entropy as S_vn = -sum(λ_i * log(λ_i)).
    
    Parameters:
        eigenstates (dict): Dictionary containing lists of eigenstate Qobjs for different Hamiltonians.
                            Expected keys include 'full', 'xx', 'proj_chaotic', 'full_chaotic'.
        compute_full (bool): If True, compute for the full Hamiltonian.
        compute_xx (bool): If True, compute for the XX Hamiltonian.
        compute_proj_chaotic (bool): If True, compute for the projected chaotic Hamiltonian.
        compute_full_chaotic (bool): If True, compute for the full chaotic Hamiltonian.
        split_index (int or str): The bipartition index for Schmidt decomposition (default is 'middle').
        
    Returns:
        dict: A dictionary mapping each Hamiltonian type to a list of computed Von Neumann entropies.
    """
    vn_entropies = {}
    # Define which keys to process based on the boolean flags.
    key_flags = {
        'full': compute_full,
        'xx': compute_xx,
        'proj_chaotic': compute_proj_chaotic,
        'full_chaotic': compute_full_chaotic
    }
    
    for key, flag in key_flags.items():
        if flag and key in eigenstates:
            ent_list = []
            for state in eigenstates[key]:
                # Compute the singular values using the Schmidt decomposition.
                S = self.schmidt_SVD(state, split_index=split_index)
                # Square the singular values to obtain the Schmidt coefficients (λ_i).
                lambdas = S**2
                # Compute the logarithm in a safe way: log(λ) for λ > 0, and 0 for λ = 0.
                log_lambdas = np.where(lambdas > 0, np.log(lambdas), 0)
                # Compute the Von Neumann entropy: -∑ λ_i log(λ_i)
                vne = -np.sum(lambdas * log_lambdas)
                ent_list.append(vne)
            vn_entropies[key] = ent_list
    
    # Optionally save the results.
    if self.save_data:
        savename = f'{self.savepath}/Data/Entropies/Schmidt_N{self.n}_D{self.d}.pkl'
        qsave(vn_entropies, savename)
    
    return vn_entropies