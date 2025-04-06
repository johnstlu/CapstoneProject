#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:23:21 2025

@author: lukejohnston
"""

def create_projectors(self, N_neighbors):
    """
    Creates projectors to isolate chaotic and integrable subspaces over N-nearest-neighbor sites.
    
    Parameters:
        N_neighbors (int): Number of adjacent qudits the projectors act on.
    
    Returns:
        tuple: (chaos_projectors, transfer_projectors) as matrices.
    """
    chaos_projectors = []
    transfer_projectors = []
    
    zero = self.basis[self.coords[0]]
    one = self.basis[self.coords[1]]
    identity = tensor([qeye(self.d)] * self.n)
    
    for n in range(self.n - N_neighbors + 1):
        # Ground state projector: all N sites in the block in |0>
        ground_list = [qeye(self.d)] * self.n
        for i in range(N_neighbors):
            ground_list[n + i] = zero * zero.dag()
        P_ground = tensor(ground_list)
        
        # Single-excitation projectors: exactly one site in |1> within the block.
        P_excited_list = []
        for i in range(N_neighbors):
            excited_list = [qeye(self.d)] * self.n
            for j in range(N_neighbors):
                excited_list[n + j] = zero * zero.dag()
            excited_list[n + i] = one * one.dag()
            P_excited_list.append(tensor(excited_list))
        
        
        
        P_transfer = P_ground + sum(P_excited_list)
        P_chaos = identity - P_transfer
        
        chaos_projectors.append(P_chaos)
        transfer_projectors.append(P_transfer)
    
    return chaos_projectors, transfer_projectors