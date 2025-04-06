#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:24:15 2025

@author: lukejohnston
"""

def create_pseudo_paulis(self):
    """
    Creates pseudo-Pauli matrices (sx, sy, sz) for the Hamiltonian.
    
    Returns:
        tuple: (sx, sy, sz) as Qobj.
    """
    zero = self.basis[self.coords[0]]
    one = self.basis[self.coords[1]]
    sx = zero * one.dag() + one * zero.dag()
    sy = -1j * zero * one.dag() + 1j * one * zero.dag()
    sz = zero * zero.dag() - one * one.dag()
    
    pauli_ops = [sx, sy, sz]
    if self.save_data:
        labels = ["sx", "sy", "sz"]
        for idx, op in enumerate(pauli_ops):
            savename = f'{self.savepath}/Data/Pseudo_Paulis/{labels[idx]}_N{self.n}_D{self.d}'
            qsave(op, savename)
    return sx, sy, sz


def nn_coupling_operators(self):
    """
    Creates nearest-neighbor coupling operators that act on a single site.
    For each site i, this returns operators of the form:
    
        σ_α^(i) = I ⊗ ... ⊗ σ_α ⊗ ... ⊗ I,   for α = x, y, z.
    
    Returns:
        tuple: (sx_list, sy_list, sz_list) as lists of Qobj.
    """
    sx, sy, sz = self.create_pseudo_paulis()
    sx_list, sy_list, sz_list = [], [], []
    for i in range(self.n):
        op_list = [qeye(self.d)] * self.n
        op_list[i] = sx
        sx_list.append(tensor(op_list))
        
        op_list = [qeye(self.d)] * self.n
        op_list[i] = sy
        sy_list.append(tensor(op_list))
        
        op_list = [qeye(self.d)] * self.n
        op_list[i] = sz
        sz_list.append(tensor(op_list))
    
    if self.save_data:
        for idx, op in enumerate(sx_list):
            qsave(op, f'{self.savepath}/Data/nn_coupling_operators/sx_op{idx}_N{self.n}_D{self.d}')
        for idx, op in enumerate(sy_list):
            qsave(op, f'{self.savepath}/Data/nn_coupling_operators/sy_op{idx}_N{self.n}_D{self.d}')
        for idx, op in enumerate(sz_list):
            qsave(op, f'{self.savepath}/Data/nn_coupling_operators/sz_op{idx}_N{self.n}_D{self.d}')
    
    return sx_list, sy_list, sz_list


def xx_hamiltonian(self):
    """
    Constructs the XX Hamiltonian:
    
      H_XX = -½ ∑₍i=0₎^(n-1) h_i σ_z^(i)
             + ½ ∑₍i=0₎^(n-1) J_i [σ_x^(i)σ_x^(i+1) + σ_y^(i)σ_y^(i+1)]
             
    Here σ_α^(i) is the tensor product operator with σ_α acting on site i and I on all others.
    Periodic boundary conditions (site n ≡ site 0) are assumed.
    
    Returns:
        Qobj: The XX Hamiltonian.
    """
    sx_list, sy_list, sz_list = self.nn_coupling_operators()
    
    H = 0
    # Magnetic term:
    for i in range(self.n):
        H -= 0.5 * self.h[i] * sz_list[i]
    
    # Interaction term: couple each site to its neighbor (periodic boundary conditions)
    for i in range(self.n):
        next_site = (i + 1) % self.n
        H += 0.5 * self.j[i] * (sx_list[i] * sx_list[next_site] + sy_list[i] * sy_list[next_site])
    
    if self.save_data:
        savename = f'{self.savepath}/Data/xx_hamiltonian/N{self.n}_D{self.d}'
        qsave(H, savename)
    return H