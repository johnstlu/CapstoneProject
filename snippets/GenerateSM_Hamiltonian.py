#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:24:45 2025

@author: lukejohnston
"""

    def generate_random_hamiltonian(self, density, use_projectors=True):
        """
        Creates a random Hamiltonian acting on all N-nearest-neighbor blocks,
        with an option to project the operator onto the chaotic subspace.
        
        The chaotic Hamiltonian is defined as:
        H = sum(P_{n,n+1} * R_{n,n+1} * P_{n,n+1})
        where R_{n,n+1} is a random Hermitian matrix for the block of sites n,n+1,
        and P_{n,n+1} is the projector onto the chaotic subspace on the same sites.
        
        Parameters:
            density (float): Density of the random Hermitian matrices.
            N_neighbors (int): Number of qudits in each local (block) operator.
            use_projectors (bool): If True, project the Hamiltonian onto the chaotic subspace.
        
        Returns:
            tuple: If use_projectors is True, returns (projected_chaotic, full_chaotic);
                   otherwise, returns full_chaotic.
        """
        
        # Dense identity operator on one qudit.
        id_op = qeye(self.d)
        
        # Generate random Hermitian operators for each valid block,
        # and set their dims to reflect N_neighbors qudits (each of dimension self.d).
        random_hams = []
        for idx in range(self.n - self.N_neighbors + 1):
            A = rand_herm(self.d**self.N_neighbors, density=density)
            A.dims = [[self.d] * self.N_neighbors, [self.d] * self.N_neighbors]
            random_hams.append(A)
#            print(f"[DEBUG] Random Hamiltonian for block {idx} (shape {A.shape}):\n{A}\n")
        
        full_chaotic_terms = []
        for n in range(self.n - self.N_neighbors + 1):
            left = [id_op] * n
            right = [id_op] * (self.n - n - self.N_neighbors)
            # Build the full operator by embedding the block operator among identities.
            # The following concatenates the list: left + [random_hams[n]] + right,
            # and then computes the tensor product of all operators.
            full_op = tensor(left + [random_hams[n]] + right)
            full_chaotic_terms.append(full_op)
#            print(f"[DEBUG] Full operator for block {n} (shape {full_op.shape}):\n{full_op}\n")
        
        # Sum the list of full operators to get the full chaotic Hamiltonian.
        if full_chaotic_terms:
            full_chaotic = sum(full_chaotic_terms[1:], full_chaotic_terms[0])
        else:
            full_chaotic = None
#        print(f"[DEBUG] Full chaotic Hamiltonian (shape {full_chaotic.shape}):\n{full_chaotic}\n")
        
        if use_projectors:
            # Create the projectors. We assume create_projectors returns a list of projectors
            # for each block (and something else we ignore here).
            P_chaos, _ = self.create_projectors(self.N_neighbors)
#            for idx, proj in enumerate(P_chaos):
#                print(f"[DEBUG] Projector for block {idx} (shape {proj.shape}):\n{proj}\n")
            
            # Apply the projector from both left and right.
            projected_terms = [P_chaos[i] * full_chaotic_terms[i] * P_chaos[i]
                               for i in range(len(P_chaos))]
#            for idx, term in enumerate(projected_terms):
#                print(f"[DEBUG] Projected term for block {idx} (shape {term.shape}):\n{term}\n")
            
            if projected_terms:
                projected_chaotic = sum(projected_terms[1:], projected_terms[0])
            else:
                projected_chaotic = None
#            print(f"[DEBUG] Projected chaotic Hamiltonian (shape {projected_chaotic.shape}):\n{projected_chaotic}\n")
            
            return projected_chaotic, full_chaotic
        
        return full_chaotic