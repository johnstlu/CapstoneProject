#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:43:01 2025

@author: lukejohnston
"""
import matplotlib.pyplot as plt
import numpy as np
from qutip import (basis, qeye, tensor, rand_herm, mesolve, anim_matrix_histogram,
                   anim_fock_distribution, anim_qubism, entropy_vn, qzero, Qobj)
from qutip.fileio import (qload, qsave)
from matplotlib.animation import PillowWriter, FuncAnimation
from scipy.stats import poisson



class QutritCross:
    def __init__(self, N, j=1,density=1,N_neighbors=1, savepath=None):
        """
        Initializes the QutritCross system with N sites and constructs the Hamiltonian.
        
        Parameters:
        N (int): The number of sites (must be at least 5).
        """

        if N < 5:
            raise ValueError("The number of sites N must be at least 5.")
        
        self.n = N
        self.d = 3
        self.density = density
        self.N_neighbors=1
        
        self.j=j
        
        self.basis = [basis(self.d, i) for i in range(self.d)]
        
        self.savepath = savepath if savepath is not None else './data'
        self.LocalProjectiveMeasurement()
        self.InitialState()
        
        self.base_hamiltonian = self.construct_hamiltonian()
        # Generate the chaotic Hamiltonians.
        self.projected_chaotic_hamiltonian, self.full_chaotic_hamiltonian = self.chaotic_hamiltonian(density=self.density)
        #self.full_hamiltonian = self.base_hamiltonian + self.projected_chaotic_hamiltonian + self.epsilon*self.full_chaotic_hamiltonian
        self.full_hamiltonian = self.base_hamiltonian + self.projected_chaotic_hamiltonian
        
    def create_projectors(self, n):
        """
        Creates a single projector that isolates the chaotic subspace for the site pair (n, n+1).
    
        Parameters:
            n (int): The index of the first site in the pair (n, n+1) on which the projector acts.
    
        Returns:
            Qobj: The chaotic subspace projector for the given site pair.
    
        The chaotic subspace is defined as the complement of the transfer subspace, where:
            - The transfer subspace is spanned by states where both qudits are in the |+> or |−> states.
            - The chaotic projector is given by P_chaos = I - P_plus - P_minus.
    
        The returned projector acts on the full system but modifies only sites (n, n+1).
        """
        minus = self.basis[0]
        plus = self.basis[2]
        minusMinus = tensor([minus,minus])
        plusMinus = tensor([plus,minus])
        minusPlus = tensor([minus,plus])
        plusPlus = tensor([plus,plus])
        
        P = minusMinus*minusMinus.dag() + plusPlus*plusPlus.dag()
        
        Projector = tensor([qeye(3)]*(n-1) + [P] + [qeye(3)]*(self.n-n-1))
        return Projector

    def generate_random_hamiltonian(self, density, use_projectors=True):
        """
        Generates a random Hamiltonian with randomness only between sites 2,3 and 3,4.
        
        Parameters:
            density (float): Density of the random Hermitian matrices.
            use_projectors (bool): If True, projects the Hamiltonian onto the chaotic subspace.
        
        Returns:
            tuple: (projected_chaotic, full_chaotic) if use_projectors is True, otherwise full_chaotic.
        """
        id_op = qeye(self.d)
    
        # Generate random Hermitian matrices only for interactions (2,3) and (3,4)
        R_23 = rand_herm(self.d**2, density=density)
        R_23.dims = [[self.d, self.d], [self.d, self.d]]
    
        R_34 = rand_herm(self.d**2, density=density)
        R_34.dims = [[self.d, self.d], [self.d, self.d]]
    
        # Embed them in the full system
        H_23 = tensor([id_op] * 2 + [R_23] + [id_op] * (self.n - 4))
        H_34 = tensor([id_op] * 3 + [R_34] + [id_op] * (self.n - 5))
    
        # Full Hamiltonian as the sum of these two interactions
        full_chaotic = H_23 + H_34
    
        if use_projectors:
            P_23 = self.create_projectors(2)
            P_34 = self.create_projectors(3)
    
            # Projected Hamiltonian
            projected_chaotic = P_23 * H_23 * P_23 + P_34 * H_34 * P_34 
            return projected_chaotic, full_chaotic
    
        return full_chaotic
    
    def chaotic_hamiltonian(self, density, use_projectors=True):
        """
        Wrapper for generating the chaotic Hamiltonian.
        
        Returns:
            tuple: (projected_chaotic, full_chaotic) if use_projectors is True,
                   otherwise (None, full_chaotic).
        """
        result = self.generate_random_hamiltonian(density, use_projectors=use_projectors)
        if use_projectors:
            return result
        else:
            return None, result
    
    def permutation_operator(self, i, j):
        """
        Constructs the permutation operator swapping sites i and j in a 3-level system.

        Parameters:
        i (int): First site index (0 ≤ i < N).
        j (int): Second site index (0 ≤ j < N, i ≠ j).

        Returns:
        Qobj: The permutation operator as a QuTiP quantum object.
        """

        # Validate inputs
        if not (0 <= i < self.n) or not (0 <= j < self.n):
            raise ValueError(f"Indices i={i}, j={j} must be in range [0, {self.n-1}].")
        if i == j:
            raise ValueError("Indices n and m must be distinct.")
        if self.n < 2:
            raise ValueError("N must be at least 2.")

        # Initialize the permutation operator as a zero matrix
        pi = tensor([qzero(self.d)] * self.n)
        
        
        Weyls = np.zeros((self.d, self.d, self.d, self.d))
        for a in range(self.d):
            for b in range(self.d):
                matrix = np.zeros((self.d, self.d))  # Create a zero matrix
                matrix[b,a] = 1  # Set (i, j) element to 1
                Weyls[a,b] = matrix  # Store in precomputed array

        
        # Convert the Weyl basis matrices to Qobj and store them in a dictionary
        S_Qobjs = {(a,b): Qobj(Weyls[a,b]) for a in range(self.d) for b in range(self.d)}
        
        sites = [qeye(self.d)]*self.n
        # Loop through Weyl matrices and build the permutation operator
        for a in range(self.d):
            for b in range(self.d):
                sites[i] = S_Qobjs[(a,b)]
                sites[j] = S_Qobjs[(b,a)]
                pi += tensor(sites)
        return pi

    def construct_hamiltonian(self):
        """
        Constructs the Hamiltonian for the QutritCross system by summing the permutation operators 
        for adjacent sites based on the cross structure.
        
        Returns:
        Qobj: The Hamiltonian operator as a QuTiP quantum object.
        """
        
        hamiltonian = 0

        # Central site (site 3) is connected to sites 1, 2, 4, 5
        hamiltonian += self.permutation_operator(0, 2)  # site 1 and 3
        hamiltonian += self.permutation_operator(1, 2)  # site 2 and 3
        hamiltonian += self.permutation_operator(2, 3)  # site 3 and 4
        hamiltonian += self.permutation_operator(2, 4)  # site 3 and 5
        
        # Even numbered sites (N >= 4) connect to each other
        if self.n >= 4:
            for i in range(self.d, self.n, 2):  # Even indexed sites 4, 6, 8, ...
                if i + 2 < self.n:
                    hamiltonian += self.permutation_operator(i, i + 2)
        
        # Odd numbered sites (N >= 5) connect to each other
        if self.n >= 5:
            for i in range(4, self.n, 2):  # Odd indexed sites 5, 7, 9, ...
                if i + 2 < self.n:
                    hamiltonian += self.permutation_operator(i, i + 2)
        hamiltonian *= self.j
        return hamiltonian
    
    def InitialState(self):
        """
        Creates the initial state for the quantum chain.
        
        Parameters:
            coefficients (np.ndarray): Coefficients for the state superposition.
            loc (int): Location in the chain where the state is injected.
        """
        plus = self.basis[2]  # |+1⟩ state
        minus = self.basis[0]  # |-1⟩ state
        
        state_list = [self.basis[1]] * self.n
        state_list[0] = plus
        state_list[1] = minus
        initial_state = tensor(state_list)
        self.state = initial_state
        
    def LocalProjectiveMeasurement(self):
        """
        Constructs a local projective measurement operator for the last Alice and Bob.
        
        This function implements the one-measurement protocol described in Section II of the paper.
        It defines a measurement operator M that distinguishes between the qutrit states |±1⟩ and |0⟩,
        but does not differentiate between |+1⟩ and |-1⟩. The measurement is applied to the last two
        qutrits in the system, corresponding to Alice and Bob's positions in the protocol.
        
        The function:
        - Defines the basis states |+1⟩ and |-1⟩
        - Constructs the measurement operator M = |+1⟩⟨+1| + |-1⟩⟨-1|
        - This operator performs a coarse grained measurement of the system
        - Constructs the full-system measurement operator using a tensor product
        - Stores the computed operator in self.CPLM for later use
        
        The resulting operator can be used to test for the presence of entanglement through local measurements.
        """
        # Define the basis states for measurement
        plus = self.basis[2]  # |+1⟩ state
        minus = self.basis[0]  # |-1⟩ state
        
        # Construct the measurement operator M
        M = plus * plus.dag() + minus * minus.dag()
        
        # Initialize a list of identity operators for all qutrits
        measurement_list = [qeye(self.d)] * self.n
        
        # Replace the last two qutrits with the measurement operator M
        measurement_list[self.n-1] = M  # Alice's qutrit
        measurement_list[self.n-2] = M  # Bob's qutrit
        
        # Construct the full system measurement operator
        CPLM = tensor(measurement_list)
        
        # Store the computed measurement operator
        self.CPLM = CPLM
        
    def Evolve(self, times=None, solve_full=True, solve_xx=False, 
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
            initial_state = self.state
    
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
        
        return times, results
    
    def eigenstuff(self, compute_full=True, compute_xx=True, 
                   compute_proj_chaotic=True, compute_full_chaotic=True, 
                   return_evals=True, return_estates=True):
        """
        Computes eigenvalues and eigenstates for the Hamiltonians.
        
        Returns:
            tuple: (eigenvalues, eigenstates) as dictionaries.
        """
        eigenvalues, eigenstates = {}, {}
        if compute_full:
            evals, estates = self.full_hamiltonian.eigenstates()
            if return_evals:
                eigenvalues['full'] = np.array(evals)
            if return_estates:
                eigenstates['full'] = np.array(estates)
        
        if compute_xx:
            evals, estates = self.base_hamiltonian.eigenstates()
            if return_evals:
                eigenvalues['xx'] = np.array(evals)
            if return_estates:
                eigenstates['xx'] = np.array(estates)
        
        if compute_proj_chaotic and self.projected_chaotic_hamiltonian is not None:
            evals, estates = self.projected_chaotic_hamiltonian.eigenstates()
            if return_evals:
                eigenvalues['proj_chaotic'] = np.array(evals)
            if return_estates:
                eigenstates['proj_chaotic'] = np.array(estates)
        
        if compute_full_chaotic and self.full_chaotic_hamiltonian is not None:
            evals, estates = self.full_chaotic_hamiltonian.eigenstates()
            if return_evals:
                eigenvalues['full_chaotic'] = np.array(evals)
            if return_estates:
                eigenstates['full_chaotic'] = np.array(estates)
        
        if self.save_data:
            spath = f'{self.savepath}/Data/eigen_stuff/N{self.n}_D{self.d}'
            if return_evals:
                for key, evals in eigenvalues.items():
                    np.savetxt(f'{spath}_eigenvalues_{key}.csv', evals, delimiter=",",
                               header="# Eigenvalues [comma separated]")
        
        return eigenvalues, eigenstates
    
    def plot_operator_timeseries(self, times, results, save=False, 
                                 savename=None, Op_Names=None, Op_Symbols=None,
                                 Op_Indices=None, figsize=(20, 10), 
                                 show_full=True, show_xx=False, 
                                 show_proj_chaotic=False, show_full_chaotic=False,
                                 title='', legend=False, pst_lines=False):
        """
        Plots the time evolution of the expectation values of specified operators
        for the selected Hamiltonians.
        
        Parameters:
            times (np.ndarray): Array of time points.
            results (dict): Dictionary of mesolve results (keys such as 'full', 'xx', 
                            'proj_chaotic', 'full_chaotic').
            save (bool): If True, the plot is saved to disk.
            savename (str): Filename to save the plot (default: constructed from self.savepath).
            Op_Names (list or str): Names of the operators (or a single name).
            Op_Symbols (list or str): Symbols of the operators (or a single symbol).
            Op_Indices (list or int): Indices of the operators to plot (or a single index).
            figsize (tuple): Figure size.
            show_full (bool): If True, plot the 'full' Hamiltonian results.
            show_xx (bool): If True, plot the 'xx' Hamiltonian results.
            show_proj_chaotic (bool): If True, plot the 'proj_chaotic' results.
            show_full_chaotic (bool): If True, plot the 'full_chaotic' results.
            title (str): Title of the plot.
            legend (bool): If True, add a legend.
            pst_lines (bool): If True, add horizontal lines at y=0 and y=1.
            
        Returns:
            None
        """
    
        # Convert inputs to lists if they are not already.
        if isinstance(Op_Names, str):
            Op_Names = [Op_Names]
        if isinstance(Op_Symbols, str):
            Op_Symbols = [Op_Symbols]
        if isinstance(Op_Indices, int):
            Op_Indices = [Op_Indices]
    
        # Set default values if needed.
        if Op_Names is None:
            Op_Names = ["Operator"]
        if Op_Symbols is None:
            Op_Symbols = ["O"]
        if Op_Indices is None:
            Op_Indices = [0]
        if savename is None:
            savename = f'{self.savepath}/Plots/Operator_Timeseries_N{self.n}_D{self.d}.png'
        
        # Build a list of Hamiltonian keys to plot.
        ham_keys = []
        if show_full and 'full' in results:
            ham_keys.append('full')
        if show_xx and 'xx' in results:
            ham_keys.append('xx')
        if show_proj_chaotic and 'proj_chaotic' in results:
            ham_keys.append('proj_chaotic')
        if show_full_chaotic and 'full_chaotic' in results:
            ham_keys.append('full_chaotic')
        
        
        for key in ham_keys:
            print(f"For key '{key}', length of expectation list: {len(results[key].expect)}")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Loop over each operator (name, symbol, index) and plot the time series for each Hamiltonian.
        for name, symbol, index in zip(Op_Names, Op_Symbols, Op_Indices):
            for key in ham_keys:
                # Expectation values are assumed to be in the .expect attribute of the result.
                # Construct a label that indicates the operator and the Hamiltonian type.
                label = f"{name} ({key})"
                print(label)
                ax.plot(times, results[key].expect[index], label=label)
                
        
        # Optionally add horizontal lines (e.g., for PST indicators).
        if pst_lines:
            ax.axhline(0, color='r', linestyle=':', linewidth=1)
            ax.axhline(1, color='r', linestyle=':', linewidth=1)
        
        # Set plot labels and limits.
        ax.set_xlabel('Time', fontsize=20)
        ax.set_ylabel('Expectation Value', fontsize=20)
        ax.set_ylim([0, 1])
        if title:
            ax.set_title(title, fontsize=24)
        
        # Add legend if requested.
        if legend:
            ax.legend(fontsize=16, loc='lower right')

        if save:
            plt.savefig(savename, dpi=200, bbox_inches='tight')

        plt.show()
        plt.close()
        
    
    def eigenvalue_diffs(self, eigenvalues, compute_full=True, compute_xx=False,
                     compute_proj_chaotic=False, compute_full_chaotic=False):
        """
        Computes differences between adjacent (filtered) eigenvalues for the Hamiltonians.
        
        Filtering: Only eigenvalues between the 25th and 75th percentiles are used.
        
        Parameters:
            eigenvalues (dict): Dictionary mapping keys (e.g. 'full', 'xx', etc.) to eigenvalue arrays.
            compute_full (bool): Process the 'full' Hamiltonian if True.
            compute_xx (bool): Process the 'xx' Hamiltonian if True.
            compute_proj_chaotic (bool): Process the 'proj_chaotic' Hamiltonian if True.
            compute_full_chaotic (bool): Process the 'full_chaotic' Hamiltonian if True.
        
        Returns:
            dict: A dictionary of eigenvalue differences for each computed Hamiltonian.
        """
        eigenvalue_differences = {}
        # Create a new dictionary of filtered eigenvalues (do not modify the original)
        filtered_eigen = {}
        for key, arr in eigenvalues.items():
            arr = np.array(arr)  # Ensure the array is a NumPy array
            real_arr = np.real(arr)
            p25, p75 = np.percentile(real_arr, [25, 75])
            mask = (real_arr >= p25) & (real_arr <= p75)
            filtered_eigen[key] = arr[mask]
        if compute_full and 'full' in filtered_eigen:
            eigenvalue_differences['full'] = np.diff(filtered_eigen['full'])
        if compute_xx and 'xx' in filtered_eigen:
            eigenvalue_differences['xx'] = np.diff(filtered_eigen['xx'])
        if compute_proj_chaotic and 'proj_chaotic' in filtered_eigen:
            eigenvalue_differences['proj_chaotic'] = np.diff(filtered_eigen['proj_chaotic'])
        if compute_full_chaotic and 'full_chaotic' in filtered_eigen:
            eigenvalue_differences['full_chaotic'] = np.diff(filtered_eigen['full_chaotic'])
        
        if self.save_data:
            savename = f'{self.savepath}/Data/eigenvalue_metrics/eigenvalue_differences_N{self.n}_D{self.d}.pkl'
            qsave(eigenvalue_differences, savename)
        
        return eigenvalue_differences
    
    
    def gap_ratios(self, eigenvalue_differences, compute_full=True, 
                   compute_xx=False, compute_proj_chaotic=False, 
                   compute_full_chaotic=False):
        """
        Computes gap ratios for the Hamiltonians using a vectorized approach.
        
        For each Hamiltonian, all pairwise ratios r = min(g_i, g_j)/max(g_i, g_j) (for i < j)
        are computed from the gap differences.
        
        Parameters:
            eigenvalue_differences (dict): Dictionary of eigenvalue differences.
            compute_full (bool): Process the 'full' Hamiltonian if True.
            compute_xx (bool): Process the 'xx' Hamiltonian if True.
            compute_proj_chaotic (bool): Process the 'proj_chaotic' Hamiltonian if True.
            compute_full_chaotic (bool): Process the 'full_chaotic' Hamiltonian if True.
        
        Returns:
            tuple: (rs, means) where rs is a dictionary of gap ratio arrays and means is a dictionary 
                   of the mean gap ratio for each Hamiltonian type.
        """
        rs = {}
        
        # Process only the keys requested
        for key in eigenvalue_differences.keys():
            if key == 'full' and not compute_full:
                continue
            if key == 'xx' and not compute_xx:
                continue
            if key == 'proj_chaotic' and not compute_proj_chaotic:
                continue
            if key == 'full_chaotic' and not compute_full_chaotic:
                continue
            
            gaps = np.array(eigenvalue_differences[key])
            n_gaps = len(gaps)
            if n_gaps < 2:
                rs[key] = np.array([])
                continue
            
            # Get indices for the upper triangular (i<j) portion of the gap matrix
            iu = np.triu_indices(n_gaps, k=1)
            g1 = gaps[iu[0]]
            g2 = gaps[iu[1]]
            denominator = np.maximum(g1, g2)
            # Avoid division by zero; if denominator is 0, ratio is set to 0.
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.where(denominator != 0, np.minimum(g1, g2) / denominator, 0)
            rs[key] = ratios
        
        # Compute the mean gap ratio for each key.
        means = {key: np.mean(val) if val.size > 0 else np.nan for key, val in rs.items()}
        
        if self.save_data:
            savename = f'{self.savepath}/Data/eigenvalue_metrics/gap_ratios_N{self.n}_D{self.d}.pkl'
            qsave(rs, savename)
        
        return rs, means              
        
    def direct_vn_entropy(self, subsystem=None, compute_full=True,
                      compute_xx=False, compute_proj_chaotic=False,
                      compute_full_chaotic=False):
        """
        Computes the Von Neumann entropy of the eigenstates for each Hamiltonian.
        
        Parameters:
            subsystem (list): Subsystem indices to trace out. Defaults to the first half of the chain.
            compute_full (bool): If True, compute for the full Hamiltonian.
            compute_xx (bool): If True, compute for the XX Hamiltonian.
            compute_proj_chaotic (bool): If True, compute for the projected chaotic Hamiltonian.
            compute_full_chaotic (bool): If True, compute for the full chaotic Hamiltonian.
        
        Returns:
            dict: Dictionary of Von Neumann entropies for each Hamiltonian.
        """
        vn_entropies = {}
        
        # Default subsystem: first half of the chain
        if subsystem is None:
            subsystem = list(range(self.n // 2))
        
        # Ensure that density matrices exist in self.dms. If not, generate them from self.eigenvectors.
        if not hasattr(self, 'dms'):
            if not hasattr(self, 'eigenvectors'):
                raise AttributeError("No 'eigenvectors' attribute found; cannot compute density matrices.")
            self.dms = {}
            for key, states in self.eigenvectors.items():
                # Compute the density matrices for each eigenstate.
                self.dms[key] = [state * state.dag() for state in states]
        
        # Decide which Hamiltonian keys to process.
        key_flags = {
            'full': compute_full,
            'xx': compute_xx,
            'proj_chaotic': compute_proj_chaotic,
            'full_chaotic': compute_full_chaotic
        }
        
        # Compute the Von Neumann entropy for each requested Hamiltonian.
        for key, flag in key_flags.items():
            if flag and key in self.dms:
                # Compute the partial trace of each density matrix over the specified subsystem.
                partial_traces = [rho.ptrace(subsystem) for rho in self.dms[key]]
                # Compute the entropy for each partial density matrix.
                vn_entropies[key] = [entropy_vn(rho) for rho in partial_traces]
        
        if self.save_data:
            savename = f'{self.savepath}/Data/Entropies/direct_N{self.n}_D{self.d}.pkl'
            qsave(vn_entropies, savename)
        
        return vn_entropies

    def schmidt_SVD(self, state, split_index='middle'):
        """
        Performs a Schmidt decomposition of a pure state using Singular Value Decomposition (SVD).
        
        This method reshapes the state vector into a matrix according to a bipartition of the chain,
        then computes its singular values. The squared singular values are the Schmidt coefficients.
        
        Parameters:
            state (Qobj): The state (pure state vector) to be decomposed.
            split_index (int or str): The index at which to split the system. If 'middle', the chain
                                      is split evenly; otherwise, the provided integer is used.
                
        Returns:
            np.ndarray: The singular values of the reshaped state.
        """
        N = self.n   # total number of qudits in the chain
        D = self.d   # local Hilbert space dimension
        # Determine partition index
        if split_index == 'middle':
            n = N // 2
        else:
            n = int(split_index)
        
        # Convert the state to a dense array and flatten it.
        # This assumes the state is a Qobj representing a column vector.
        vec = state.full().flatten()
        
        # Reshape the vector into a matrix with shape (D**n, D**(N-n))
        C = vec.reshape((D**n, D**(N - n)))
        
        # Compute the singular values via SVD (no need for U or V)
        S = np.linalg.svd(C, compute_uv=False)
        return S
    
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
        
    def Projection_Expectation(self, eigenstates, eigenvalues, 
                               compute_full=True, compute_xx=False,
                               compute_proj_chaotic=False, compute_full_chaotic=False, 
                               use_snn_projectors=False):
        """
        Computes the expectation values of the projection operator for the eigenstates
        of the specified Hamiltonians.
        
        Parameters:
            eigenstates (dict): Dictionary mapping Hamiltonian types (e.g. 'full', 'xx', etc.)
                                to lists of eigenstate Qobjs.
            eigenvalues: (Not used in this method; provided for compatibility.)
            compute_full (bool): If True, compute for the full Hamiltonian.
            compute_xx (bool): If True, compute for the XX Hamiltonian.
            compute_proj_chaotic (bool): If True, compute for the projected chaotic Hamiltonian.
            compute_full_chaotic (bool): If True, compute for the full chaotic Hamiltonian.
            use_snn_projectors (bool): If True, use self.create_snn_projectors() instead of self.create_projectors().
        
        Returns:
            dict: A dictionary mapping each Hamiltonian type to an array of expectation values.
        """
        
        p_c, _ = self.create_projectors(self.N_neighbors)
        
        # Form the overall projection operator by summing over the projectors.
        Projector = sum(p_c)
        
        # Define which keys to compute based on the flags.
        key_flags = {
            'full': compute_full,
            'xx': compute_xx,
            'proj_chaotic': compute_proj_chaotic,
            'full_chaotic': compute_full_chaotic
        }
        
        Projection_Expectations = {}
        
        # Loop over the requested Hamiltonian types and compute expectation values.
        for key, flag in key_flags.items():
            if flag and key in eigenstates:
                exps = []
                for state in eigenstates[key]:
                    # Compute the expectation value <psi|Projector|psi>.
                    # Using .tr() on the Qobj gives a 1x1 matrix; extract its real part.
                    val = (state.dag() * Projector * state)
                    exps.append(val)
                Projection_Expectations[key] = np.array(exps)
        
        # Save the results if requested.
        if self.save_data:
            for key, values in Projection_Expectations.items():
                savename = f'{self.savepath}/Data/Projection_Expectation/N{self.n}_D{self.d}_Projection_Expectations_{key}.csv'
                np.savetxt(savename, values, delimiter=",",
                           header="# Expectation Values of the Projection Operator w.r.t. Eigenstates")
        
        return Projection_Expectations
        
    def classify_states(self, eigenstates, Projection_Expectations, 
                        scar_threshold=1e-10, thermal_threshold=0.1):
        """
        Classifies the eigenstates into 'scar' and 'thermal' categories based on 
        the projection expectation values.
    
        Parameters:
            eigenstates (dict): Dictionary mapping keys (e.g., 'full', 'xx', etc.) 
                                to lists of eigenstate Qobjs.
            Projection_Expectations (dict): Dictionary mapping the same keys to arrays 
                                            of projection expectation values.
            scar_threshold (float): States with |projection expectation| below this value 
                                    are classified as scars (default 1e-10).
            thermal_threshold (float): States with |projection expectation| above this 
                                       value are classified as thermal (default 0.1).
    
        Returns:
            tuple: 
                indices (dict): Dictionary mapping each key to a sub-dictionary with 
                                'scar' and 'thermal' indices.
                states (dict): Dictionary mapping each key to a sub-dictionary with the 
                               actual 'scar' and 'thermal' eigenstates.
        """
        indices = {}
        # Loop over each Hamiltonian type (key) in the projection expectations.
        for key, proj_vals in Projection_Expectations.items():
            # Classify indices based on the absolute projection expectation values.
            scar_indices = np.where(np.abs(proj_vals) < scar_threshold)[0]
            thermal_indices = np.where(np.abs(proj_vals) > thermal_threshold)[0]
            indices[key] = {'scar': scar_indices, 'thermal': thermal_indices}
        
        # Now extract the corresponding eigenstates.
        states = {}
        for key, idx_dict in indices.items():
            if key not in eigenstates:
                raise KeyError(f"Key '{key}' found in Projection_Expectations but missing in eigenstates.")
            states[key] = {
                'scar': [eigenstates[key][i] for i in idx_dict['scar']],
                'thermal': [eigenstates[key][i] for i in idx_dict['thermal']]
            }
        
        return indices, states
    
    def create_separation_operators(self, states, save=False, savenames=None):
        """
        Creates separation (projection) operators for scar and thermal states based on 
        the classified eigenstates.
        
        Parameters:
            states (dict): Dictionary containing classified states for each Hamiltonian type.
                           It is expected that states['full'] is a dictionary with keys 'scar' 
                           and 'thermal' whose values are lists of Qobj eigenstates.
            save (bool): If True, save the separation operators.
            savenames (list, optional): List of filenames to save the scar and thermal operators.
        
        Returns:
            tuple: (scar_operator, thermal_operator) as Qobj.
        """
        try:
            scar_states = states['full']['scar']
            thermal_states = states['full']['thermal']
        except KeyError as err:
            raise KeyError(f"Missing required key in states: {err}")
        
        # Build operators by summing the projectors of each state.
        scar_operator = sum(state * state.dag() for state in scar_states)
        thermal_operator = sum(state * state.dag() for state in thermal_states)
        
        if self.save_data or save:
            if savenames is None:
                savenames = [
                    f'{self.savepath}/Data/Separations/Scars_N{self.n}_D{self.d}.qobj',
                    f'{self.savepath}/Data/Separations/Thermals_N{self.n}_D{self.d}.qobj'
                ]
            qsave(scar_operator, savenames[0])
            qsave(thermal_operator, savenames[1])
        
        return scar_operator, thermal_operator
    
    
    def separate_array(self, variable: dict, indices: dict) -> dict:
        """
        Splits each list of values in a dictionary into 'scar' and 'thermal' parts using 
        the provided indices.
        
        Parameters:
            variable (dict): A dictionary where keys are identifiers (e.g., 'full', 'xx', etc.)
                             and values are lists (e.g., eigenvalues, gap ratios, etc.).
            indices (dict): A dictionary where keys match those in `variable` and values are 
                            dictionaries with keys 'scar' and 'thermal' containing index arrays.
        
        Returns:
            dict: A new dictionary where for each key the list is split into:
                  {'scar': [...], 'thermal': [...]}
        """
        split_variable = {}
        for key, values in variable.items():
            if key not in indices:
                raise KeyError(f"Key '{key}' not found in indices dictionary.")
            scar_idx = indices[key].get('scar', [])
            thermal_idx = indices[key].get('thermal', [])
            scar_values = [values[i] for i in scar_idx]
            thermal_values = [values[i] for i in thermal_idx]
            split_variable[key] = {'scar': scar_values, 'thermal': thermal_values}
        return split_variable
                
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
        
    def plot_eigenvalue_metrics(self, eigenvalues, eigenvalue_differences,
                            save=True, savename=None, show_full=True, 
                            show_xx=False, show_proj_chaotic=False, 
                            show_full_chaotic=False, show_hist=True, 
                            show_gaps=True, show_eigs=True, fit=True,
                            print_avgrs=True, avgrs=None, extra_name=''):
        """
        Plots histograms, gap plots, and sorted eigenvalues for each Hamiltonian, and optionally overlays fit results.
        
        Parameters:
            eigenvalues (dict): Dictionary mapping Hamiltonian types (e.g. 'full', 'xx', etc.) to sorted eigenvalues.
            eigenvalue_differences (dict): Dictionary mapping Hamiltonian types to arrays of adjacent eigenvalue gaps.
            save (bool): If True, save the plot.
            savename (str): Filename to save the plot.
            show_full (bool): If True, include the 'full' Hamiltonian.
            show_xx (bool): If True, include the 'xx' Hamiltonian.
            show_proj_chaotic (bool): If True, include the 'proj_chaotic' Hamiltonian.
            show_full_chaotic (bool): If True, include the 'full_chaotic' Hamiltonian.
            show_hist (bool): If True, show histogram of eigenvalue gaps.
            show_gaps (bool): If True, show the gap plot.
            show_eigs (bool): If True, show sorted eigenvalues.
            fit (bool): If True, overlay a fit (e.g. Wigner surmise or Poisson) on the histogram.
            print_avgrs (bool): If True, print and display the average gap ratio.
            avgrs (dict): Dictionary mapping Hamiltonian types to average gap ratios.
            extra_name (str): Additional string to append to the filename.
        
        Returns:
            None
        """
        # Set default savename if not provided.
        if savename is None:
            savename = f'{self.savepath}/Plots/Eigenvalue_Analysis_N{self.n}_D{self.d}_{extra_name}.png'
        
        # If average gap ratios are to be printed, check if they are provided.
        if print_avgrs:
            if avgrs is None:
                print('Average gap ratios (avgrs) not provided; they will not be displayed.')
                print_avgrs = False
    
        # Build a cosmetics dictionary based on the Hamiltonian flags.
        cosmetics = {}
        if show_full:
            cosmetics['full'] = {
                'color': 'royalblue', 
                'label': r'$H = H_{XX} + \sum PhP$',
                'hbox_loc': (0.95, 0.95),
                'fit_func': 'wigner'
            }
        if show_xx:
            cosmetics['xx'] = {
                'color': 'green',
                'label': r'$H = H_{XX}$',
                'hbox_loc': (0.95, 0.95),
                'fit_func': 'poisson'
            }
        if show_proj_chaotic:
            cosmetics['proj_chaotic'] = {
                'color': 'red',
                'label': r'$H = \sum PhP$',
                'hbox_loc': (0.95, 0.95),
                'fit_func': 'wigner'
            }
        if show_full_chaotic:
            cosmetics['full_chaotic'] = {
                'color': 'orange',
                'label': r'$H = H_{XX} + \sum h$',
                'hbox_loc': (0.95, 0.95),
                'fit_func': 'wigner'
            }
        
        # Determine subplot grid dimensions.
        num_rows = len(cosmetics)
        num_cols = sum([show_hist, show_gaps, show_eigs])
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 5 * num_rows + 2))
        axs = np.atleast_2d(axs)  # Ensure axs is 2D
        
        # Loop over each Hamiltonian type and plot the requested panels.
        for i, (key, cos) in enumerate(cosmetics.items()):
            color = cos['color']
            hbox_loc = cos['hbox_loc']
            col = 0
            
            # Plot histogram of eigenvalue differences (normalized by mean).
            if show_hist:
                hist_data = eigenvalue_differences[key]
                mean_gap = np.mean(hist_data)
                hist_data_norm = hist_data / mean_gap if mean_gap != 0 else hist_data
                # Filter out extremely high values for clarity.
                hist_data_norm = hist_data_norm[hist_data_norm <= 5]
                axs[i, col].hist(hist_data_norm, bins=60, edgecolor='black',
                                 color=color, density=True, alpha=0.6)
                axs[i, col].set_xlabel('Eigenvalue Gap / Mean Gap', fontsize=14)
                axs[i, col].set_ylabel('Frequency', fontsize=14)
                # Add Hamiltonian label in a text box.
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                axs[i, col].text(hbox_loc[0], hbox_loc[1], cos['label'],
                                 transform=axs[i, col].transAxes, fontsize=20,
                                 verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='serif')
                # Overlay fit function if requested.
                if fit:
                    xdata = np.linspace(0, max(hist_data_norm), 100)
                    if cos['fit_func'] == 'wigner':
                        # Wigner surmise for level spacings (normalized)
                        ydata1 = (32 / np.pi**2) * xdata**2 * np.exp(- (4 * xdata**2) / np.pi)
                        ydata2 = (np.pi/2) * xdata * np.exp(- (np.pi * xdata**2) / 4)
                        axs[i, col].plot(xdata, ydata1, 'r--', label='Wigner Fit - 1')
                        axs[i, col].plot(xdata, ydata2, ':', label='Wigner Fit - 2')
                        axs[i, col].legend()
                    elif cos['fit_func'] == 'poisson':
                        # For Poisson, the spacing distribution is exponential.
                        ydata = np.exp(-xdata)
                        axs[i, col].plot(xdata, ydata, 'r-', label='Poisson Fit')
                # Optionally, print average gap ratios.
                if print_avgrs and avgrs is not None and key in avgrs:
                    avg_r = avgrs[key]
                    axs[i, col].text(0.05, 0.95, fr'$\langle r \rangle = {avg_r:.2f}$',
                                     transform=axs[i, col].transAxes, fontsize=20,
                                     verticalalignment='top', horizontalalignment='left',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                col += 1
            
            # Plot gap sequence.
            if show_gaps:
                gaps = eigenvalue_differences[key]
                axs[i, col].plot(range(len(gaps)), gaps, marker='o', color=color)
                axs[i, col].set_xlabel('Order in Sorted List', fontsize=14)
                axs[i, col].set_ylabel('Gap', fontsize=14)
                col += 1
            
            # Plot sorted eigenvalues.
            if show_eigs:
                eigs = eigenvalues[key]
                axs[i, col].plot(range(len(eigs)), eigs, marker='o', color=color)
                axs[i, col].set_xlabel('Index', fontsize=14)
                axs[i, col].set_ylabel('Eigenvalue', fontsize=14)
                col += 1
        
        fig.suptitle(f'Eigenvalue Analysis: N = {self.n}, D = {self.d}', fontsize=20)
        if save:
            plt.savefig(savename, dpi=200, bbox_inches='tight')
            print(f'Plot saved to: {savename}')
        plt.show()
        plt.close()

    def plot_vn_entropy(self, eigenvalues: dict, entropies: dict, split: bool = False, 
                    save: bool = True, savename: str = None, 
                    show_full: bool = True, show_xx: bool = False,
                    show_proj_chaotic: bool = False, show_full_chaotic: bool = False,
                    extra_name: str = "") -> None:
        """
        Plots the Von Neumann entropy of eigenstates for each Hamiltonian type.
        
        Parameters:
            eigenvalues (dict): Dictionary of eigenvalues. For each key (e.g. 'full', 'xx', etc.),
                                the value can be either an array of sorted eigenvalues or a dict 
                                with keys 'scar' and 'thermal' (if split=True).
            entropies (dict): Dictionary of Von Neumann entropies. For each key, either an array or a dict 
                              with keys 'scar' and 'thermal'.
            split (bool): If True, plot separate curves for thermal and scar states.
            save (bool): If True, save the plot.
            savename (str): Filename for saving the plot. If None, a default name is used.
            show_full (bool): If True, include the 'full' Hamiltonian.
            show_xx (bool): If True, include the 'xx' Hamiltonian.
            show_proj_chaotic (bool): If True, include the 'proj_chaotic' Hamiltonian.
            show_full_chaotic (bool): If True, include the 'full_chaotic' Hamiltonian.
            extra_name (str): Additional string appended to the filename.
            
        Returns:
            None
        """
        
        # Set default filename if none provided.
        if savename is None:
            savename = f'{self.savepath}/Plots/Von_Neumann_Entropies_N{self.n}_D{self.d}_{extra_name}.png'
        
        # Build cosmetics dictionary based on flags.
        cosmetics = {}
        if show_full:
            cosmetics['full'] = {'color': 'royalblue', 'label': 'Full Hamiltonian'}
        if show_xx:
            cosmetics['xx'] = {'color': 'green', 'label': 'XX Hamiltonian'}
        if show_proj_chaotic:
            cosmetics['proj_chaotic'] = {'color': 'red', 'label': 'Projected Chaotic'}
        if show_full_chaotic:
            cosmetics['full_chaotic'] = {'color': 'orange', 'label': 'Full Chaotic'}
        
        num_plots = len(cosmetics)
        # Create one subplot per Hamiltonian type.
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
        axs = np.atleast_1d(axs)
        
        for i, (key, cosmetic) in enumerate(cosmetics.items()):
            color = cosmetic['color']
            ax = axs[i]
            
            # If split mode is enabled, plot thermal and scar subsets separately.
            if split:
                try:
                    thermal_eigs = eigenvalues[key]['thermal']
                    thermal_entropy = entropies[key]['thermal']
                    scar_eigs = eigenvalues[key]['scar']
                    scar_entropy = entropies[key]['scar']
                except KeyError as e:
                    raise KeyError(f"Key error in split mode for '{key}': {e}")
                
                ax.plot(thermal_eigs, thermal_entropy, 'o', color=color, label='Thermal States')
                ax.plot(scar_eigs, scar_entropy, 'D', color='k', label='Scar States', markersize=10)
                num_scar = len(scar_eigs)
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.61, 0.15, f'Number of Scars: {num_scar}', transform=ax.transAxes, 
                        fontsize=20, verticalalignment='top', bbox=props)
            else:
                # Otherwise, plot a single curve for the given key.
                print(eigenvalues.get(key,[]), 'ent', entropies.get(key,[]), key, "--------------------------------------")
                ax.plot(eigenvalues.get(key,[]), entropies.get(key,[]), 'o', color=color, markersize=10)
            
            ax.set_xlabel('Eigenvalue E', fontsize=14)
            ax.set_ylabel(r'$S_{VN}$', fontsize=14)
            epsilon_text = rf"$\varepsilon = {self.epsilon:.3f}$"
            ax.text(0.05, 0.95, epsilon_text, transform=ax.transAxes, fontsize=20,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
            # Add Hamiltonian-specific text (assuming add_hamiltonian_text is defined).
            self.add_hamiltonian_text(fig, key, axs_index=i)
        
        fig.suptitle(f'Von Neumann Entropy of Eigenstates (N={self.n}, D={self.d})', fontsize=20)
        
        
        
        if save:
            plt.savefig(savename, dpi=200, bbox_inches='tight')
            print(f'Plot saved to: {savename}')
        plt.show()
        plt.close()
    
    def plot_proj_expectation(self, Energies: dict, Expectations: dict, save: bool = True, 
                          savename: str = None, show_full: bool = True, show_xx: bool = False,
                          show_proj_chaotic: bool = False, show_full_chaotic: bool = False, 
                          extra_name: str = "", split: bool = False) -> None:
        """
        Plots the expectation value of the projection operator with respect to the eigenstates.
        
        Parameters:
            Energies (dict): Dictionary containing the energies. If split is True, for each key 
                             (e.g. 'full') the value is expected to be a dict with keys 'thermal' and 'scar'.
            Expectations (dict): Dictionary containing the projection expectation values (same format as Energies).
            save (bool): If True, save the plot.
            savename (str): Filename to save the plot.
            show_full (bool): If True, include the 'full' Hamiltonian.
            show_xx (bool): If True, include the 'xx' Hamiltonian.
            show_proj_chaotic (bool): If True, include the 'proj_chaotic' Hamiltonian.
            show_full_chaotic (bool): If True, include the 'full_chaotic' Hamiltonian.
            extra_name (str): Additional string appended to the saved filename.
            split (bool): If True, plot separate curves for thermal and scar states.
            
        Returns:
            None
        """
        # Set default savename if not provided.
        if savename is None:
            savename = f'{self.savepath}/Plots/Projector_Expectations_N{self.n}_D{self.d}_{extra_name}.png'
        
        # Build cosmetics dictionary from the flags.
        cosmetics = {}
        if show_full:
            cosmetics['full'] = {'color': 'royalblue'}
        if show_xx:
            cosmetics['xx'] = {'color': 'green'}
        if show_proj_chaotic:
            cosmetics['proj_chaotic'] = {'color': 'red'}
        if show_full_chaotic:
            cosmetics['full_chaotic'] = {'color': 'purple'}
        
        # Determine the number of subplots from the active Hamiltonian types.
        num_plots = len(cosmetics)
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
        axs = np.atleast_1d(axs)
        
        # Loop over each Hamiltonian type to plot.
        for i, (key, cos) in enumerate(cosmetics.items()):
            color = cos['color']
            ax = axs[i]
            if split:
                # In split mode, each entry is expected to be a dict with 'thermal' and 'scar'.
                try:
                    thermal_energies = Energies[key]['thermal']
                    thermal_expectations = Expectations[key]['thermal']
                    scar_energies = Energies[key]['scar']
                    scar_expectations = Expectations[key]['scar']
                except KeyError as e:
                    raise KeyError(f"Expected split data with keys 'thermal' and 'scar' for '{key}': {e}")
                
                ax.plot(thermal_energies, thermal_expectations, 'o', color=color, label='Thermal States')
                ax.plot(scar_energies, scar_expectations, 'D', color='k', label='Scar States', markersize=10)
                num_scar_states = len(scar_energies)
                props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                ax.text(0.61, 0.15, f'Number of Scars: {num_scar_states}', transform=ax.transAxes,
                        fontsize=20, verticalalignment='top', bbox=props, fontfamily='serif')
                # Set y-limit based on thermal expectation values.
                y_max = 1.1 * max(thermal_expectations)
            else:
                # Non-split mode: directly plot the array.
                ax.plot(Energies[key], Expectations[key], 'o', color=color)
                y_max = 1.1 * max(Expectations[key])
            
            ax.set_xlabel('Energy E', fontsize=14)
            ax.set_ylabel(r'$\langle \sum_i P_i \rangle$', fontsize=14)
            ax.set_ylim(-0.1, y_max)
            # Add extra text specific to the Hamiltonian (this function is assumed to be defined).
            self.add_hamiltonian_text(fig, key, axs_index=i)
        
        fig.suptitle(f'Expectation Value of Projection Operator w.r.t. Eigenstates (N={self.n}, D={self.d})', fontsize=20)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            plt.savefig(savename, dpi=200, bbox_inches='tight')
            print(f'Plot saved to: {savename}')
        plt.show()
        plt.close()
    
    def tfo_phase_animation(self, savename=None, num_frames=100, duration=10, initial_state='default'):
        """
        Creates and saves an animation of the transfer fidelity operator (TFO)
        time evolution as the phase parameter varies over one full cycle (0 to 2π).
    
        Parameters:
            savename (str): Filename (with path) to save the animation as a GIF.
                            If None, a default name based on self.savepath and self.n is used.
            num_frames (int): Number of frames in the animation.
            duration (int): Duration of the time evolution in each frame.
            initial_state (Qobj or 'default'): The initial state for the time evolution.
                                               If 'default' or None, self.initial_state is used.
    
        Returns:
            None
        """
    
        # Set default savename if not provided.
        if savename is None:
            savename = f'{self.savepath}/animations/tfo_phase/tfo_phase_N{self.n}.gif'
        
        # Use self.initial_state if initial_state is 'default' or None.
        if initial_state == 'default' or initial_state is None:
            initial_state = self.initial_state
    
        # Create time and phase arrays.
        times = np.linspace(0, duration, num_frames)
        phase_values = np.linspace(0, 2 * np.pi, num_frames)
        
        # Set up the figure and axis.
        fig, ax = plt.subplots(figsize=(10, 6))
        # Create an initially empty line.
        line, = ax.plot([], [], 'b-', label="Fidelity")
        
        # Add fixed reference lines.
        ax.axhline(1, color='r', linestyle=":")
        ax.axhline(0, color='r', linestyle=":")
        ax.axvline(np.pi/2, color='r', linestyle=":")
        ax.axvline(3*np.pi/2, color='r', linestyle=":")
        ax.axvline(5*np.pi/2, color='r', linestyle=":")
        ax.legend(fontsize=12)
        ax.set_title("TFO Timeseries Animation", fontsize=16)
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("Fidelity", fontsize=14)
        
        # Add a text box with the initial state's description.
        normalization = np.sum(np.array(self.coeffs)**2)
        if normalization != 1:
            text_str = rf"$|\phi_{{{self.loc}}}\rangle = \frac{{1}}{{\sqrt{{{normalization}}}}}({self.coeffs[0]}|0\rangle + {self.coeffs[1]}|1\rangle + {self.coeffs[2]}|2\rangle)$"
        else:
            text_str = rf"$|\phi_{{{self.loc}}}\rangle = {self.coeffs[0]}|0\rangle + {self.coeffs[1]}|1\rangle + {self.coeffs[2]}|2\rangle$"
        ax.text(0.95, 0.95, text_str, ha='right', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5), transform=ax.transAxes)
        
        # Animation initialization: clear the line.
        def init():
            line.set_data([], [])
            return (line,)
    
        # Update function: for each frame, update the phase, compute TFO and solve the evolution.
        def update(frame):
            phase = phase_values[frame]
            # Compute the transfer fidelity operator for the current phase.
            tfo, _ = self.create_tfo(phase=phase)
            # Solve the evolution with the current TFO in the op_list.
            _, results = self.solve(times, op_list=[tfo], initial_state=initial_state)
            # Expectation values for tfo are assumed to be in results['full'].expect[0].
            fidelity = results['full'].expect[0]
            line.set_data(times, fidelity)
            ax.set_title(f"TFO Timeseries (Phase = {phase:.2f})", fontsize=16)
            return (line,)
        
        # Create and save the animation.
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
        ani.save(savename, writer='pillow', dpi=200)
        plt.show()
        plt.close()
    
    def add_hamiltonian_text(self, fig, hamiltonian, axs_index=0):
        """
        Adds a text box with Hamiltonian information to the specified axis.
    
        Parameters:
            fig (matplotlib.figure.Figure): Figure containing the axes.
            hamiltonian (str): One of 'full', 'xx', 'proj_chaotic', 'full_chaotic'.
            axs_index (int): Index of the axis to annotate.
        """
        ax = fig.axes[axs_index]
        hbox_loc = (0.95, 0.90)
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        if hamiltonian == 'full':
            textstr = r'$H = H_{XX} + \sum PhP$'
        elif hamiltonian == 'xx':
            textstr = r'$H = H_{XX}$'
        elif hamiltonian == 'proj_chaotic':
            textstr = r'$H = \sum PhP$'
        elif hamiltonian == 'full_chaotic':
            textstr = r'$H = \sum h$'
        else:
            print("Invalid Hamiltonian name! Valid options: 'full', 'xx', 'proj_chaotic', 'full_chaotic'.")
            return
        ax.text(hbox_loc[0], hbox_loc[1], textstr, transform=ax.transAxes, fontsize=20,
                verticalalignment='top', horizontalalignment='right', bbox=props, fontfamily='serif')

qc = QutritCross(5)
times,results = qc.Evolve(solve_full=True, solve_xx=True, solve_proj_chaotic=True, solve_full_chaotic=True, op_list=[qc.CPLM])
qc.plot_operator_timeseries(times, results, show_full=True, show_xx=True, show_proj_chaotic=True, show_full_chaotic=True, legend=True)
qc.plot_operator_timeseries(times, results, show_full=False, show_xx=False, show_proj_chaotic=True, show_full_chaotic=False, legend=True)

