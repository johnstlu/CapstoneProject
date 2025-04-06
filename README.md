# CapstoneProject

This repository contains the Python code and simulations used in my Capstone Project for my undergraduate degree in Theoretical Physics at Trinity College Dublin: "Perfect Quantum State Transfer in Chaotic Systems". This project was supervised by Dr. Shane Dooley. 

This purpose of code written for this project was to construct the following 

- An exact representation of a heisenberg spin chain capable of Perfect Quantum State Transfer.
- Local Projection Operators and Random Local Interactions to construct Quantum Many Body Scars
- Methods for computing the evolution of the system under the Schrodinger Eqn.
- Methods for computing and analysing the spectrum of the hamiltonian, including eigenvalue gap statistics, Von Neumann Entropy, and Expectation Values of Eiegenstates w.r.t the Projection Operators
- Visualisation methods, including plotting and animation

## Project Structure

The repository is organized into the following directories and files:

- **main_system/**: Contains the primary class that models and simulates the XX-Scar system. This class includes methods for system construction, simulation, and visualization.
  - `Qudit_Chain_OOP.py`: Main OOP class for the simulation.
  
- **other_systems/**: Houses code for alternative or work-in-progress simulations that are based on variations of the primary system.
  - `QutritEntanglementSharing.py`: Simulations of an alternative quantum system.

- **snippets/**: Includes smaller code snippets or scripts for testing, plotting, or experimenting with various methods used in the project.
  - `ClassDefinition.py`: The initialisation of the class quditChain
  - `ProjectorsConstruction.py`: A function to generate the projectors used in the Shiriashi-Mori Hamiltonian
  - `GenerateSM_Hamiltonian.py`: A set of functions to generate the Shiriashi-Mori Hamiltonian
  - `SchmidtSVN.py`: A function compute the Von Neumann Entropy of the Eigenstates of the hamiltonian, using a schmidt decompositon. 
  - `XXHamiltonianGeneration.py`: A set of functions to generate the XX Hamiltonian
  - `Solve.py`: A function to simulate the evolution of the sytem
    
- **requirements.txt**: A list of Python packages required to run the code.

## Installation

To run the code locally, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/CapstoneProject.git
cd CapstoneProject
pip install -r requirements.txt

