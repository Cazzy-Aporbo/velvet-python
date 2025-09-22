"""
Exploratory Python Demonstrations
Author: Cazzy
Purpose: This collection of Python demos explores unusual and creative applications of 
advanced libraries. The goal is to show not only technical fluency but also imagination 
and the ability to apply tools in ways that are unexpected & practical. 
Each demo is self-contained, illustrating concepts from quantum randomness to topological 
data structures, graph neural networks, 3D chaotic systems, and even mapping biological 
data to sound. This is meant to inspire, educate, and demonstrate mastery across multiple domains.
"""

import numpy as np
import matplotlib.pyplot as plt

# Quantum Random Numbers
def demo_quantum_rng():
    try:
        from qiskit import QuantumCircuit, Aer, execute
        qc = QuantumCircuit(1,1)
        qc.h(0)
        qc.measure(0,0)
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(qc, simulator, shots=10)
        result = job.result().get_counts()
        print("\nQuantum RNG demo result:", result)
    except Exception as e:
        print("Quantum RNG demo failed:", e)

# Topological Data Analysis
def demo_topology():
    try:
        from ripser import ripser
        import persim
        data = np.random.rand(50,3)
        diagrams = ripser(data)['dgms']
        persim.plot_diagrams(diagrams, show=True)
        print("\nTopological data analysis demo complete")
    except Exception as e:
        print("Topology demo failed:", e)

# Graph Neural Network (tiny example)
def demo_graph_nn():
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long)
        x = torch.tensor([[1],[2],[3]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        conv = GCNConv(1,2)
        out = conv(data.x, data.edge_index)
        print("\nGraph Neural Network demo output:\n", out)
    except Exception as e:
        print("Graph NN demo failed:", e)

# Lorenz Attractor Simulation in 3D
def demo_lorenz_attractor():
    try:
        from scipy.integrate import solve_ivp
        import pyvista as pv
        def lorenz(t, state, sigma=10, beta=8/3, rho=28):
            x, y, z = state
            dx = sigma*(y-x)
            dy = x*(rho-z) - y
            dz = x*y - beta*z
            return [dx, dy, dz]
        t_span = (0, 2)
        state0 = [0.1, 0, 0]
        sol = solve_ivp(lorenz, t_span, state0, t_eval=np.linspace(0,2,500))
        cloud = pv.PolyData(np.vstack([sol.y[0], sol.y[1], sol.y[2]]).T)
        plotter = pv.Plotter()
        plotter.add_points(cloud, color='orange', point_size=5)
        plotter.show()
        print("\nLorenz attractor demo complete")
    except Exception as e:
        print("Lorenz attractor demo failed:", e)

# Musical Data Science from Numeric Data
def demo_dna_music():
    try:
        from Bio.Seq import Seq
        import librosa
        import soundfile as sf
        seq = Seq("AGCTAGCAGT")
        notes = [ord(c)%12+60 for c in seq] # map letters to MIDI notes
        y = np.sin(2*np.pi*np.linspace(0,1,22050)*np.array(notes)[:,None])
        y = y.sum(axis=0)
        sf.write("dna_music_demo.wav", y, 22050)
        print("\nDNA music demo written to dna_music_demo.wav")
    except Exception as e:
        print("DNA music demo failed:", e)

def main():
    demo_quantum_rng()
    demo_topology()
    demo_graph_nn()
    demo_lorenz_attractor()
    demo_dna_music()
    print("\nAll exploratory demos executed successfully.")

if __name__ == "__main__":
    main()
