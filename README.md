# GoCModel_Basic
Basic code to generate network of electrically coupled cerebellar Golgi cells with low frequency background inputs

## Model descriptions
Channel and synapse mechanisms are in [Mechanisms](Mechanisms)

Cell descriptions are in [Cells/Golgi](Cells/Golgi)

- [/solinas](Cells/Golgi/solinas) includes the original Solinas morphology with soma, 3 dendrites and long axon compartment. These are also the morphologies of [GoC.cell.nml](Cells/Golgi/GoC.cell.nml) and [GoC_2Pools.cell.nml](Cells/Golgi/GoC_2Pools.cell.nml)
- [/reduced](Cells/Golgi/reduced) includes 10 compartment reduced models, optimised by E. Piasini (see [Piasini, 2015](https://discovery.ucl.ac.uk/id/eprint/1464128/1/eugenio_piasini_phd_thesis.pdf)).
- An example of full reconstructed morphology is present [here](Cells/Golgi/Golgi_040408_C1.cell.nml)

More channel density sets are stored in [Parameters](Parameters/useParams_FI_14_25.pkl) that when inserted into the Solinas morphology, produce autonomous firing rates of 2-9 Hz and F-I slope of 14-25 Hz/nA.

### Varying channel densities
To construct new GoC files using one of the morphologies and different parameter sets, use `vary_channels.py` or `vary_channels_2pools.py` - depending on whether you want a cell of class [Cell](https://docs.neuroml.org/Userdocs/Schemas/Cells.html#cell) or [Cell2CaPools](https://docs.neuroml.org/Userdocs/Schemas/Cells.html#cell2capools).


## Generating networks

- [simulate_one_cell.py](Network/simulate_one_cell.py) generates a network with 1 cell (from specified GoC model). For example use, see [example_gen_onegoc.ipynb](Network/example_gen_onegoc.ipynb)
- [generate_simple_network.py](Network/generate_simple_network.py) generates network of ~40 cells, with electrical connectivity and background inputs (low frequency Poisson spiketrains). For example use, see [this nb](Network/example_gen_network.ipynb)

## Utils

[PythonUtils](PythonUtils) has function definitions for generating connectivity (electrical or chemical)


##  Known Issues :construction_worker:
- Using reduced or full morphology with current channel densities causes model to fail (Vm goes to 80mV after around 500 ms -> which channel is unstable? Density adjustment for morphology?)
  - :white_check_mark: Fix: NaT reaches very small time constants at spike -> use much smaller integration dt (0.001 ms or lower)
- Constructing network with a Population that has ComponentType  from class Cell2CaPools -> LEMS file fails to be generated
  - :small_blue_diamond: Diagnosis: Cannot create events file using event port 'spike' - not supported for class Cell2CaPools?


## Requirements

- python2.7
- NeuroML and pyNeuroML python libraries
- jvm

- Neuron 7.3 or above (compile mod)
