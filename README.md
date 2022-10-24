# GoCModel_Basic
 network of gocs with low frequency background inputs

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



## Utils

[PythonUtils](PythonUtils) has function definitions for generating connectivity (electrical or chemical)


##  Known Issues

!x Known Issues
- Using reduced morphology with current channel densities causes model to fail (Vm goes to 80mV after around 500 ms -> which channel is unstable? Density adjustment for morphology?)
- Constructing network with a Population that has ComponentType  from class Cell2CaPools -> LEMS file fails to be generated
