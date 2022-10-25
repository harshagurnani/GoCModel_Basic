import neuroml as nml
from pyneuroml import pynml
from pyneuroml.lems import LEMSSimulation
import lems.api as lems

import os
import sys
sys.path.append('../PythonUtils/')
sys.path.append('../Mechanisms/')

import numpy as np
import pickle as pkl
import math

import network_utils as nu


def create_LEMS_sim( duration=1000, dt=0.025, goc_path='../Cells/Golgi/', goc_file='GoC', has2Pools=False, run=False, seed=100 ):
	"""
	Create LEMS simulation with single cell

	Parameters:
	==========
	duration:  Simulation length in ms
	goc_path:  Path to GoC NML file.
	goc_file:  GoC model filename (exclude .cell.nml extension)
	has2Pools: Of class Cell2CaPools or Cell?
	"""


	# Directory to save data
	datadir = '../Data_OneGoC/'
	if not os.path.exists(datadir):
		os.makedirs(datadir)

	if has2Pools:
		poolID = '_2CaPools'
	else:
		poolID = ''
	simid = 'OneGoC_' + goc_file + poolID


	# Network file
	net = nml.Network( id=simid, type="networkWithTemperature" , temperature="23 degC" )
	net_doc = nml.NeuroMLDocument( id=net.id )
	net_doc.networks.append( net )


	# GoC
	goc_filename = goc_path + '{}.cell.nml'.format( goc_file )
	net_doc.includes.append( nml.IncludeType( href=goc_filename ) )
	if has2Pools:
		goc_type = pynml.read_neuroml2_file( goc_filename ).cell2_ca_poolses[0]
	else:
		goc_type = pynml.read_neuroml2_file( goc_filename ).cells[0]

	# Create GoC population
	goc_pop = nml.Population( id=goc_type.id+"Pop", component = goc_type.id, type="populationList", size=1 )
	goc_pop.instances.append(  nml.Instance( id=0, location=nml.Location( x=0, y=0, z=0) )  ) 	# add 1 GoC
	net.populations.append( goc_pop ) 	# add pop to network description


	print('Writing files...')
	### --------------  Write files

	net_filename = simid+'.net.nml'							#NML2 description of network instance
	pynml.write_neuroml2_file( net_doc, net_filename )

	# LEMS simulation and file
	simid = 'sim_'+net.id
	ls = LEMSSimulation( simid, duration=duration, dt=dt, simulation_seed=seed )
	ls.assign_simulation_target( net.id )
	ls.include_neuroml2_file( net_filename)							#Network NML2 file

	# Specify outputs
	'''
	eof0 = 'Events_file'
	ls.create_event_output_file(eof0, datadir+"%s.spikes.dat"%simid,format='ID_TIME')
	ls.add_selection_to_event_output_file( eof0, 0, '{}/{}/{}'.format( goc_pop.id, 0, goc_type.id), 'spike' )
	'''
	of0 = 'Volts_file'
	ls.create_output_file(of0, datadir+"%s.v.dat"%simid)
	ls.add_column_to_output_file(of0, 0, '{}/{}/{}/v'.format( goc_pop.id, 0, goc_type.id))

	#Create Lems file to run
	lems_simfile = ls.save_to_file()


	# Create the Neuron-python simulation file (same name as LEMS - but with .py )
	if run:
		res = pynml.run_lems_with_jneuroml_neuron( lems_simfile, max_memory="2G", nogui=True, plot=False)
	else:
		res = pynml.run_lems_with_jneuroml_neuron( lems_simfile, max_memory="2G", only_generate_scripts = True, compile_mods = False, nogui=True, plot=False)
		# -----> before running sim, remember to compile mod files. Then run the LEMS***.py file

	return res



if __name__ =='__main__':
	runid=0

	res = create_LEMS_sim( duration = 2000, dt=0.025, seed = 123)
	print(res)
