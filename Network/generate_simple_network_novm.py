import neuroml as nml
from pyneuroml import pynml
from pyneuroml.lems import LEMSSimulation
import lems.api as lems

import os
import sys
sys.path.append('../PythonUtils/')
sys.path.append('../Parameters/')
sys.path.append('../Mechanisms/')

import numpy as np
import pickle as pkl
import math

import initialise_network_params as inp

def create_GoC_network( duration=1000, dt=0.025, seed=100, runid=0, hom=True, run=False, goc_path = '../Cells/Golgi/', goc_fileid='GoC_', has2Pools=False, dend_id = [1,2,5] ):
	"""
	Create simple network of GJ coupled GoCs, with low frequency background inputs


	Parameters:
	==========
	duration:  Simulation length in ms
	runid:     Random seed, to select GoC cell models
	hom:	  (Boolean) Homogeneous network (all GoCs same?)
	"""

	### ---------- Load Params
	nDend = len(dend_id)
	p = get_params( runid=runid, hom=hom, nGJ_dend=nDend )

	hom_str = 'hom' if hom else 'het'
	# Directory to save data
	datadir = '../Data_SimpleNet_' + hom_str + '/'

	if not os.path.exists(datadir):
		os.makedirs(datadir)

	simid = 'GoCNet_'+format(runid,'05d')+'_'+hom_str

	net = nml.Network( id=simid, type="networkWithTemperature" , temperature="23 degC" )
	net_doc = nml.NeuroMLDocument( id=net.id )
	net_doc.networks.append( net )


	### -------------- Component types ------------------------- ###
	"""
	include the relevant nml files - cells, synapses, spike generators
	for heterogeneous networks, 5 different GoC files
	"""

	# ---- 1. GoC types
	goc_params = np.unique( np.asarray(p["GoC_ParamID"]))

	ctr=0
	goc_type=[]
	for pid in goc_params:
		gocID = goc_fileid+format(pid, '05d')
		goc_type.append( gocID )
		goc_filename = goc_path + '{}.cell.nml'.format( goc_type[ctr] )
		if has2Pools:
			goc_type[ctr] = pynml.read_neuroml2_file( goc_filename ).cell2_ca_poolses[0]
		else:
			goc_type[ctr] = pynml.read_neuroml2_file( goc_filename ).cells[0]
		net_doc.includes.append( nml.IncludeType( href=goc_filename ) )
		ctr += 1

	# ---- 2. Inputs
	synapse = {}
	inputBG = {}

	for input in p["Inputs"]["types"]:
		# Load synapse type
		Inp = p["Inputs"][input]
		net_doc.includes.append( nml.IncludeType(href=Inp["syn_type"][0] ))	# filename
		if Inp["syn_type"][1] =='ExpThreeSynapse':
			synapse[input] = pynml.read_neuroml2_file( Inp["syn_type"][0] ).exp_three_synapses[0] #Component Type
		elif Inp["syn_type"][1] =='ExpTwoSynapse':
			synapse[input] = pynml.read_neuroml2_file( Inp["syn_type"][0] ).exp_two_synapses[0]

		# Set up spike generators
		if Inp["type"] == 'poisson':
			inputBG[input] = nml.SpikeGeneratorPoisson( id = input, average_rate="{} Hz".format(Inp["rate"][0]) )
			net_doc.spike_generator_poissons.append( inputBG[input] )

		elif Inp["type"] == 'constantRate':
			inputBG[input] = nml.SpikeGenerator( id = input, period="{} ms".format(1000.0/(Inp["rate"][0])) )
			net_doc.spike_generators.append( inputBG[input] )


	# ---- 3.  Gap Junctions
	gj = nml.GapJunction( id="GJ_0", conductance="426pS" )	# GoC synapse
	net_doc.gap_junctions.append(gj)


	### ------------------ Populations ------------------------- ###
	"""
	Construct cell and input populations
	"""

	# Create GoC population
	goc_pop=[]
	goc=0
	for pid in range(p["nPop"]):
		goc_pop.append( nml.Population( id=goc_type[pid].id+"Pop", component = goc_type[pid].id, type="populationList", size=p["nGoC_per_pop"] ) )
		for ctr in range(p["nGoC_per_pop"]):
			inst = nml.Instance( id=ctr )
			goc_pop[pid].instances.append( inst )
			inst.location = nml.Location( x=p["GoC_pos"][goc,0], y=p["GoC_pos"][goc,1], z=p["GoC_pos"][goc,2] )
			goc+=1
		net.populations.append( goc_pop[pid] )



	### Background Input population
	inputBG_pop = {}
	for input in inputBG:
		Inp = p["Inputs"][input]
		inputBG_pop[input] =  nml.Population(id=inputBG[input].id+"_pop", component=inputBG[input].id, type="populationList", size=Inp["nInp"])
		for ctr in range( Inp["nInp"] ):
			inst = nml.Instance(id=ctr)
			inputBG_pop[input].instances.append( inst )
			inst.location = nml.Location( x=Inp["pos"][ctr,0], y=Inp["pos"][ctr,1], z=Inp["pos"][ctr,2] )
		net.populations.append( inputBG_pop[input] )



	### ------------ Connectivity
	"""
	Connect GoCs with electrical synapses
	Connect spiking background inputs (inputBG) to GoCs
	"""



	### 1. Background  inputs:

	BG_Proj = {}
	for input in inputBG:
		Inp = p["Inputs"][input]
		BG_Proj[input] = []

		for jj in range( p["nPop"] ):
			BG_Proj[input].append( nml.Projection(id='{}_to_{}'.format(input, goc_type[jj].id), presynaptic_population=inputBG_pop[input].id, postsynaptic_population=goc_pop[jj].id, synapse=synapse[input].id) )
			net.projections.append(BG_Proj[input][jj])
			ctr=0
			nSyn = Inp["conn_pairs"][jj].shape[1]
			for syn in range( nSyn ):
				pre, goc = Inp["conn_pairs"][jj][:, syn]
				if Inp["syn_loc"] == 'soma':
					conn = nml.ConnectionWD(id=ctr, pre_cell_id='../{}/{}/{}'.format(inputBG_pop[input].id, pre, inputBG[input].id), post_cell_id='../{}/{}/{}'.format(goc_pop[jj].id, goc, goc_type[jj].id), post_segment_id='0', post_fraction_along="0.5", weight=Inp["conn_wt"][jj][syn], delay="0 ms")	#on soma
				elif Inp["syn_loc"] == 'dend':
					#conn = nml.ConnectionWD(id=ctr, pre_cell_id='../{}/{}/{}'.format(inputBG_pop[input].id, pre, inputBG[input].id), post_cell_id='../{}/{}/{}'.format(goc_pop[jj].id, goc, goc_type[jj].id), post_segment_id=dend_id[int(Inp["conn_loc"][jj][0,syn])], post_fraction_along=dend_id[int(Inp["conn_loc"][jj][1,syn])], weight=Inp["conn_wt"][jj][syn], delay="0 ms")	#on dend
					conn = nml.ConnectionWD(id=ctr, pre_cell_id='../{}/{}/{}'.format(inputBG_pop[input].id, pre, inputBG[input].id), post_cell_id='../{}/{}/{}'.format(goc_pop[jj].id, goc, goc_type[jj].id), post_segment_id=dend_id[int(Inp["conn_loc"][jj][0,syn])], post_fraction_along=0.5, weight=Inp["conn_wt"][jj][syn], delay="0 ms")	#on dend
				BG_Proj[input][jj].connection_wds.append(conn)
				ctr+=1


	### 2. Electrical coupling between GoCs

	GoCCoupling = []
	ctr=0
	for pre in range( p["nPop"] ):
		for post in range( pre, p["nPop"] ):
			GoCCoupling.append( nml.ElectricalProjection( id="GJ_{}_{}".format(goc_pop[pre].id, goc_pop[post].id), presynaptic_population=goc_pop[pre].id, postsynaptic_population=goc_pop[post].id ) )
			net.electrical_projections.append( GoCCoupling[ctr] )

			gjParams = p["econn_pop"][pre][post-pre]
			for jj in range( gjParams["GJ_pairs"].shape[0] ):
				conn = nml.ElectricalConnectionInstanceW( id=jj, pre_cell='../{}/{}/{}'.format(goc_pop[pre].id, gjParams["GJ_pairs"][jj,0], goc_type[pre].id), pre_segment=dend_id[int(gjParams["GJ_loc"][jj,0])], pre_fraction_along='0.5', post_cell='../{}/{}/{}'.format(goc_pop[post].id, gjParams["GJ_pairs"][jj,1], goc_type[post].id), post_segment=dend_id[int(gjParams["GJ_loc"][jj,1])], post_fraction_along='0.5', synapse=gj.id, weight=gjParams["GJ_wt"][jj] )
				GoCCoupling[ctr].electrical_connection_instance_ws.append( conn )
			ctr+=1




	print('Writing files...')
	### --------------  Write files

	"""
	Create network description (NML)
	Specify output files
	Create and write Lems Simulation
	Specify Neuron as simulator and costruct the relevant hoc/mod/python sim file
	"""

	net_filename = simid+'.net.nml'							#NML2 description of network instance
	pynml.write_neuroml2_file( net_doc, net_filename )


	simid = 'sim_'+net.id
	ls = LEMSSimulation( simid, duration=duration, dt=dt, simulation_seed=seed )
	ls.assign_simulation_target( net.id )
	ls.include_neuroml2_file( net_filename)							#Network NML2 file
	ls.include_lems_file( '../Mechanisms/inputGenerators.xml')

	# Specify outputs

	eof0 = 'Events_file'
	ls.create_event_output_file(eof0, datadir+"%s.spikes.dat"%simid,format='ID_TIME')
	ctr=0
	for pid in range( p["nPop"] ):
		for jj in range( goc_pop[pid].size):
			ls.add_selection_to_event_output_file( eof0, ctr, '{}/{}/{}'.format( goc_pop[pid].id, jj, goc_type[pid].id), 'spike' )
			ctr += 1




	'''
	of0 = 'Volts_file'
	ls.create_output_file(of0, datadir+"%s.v.dat"%simid)
	ctr=0
	for pid in range( p["nPop"] ):
		for jj in range( goc_pop[pid].size ):
			ls.add_column_to_output_file(of0, ctr, '{}/{}/{}/v'.format( goc_pop[pid].id, jj, goc_type[pid].id))
			ctr +=1

	'''

	#Create Lems file to run
	lems_simfile = ls.save_to_file()


	# Create the Neuron-python simulation file (same name as LEMS - but with .py )

	if run:
		res = pynml.run_lems_with_jneuroml_neuron( lems_simfile, max_memory="2G", nogui=True, plot=False)
	else:
		res = pynml.run_lems_with_jneuroml_neuron( lems_simfile, max_memory="2G", only_generate_scripts = True, compile_mods = False, nogui=True, plot=False)
		# -----> before running sim, remember to compile mod files. Then run the LEMS***.py file

	return res


def get_params( runid=0, hom=False, nGJ_dend=3 ):

	Input_prob = {	"MF_bg" : 0.3,
					"PF_bg" : 0.5
				 }
	nMF = 60
	nPF = 150

	nPop=1 if hom else 5

	nInputs_max = { "MF_bg": nMF, "PF_bg" : nPF }
	nInputs_frac = { "MF_bg": 1.0, "PF_bg" : 1.0}
	Input_dend = { "PF_bg": nGJ_dend, "MF_bg": nGJ_dend,}

	params = inp.get_simulation_params( runid, nInputs_max=nInputs_max, nInputs_frac=nInputs_frac, nGoC_pop = nPop, Input_prob=Input_prob, nGJ_dend=nGJ_dend, Input_dend=Input_dend )
	return params



if __name__ =='__main__':
	runid=0

	print('Generating network using parameters for runid=', runid)
	res = create_GoC_network( duration = 2000, dt=0.025, seed = 123, runid=runid)
	print(res)
