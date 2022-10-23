# Generates file binetwork_params.pkl -> parameters for full model og Golgi network with ON/OFF MF and PF inputs
# Run as: python initialize_bimodal_network_params.py
# Or load module and call get_simulation_params() withh different arguments

import numpy as np
from numpy.core import multiarray
import pickle as pkl

import sys
sys.path.append('../PythonUtils')

import network_utils as nu


def get_simulation_params(simid,
						  sim_durn = 20000,
                          nGoC=0,
						  nGoC_pop = 5,
						  densityParams = '../Parameters/useParams_FI_14_25.pkl',
                          volume=[300, 300, 100],
						  GoC_density=4607,
						  GJ_dist_type='Boltzmann',
						  GJ_wt_type='Szo16_oneGJ',
						  nGJ_dend =3,
						  GoC_wtscale = 1.0,
						  GoC_probscale = 1.0,
						  bgInputs     = ["MF_bg", "PF_bg"],
						  input_nFiles= {"MF_bg": 60, "PF_bg" : 60},
						  nInputs_max = {"MF_bg": 60, "PF_bg" : 300},
						  nInputs_frac = { "MF_bg": .10, "PF_bg" : .35},
						  Input_density = {"MF_bg": 0, "PF_bg" : 0 },
						  Input_dend = { "PF_bg": 3, "MF_bg": 3,},
						  Input_nRosette = { "MF_bg": 0, "PF_bg" : 0 },
						  Input_loc  = { "MF_bg"   : 'random',
										 "PF_bg"   : 'random',
									   },
						  Input_type = { "MF_bg"   : 'poisson',
										 "PF_bg"   : 'poisson',
									   },
						  Input_id   = { "MF_bg"   : 0,
										 "PF_bg"   : 0,
									   },
						  Input_syn  = { "MF_bg"   : ['../Mechanisms/MF_GoC_Syn.nml', 'ExpThreeSynapse'],
										 "PF_bg"   : ['../Mechanisms/PF_GoC_Syn.nml', 'ExpTwoSynapse'],
									   },
						  Input_conn = { "MF_bg"   : 'random_prob',
										 "PF_bg"   : 'random_prob',
									   },
						  Input_prob = { "MF_bg"   : 0.3,
										 "PF_bg"   : 0.8,
									   },
						  Input_rate = { "MF_bg"   : [5],
										 "PF_bg"   : [2]
									   },
						  Input_nGoC = { "MF_bg" : 0,
										 "PF_bg" : 0,
									   },
						  Input_wt   = { "MF_bg" : 1,
										 "PF_bg" : 1,
									   },
						  Input_maxD = { "MF_bg" : [300],
										 "PF_bg" : [100, 2000, 300],
									   },
						  Input_cellloc = { "MF_bg" : 'soma',
											"PF_bg" : 'dend',
										   },
						  connect_goc = { "MF_bg": False,"PF_bg" : False}

						 ):


	params={}

	# 1. Golgi Population - Type, Location and Electrical Coupling

	params["nGoC"], params["GoC_pos"] = nu.locate_GoC( nGoC, volume, GoC_density, seed=simid)

	params["nPop"] = nGoC_pop
	params["GoC_ParamID"], params["nGoC"] = nu.get_hetero_GoC_id( params["nGoC"], nGoC_pop, densityParams, seed=simid )
	popsize = params["nGoC"]/params["nPop"]
	params["nGoC_per_pop"] = popsize
	params["nGoC"], params["GoC_pos"] = nu.locate_GoC( params["nGoC"], volume, GoC_density, seed=simid)

	print("Created parameters for {} GoC".format(params["nGoC"]) )
	usedID = np.unique( np.asarray( params["GoC_ParamID"]))
	# Get connectivity and sort into populations:
	params["econn_pop"] = [ [{} for post in range(nGoC_pop-pre)] for pre in range(nGoC_pop)]
	gj, wt, loc = nu.GJ_conn( params["GoC_pos"], GJ_dist_type, GJ_wt_type, nDend=nGJ_dend, wt_k=GoC_wtscale, prob_k=GoC_probscale, seed=simid )

	for pre in range( nGoC_pop):
		for post in range(pre,nGoC_pop):
			pairid = [x for x in range(gj.shape[0]) if ((np.floor_divide(gj[x,1],popsize)==post) & (np.floor_divide(gj[x,0],popsize)==pre)) ]
			params["econn_pop"][pre][post-pre] = { "GJ_pairs": np.mod( gj[pairid,:], popsize), "GJ_wt": wt[pairid], "GJ_loc": loc[pairid,:] }


	params["Inputs"] = { "types" : [] }

	ctr=1
	np.random.seed(simid+1000)
	for input in bgInputs:
		Inp = { "type" : Input_type[input], "rate": Input_rate[input], "syn_type": Input_syn[input] , "syn_loc": Input_cellloc[input]}
		if connect_goc[input]:
			choose_goc = np.random.randint( params["nGoC"] )
			Inp["nInp"], Inp["pos"], Inp["conn_pairs"], Inp["conn_wt"], Inp["conn_loc"] = nu.connect_inputs( maxn=nInputs_max[input], frac= nInputs_frac[input], density=Input_density[input], volume=volume, mult=Input_nRosette[input], connType=Input_conn[input], connProb=Input_prob[input], connGoC=Input_nGoC[input], connWeight=Input_wt[input], connDist=Input_maxD[input], GoC_pos=np.reshape(params["GoC_pos"][choose_goc,:],(1,3)), cell_loc=Input_cellloc[input], seed=simid)
			if len(Inp["conn_pairs"])>1:
				for jj in range(len(Inp["conn_pairs"][1])):
					Inp["conn_pairs"][1][jj] = choose_goc
		else:
			Inp["nInp"], Inp["pos"], Inp["conn_pairs"], Inp["conn_wt"], Inp["conn_loc"] = nu.connect_inputs( maxn=nInputs_max[input], frac= nInputs_frac[input], density=Input_density[input], volume=volume, mult=Input_nRosette[input], connType=Input_conn[input], connProb=Input_prob[input], connGoC=Input_nGoC[input], connWeight=Input_wt[input], connDist=Input_maxD[input], GoC_pos=params["GoC_pos"], cell_loc=Input_cellloc[input], nDend = Input_dend[input], seed=simid*ctr)
		tmp={ "conn_pairs": [], "conn_wt" : [], "conn_loc": [] }
		if Inp["nInp"]>0:
			np.random.seed(simid*Inp["nInp"]*ctr)
			Inp["sample"] = np.random.permutation(input_nFiles[input])[0:Inp["nInp"] ]
			for pid in usedID:
				pairid = [x for x in range(Inp["conn_pairs"].shape[1]) if params["GoC_ParamID"][Inp["conn_pairs"][1,x]]==pid ]
				tmp["conn_pairs"].append( Inp["conn_pairs"][:,pairid] )
				tmp["conn_wt"].append( Inp["conn_wt"][pairid] )
				tmp["conn_loc"].append( Inp["conn_loc"][:,pairid] )
			for key in ["conn_pairs", "conn_wt", "conn_loc"] :
				Inp[key] = tmp[key]
			for jj in range( nGoC_pop):
				Inp["conn_pairs"][jj][1,:]=np.mod(Inp["conn_pairs"][jj][1,:], popsize)
			params["Inputs"]["types"].append(input )

		params["Inputs"][input] = Inp
		ctr+=1

		print Inp["nInp"]
		params["Inputs"][input] = Inp
		ctr+=1

	return params



if __name__ =='__main__':

	nSim = 1
	params_list = [ get_simulation_params(simid) for simid in range(nSim) ]

	file = open('binetwork_params.pkl','wb')
	pkl.dump(params_list,file); file.close()
