'''
Defines functions for making the matrix based population.
'''

import numpy as np
import pandas as pd
from . import utils as cvu
from . import base as cvb
from . import defaults as cvd
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Matrix:
	@staticmethod
	def read_density(filepath):
		density = dict()
		with open(filepath, 'r') as fp:
			for line in fp:
				line = line.strip()
				if line:
					qkey, val = line.split('\t')
					density[qkey] = float(val)
		return density

	@staticmethod
	def read_tile(filepath):
		tiles = []
		with open(filepath, 'r') as fp:
			for line in fp:
				line = line.strip()
				if line:
					tiles.append(line)
		return tiles

	@staticmethod
	def generate_home_contact(age_based_uids, hm):
		n_age_bins = 16
		used_uids = [0] * n_age_bins
		p1, p2 = [], []
		avg_home_size, total_home = 0, 0
		for i in range(0, n_age_bins):
		# for i in range(n_age_bins-1, -1, -1):
			start, n_uids = used_uids[i], len(age_based_uids[i])
			for ui in range(start, n_uids):
				if ui < used_uids[i]:
					continue
				curr_uid = age_based_uids[i][ui]
				home = [curr_uid]
				used_uids[i] += 1
				for j in range(i, n_age_bins):
				# for j in range(i, -1, -1):
					num_contact = cvu.poisson(hm[i, j])
					start, end = used_uids[j], min(used_uids[j] + num_contact, len(age_based_uids[j]))
					used_uids[j] = end
					for k in range(start, end):
						home.append(age_based_uids[j][k])
				total_home += 1
				avg_home_size += (len(home) - avg_home_size) / total_home
				for s in range(len(home)):
					for t in range(s+1, len(home)):
						p1.append(home[s]); p2.append(home[t])
		return p1, p2, avg_home_size

	@staticmethod
	def generate_ws_contact(age_based_uids, matrix):
		n_age_bins = 16
		p1, p2 = [], []
		avg_contact_per_age = [0] * n_age_bins
		for i in range(n_age_bins):
			ni_uids = len(age_based_uids[i])
			for j in range(i, n_age_bins):
				nj_uids = len(age_based_uids[j])
				actual_contacts = np.zeros(nj_uids, dtype=cvd.default_int)
				num_contacts = cvu.n_poisson(matrix[i, j], ni_uids)
				for ui in range(ni_uids):
					req_contact = num_contacts[ui]
					if i == j:
						req_contact = max(req_contact-actual_contacts[ui], 0)
					start, end = ui+1, ui+1+req_contact
					for uj in range(start, end):
						if uj >= nj_uids:
							uj = (uj % nj_uids)
						if age_based_uids[i][ui] == age_based_uids[j][uj]:  # avoid self loop
							continue
						if i == j: # calculate actual contacts if both are in same bins
							actual_contacts[ui] += 1
						actual_contacts[uj] += 1 # calculate actual contacts
						p1.append(age_based_uids[i][ui])
						p2.append(age_based_uids[j][uj])
						avg_contact_per_age[j] += 1
						avg_contact_per_age[i] += 1

		for i in range(n_age_bins):
			avg_contact_per_age[i] /= len(age_based_uids[i])
		return p1, p2, avg_contact_per_age

	@staticmethod
	def get_same_tile_contact(tile_based_uids, matrix, ctype):
		p1, p2 = [], []
		curr_size, tile_info = 0, []
		avg_home_size, total_tiles = 0, 0
		n_age_bins = 16
		avg_ws_contact_per_age = [0] * n_age_bins
		for age_based_uids in tile_based_uids:
			total_tiles += 1
			if ctype == 'home':
				qp1, qp2, qavg_size = Matrix.generate_home_contact(age_based_uids, matrix)
				avg_home_size += (qavg_size - avg_home_size) / total_tiles
			elif ctype == 'work' or ctype == 'school':
				qp1, qp2, qavg_size = Matrix.generate_ws_contact(age_based_uids, matrix)
				for i in range(n_age_bins):
					avg_ws_contact_per_age[i] += (qavg_size[i] - avg_ws_contact_per_age[i]) / total_tiles
			else:
				raise Exception(f'ctype should be either home, work or school, but was given {ctype}')
			p1.append(qp1); p2.append(qp2)
			curr_size += len(qp1)
			tile_info.append(curr_size)
		# print statements for verification 
		# if ctype == 'home':
		# 	print(f'Avg home size: {avg_home_size}')
		# if ctype == 'work' or ctype == 'school':
		# 	print(f'Avg contact per age bin in layer {ctype}')
		# 	for i in range(n_age_bins):
		# 		print(f'age_bin_{i} : {avg_ws_contact_per_age[i]}')
		output = dict(p1=np.concatenate(p1, dtype=cvd.default_int), p2=np.concatenate(p2, dtype=cvd.default_int))
		return output, tile_info

	@staticmethod
	def get_tile_id(prob, cum_probs):
		l, r = 0, len(cum_probs)-1
		while (l < r):
			mid = l + (r-l)//2
			if cum_probs[mid] > prob:
				r = mid
			else:
				l = mid + 1
		return l

	@staticmethod
	def get_community_contact(tile_based_uids, mobility, cm, contact_factor):
		n_tiles, n_age_bins = len(tile_based_uids), 16
		used_uids = [[0]*n_age_bins for t in range(n_tiles)]
		p1, p2 = [], []
		for t in range(n_tiles):
			age_based_uids = tile_based_uids[t]
			cum_mobility = mobility[t].cumsum()
			for i in range(n_age_bins):
				n_uids = len(age_based_uids[i])
				for j in range(n_uids):
					source_uid = age_based_uids[i][j]
					probs = np.random.rand(n_age_bins)
					for k in range(n_age_bins):
						# num_contact = int(cvu.poisson(cm[i, k])/2)  #division by 2.0 is necessary so that we donot double the expected contact 
						num_contact = cvu.poisson(cm[i, k]) if probs[k] > 0.5 else 0  # it is similar to dividing by 2
						if num_contact > 0:
							tile_id = Matrix.get_tile_id(probs[k], cum_mobility)
							selected_uids = tile_based_uids[tile_id][k]
							selected_uids_length = len(selected_uids)
							num_contact = int(num_contact * contact_factor[tile_id]) # how does contact factor changes based on tile
							start, end = used_uids[tile_id][k], used_uids[tile_id][k]+num_contact
							used_uids[tile_id][k] = (end % selected_uids_length)
							for pos in range(start, end):
								pos = (pos % selected_uids_length)
								if source_uid == selected_uids[pos]: # skip self loop
									continue
								p1.append(source_uid); p2.append(selected_uids[pos])
		output = dict(p1=np.array(p1, dtype=cvd.default_int), p2=np.array(p2, dtype=cvd.default_int))
		return output

	@staticmethod
	def display_synthetic_contact_matrix(contacts, ages, actual_matrix, name):
		print(f'Layer name: {name}')
		bin_size, n_age_bins = 5, 16

		pop_by_age = np.zeros(n_age_bins)
		for i in range(len(ages)):
			idx = 15 if ages[i] >=75 else int(ages[i]/bin_size)
			pop_by_age[idx] += 1

		p1, p2 = contacts['p1'], contacts['p2']
		syn_matrix = np.zeros((n_age_bins, n_age_bins))
		for i in range(len(p1)):
			m = 15 if ages[p1[i]] >= 75 else int(ages[p1[i]]/bin_size)
			n = 15 if ages[p2[i]] >= 75 else int(ages[p2[i]]/bin_size)
			syn_matrix[m, n] += 1
			syn_matrix[n, m] += 1

		syn_matrix /= pop_by_age[:, np.newaxis]

		# np.savetxt('tmp.csv', syn_matrix, delimiter=',')

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))		
		sns.heatmap(syn_matrix, linewidth=0.5, cmap="Blues", ax=ax1)
		ax1.set_title(f'synthetic {name} contact matrix')
		if actual_matrix is not None:
			sns.heatmap(actual_matrix, linewidth=0.5, cmap="Blues", ax=ax2)
			ax2.set_title(f'actual {name} contact matrix')

		# plt.savefig(f'syn_{name}_contact_matrix.png')
		plt.show()

	@staticmethod
	def make_population(pars, pop_size, ages):
	    pop_size = int(pop_size) # Number of people

	    bin_size, n_age_bins = 5, 16
	    age_based_uids = [[] for i in range(n_age_bins)]
	    for uid in range(pop_size):
	        age_idx = -1 if ages[uid]>=75 else int(ages[uid]/bin_size)
	        age_based_uids[age_idx].append(uid)

	    tiles = Matrix.read_tile(pars['tiles'])
	    pop_density = Matrix.read_density(pars['pop_density'])
	    tile_based_uids = []
	    used_uids = [0] * n_age_bins
	    for idx, qkey in enumerate(tiles):
	    	tile_density = pop_density[qkey]
	    	tile_age_uids = [[] for i in range(n_age_bins)]
	    	for i, bins in enumerate(age_based_uids):
	    		start, end = used_uids[i], used_uids[i] + int(len(bins) * tile_density) + 1
	    		end = min(end, len(bins))
	    		for j in range(start, end):
	    			tile_age_uids[i].append(bins[j])
	    		used_uids[i] = end
	    	tile_based_uids.append(tile_age_uids)

	    contacts = dict()
	    tile_info = cvb.TileInfo()

	    # home contact
	    hm = np.genfromtxt(pars['home_matrix'], delimiter=' ')
	    contacts['h'], hinfo = Matrix.get_same_tile_contact(tile_based_uids, hm, 'home')
	    tile_info.set_key(hinfo, 'h')
	    # Matrix.display_synthetic_contact_matrix(contacts['h'], ages, hm, 'home')

	    # work contact
	    wm = np.genfromtxt(pars['work_matrix'], delimiter=' ')
	    contacts['w'], winfo = Matrix.get_same_tile_contact(tile_based_uids, wm, 'work')
	    tile_info.set_key(winfo, 'w')
	    # Matrix.display_synthetic_contact_matrix(contacts['w'], ages, wm, 'work')

	    # school contact
	    sm = np.genfromtxt(pars['school_matrix'], delimiter=' ')
	    contacts['s'], sinfo = Matrix.get_same_tile_contact(tile_based_uids, sm, 'school')
	    tile_info.set_key(sinfo, 's')
	    # Matrix.display_synthetic_contact_matrix(contacts['s'], ages, sm, 'school')

	    # community contact
	    cm = np.genfromtxt(pars['community_matrix'], delimiter=',')
	    mobility = np.genfromtxt(pars['mobility'], delimiter=',')
	    contact_factor = np.ones(len(tiles), dtype=float)
	    contacts['c'] = Matrix.get_community_contact(tile_based_uids, mobility, cm, contact_factor)
	    # Matrix.display_synthetic_contact_matrix(contacts['c'], ages, cm, 'community')

	    return contacts, tile_based_uids, tile_info, cm, contact_factor
