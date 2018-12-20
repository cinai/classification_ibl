'''
Hidden markov model module.

There are:
	- Hidden states
	- Observations represented as a list of feature vectors
	- Probability distributions of feature vectors per state
'''

import numpy as np

class HMM:

	'''
	Defines instance variables of an HMM

	Args:
		p0: vector with the n states' prior probabilites
		tpm: transition probability matrix (n x n)
		f: vector with the n probability density functions. f will indicate
		the probability that an observed feature vector is related to each
		state.
		mean: lista de medias
		cov: lista de covarianzas
	'''
	def __init__(self,p0,tpm,f,mean,cov,vh):
		self.p0 = p0
		self.n = len(p0)
		assert self.n == len(tpm)
		self.tpm = tpm
		self.f = f
		self.mean = mean
		self.cov = cov
		self.vh = vh

	'''
	Calculates the sequence of states that maximizes the probability of
	being produced by the set of observations. 

	Args:
		observations (list) : list of feature vectors 
	Returns:
		sequence_of_states (list)

	'''
	def get_sequence_of_states(self,observations):
		n = self.n
		t = len(observations)
		V = np.zeros((n,t))
		S = np.zeros((n,t))*-1
		f_observations = [get_f_observations(o) for o in observations]
		self.get_probabilities_first_observation(V,f_observations[0])
		self.fill_matrix(V,S,f_observations)
		sequence_of_states = self.find_best_sequence(V[:,-1],S)

		return sequence_of_states

	'''
	Applies self.f_i to the observation, with i: 0...n-1

	Args:
		observation : feature vector
	'''
	def get_f_observations(observation):
		return [f_i(np.matmul(self.vh,o)) for f_i in f] # i: 0...n-1

	'''
	Uses the prior probability and the first observation to calculate
	the first column of the V matrix.

	Args:
		V: n x n matrix 
		f_o : probability of the first observation to belong to the states
	'''
	def get_probabilities_first_observation(self,V,f_o):
		for i in range(self.n):
			V[0,:] = self.p[i]*f_o[i]

	'''
	Get the probabilities of the last stage to be a certain state

	Args:
		v_last: V column of the last stage
		state (integer) : indicates a certain state
		f_o: probability of the observation to belong to a certain state
	'''
	def get_last_stage_probabilities(self,v_last,state,f_o):
		probabilities = []
		for last_state in range(self.n):
			p = v_last[last_state]*self.tpm[last_state,state]*f_o
			probabilities.append(p)
		return probabilities

	'''
	Fill the V matrix according to Viterbi algorithm
	
	Args:
		V: n x n matrix
		S: n x n matrix, save the previous most probable state for each (state,stage)
		f_observations: probability of the observations to belong to the states
	'''
	def fill_matrix(self,V,S,f_observations):
		tpm = self.tpm
		for j in range(1,len(f_observations)): #j: 1...t-1
			for i in range(self.n):
				f_o = f_observations[j][i]
				p_last_stage = get_last_stage_probabilities(V[:,j-1],i,f_o)
				S[i,j] = np.argmax(p_last_stage)
				V[i,j] = p_last_stage[S[i,j]]

	'''
	Find the best sequence using as starting point the maximum state from
	the last column of the V matrix; and then, finding the path indicated
	through the S matrix.

	Args:
		v_final: last column of V matrix
		S: n x n,  save the previous most probable state for each (state,stage)
		f_observations: probability of the observations to belong to the states
	'''
	def find_best_sequence(self,v_final,S):
		stage = np.argmax(v_final)
		best_sequence = [stage]
		for j in range(S.shape[1]-1,0,-1):
			stage = S[j,stage]
			best_sequence.insert(0,stage)
		return best_sequence