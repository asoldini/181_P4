# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey

# ranges for q-table values

class Learner(object):
	'''
	This agent jumps randomly.
	'''

	def __init__(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.alpha = 0.6
		self.epsilon = 0.001
		self.lastV = None
		self.g = npr.choice([1,4])
		self.gassigned = False
		self.stateSeq = []
		self.rewards = []
		self.Q_table = {}
		for i in range(-18,18):
			for j in range(-8,30):
				for bot in range(-5,18):
					for vel in range(-8,8):
						for g in range(1,5,3):
							self.Q_table[(i,j,bot,vel,g)]=[0,0]

	def reset(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.g = npr.choice([1,4])
		self.gassigned = False
		self.rewards = []
		self.stateSeq = []
		# self.alpha *= 0.999

	def low_grav_bins(self, horzDist):
		if horzDist < 0:
			return -1
		elif horzDist < 100:
			return 1
		elif horzDist < 150:
			return 2
		elif horzDist < 200:
			return 3
		elif horzDist < 240:
			return 4
		elif horzDist < 270:
			return 5
		elif horzDist < 300:
			return 6
		elif horzDist < 330:
			return 7
		elif horzDist < 360:
			return 8
		elif horzDist < 390:
			return 9
		elif horzDist < 420:
			return 10
		elif horzDist < 460:
			return 11
		elif horzDist < 500:
			return 12
		elif horzDist < 550:
			return 13
		elif horzDist < 600:
			return 14

	def action_callback(self, state):
		'''
		Implement this function to learn things and take actions.
		Return 0 if you don't want to jump and 1 if you do.
		'''

		# You might do some learning here based on the current state and the last state.

		# You'll need to select and action and return it.
		# Return 0 to swing and 1 to jump.

  #       { ’score’: <current score>,
  # ’tree’: { ’dist’: <pixels to next tree trunk>,
  #        ’top’: <height of top of tree trunk gap>,
  #        ’bot’: <height of bottom of tree trunk gap> },
  # ’monkey’: { ’vel’: <current monkey y-axis speed>,
  #         ’top’: <height of top of monkey>,
  #         ’bot’: <height of bottom of monkey> }}

		# here is where I need to update the various tables
		horzDist = state['tree']['dist'] 
		vertDist = state['monkey']['bot'] - state['tree']['bot']
		botDist = state['monkey']['bot']
		vel = state['monkey']['vel']
		
		if self.g == 4:
			horzbin = horzDist // 50
			velbin = vel // 40
			vertbin = vertDist // 30
		else: 
			horzbin = horzDist// 25 #self.low_grav_bins(horzDist)
			velbin = vel // 20
			vertbin = vertDist // 20
		

		botbin = botDist // 20
		
		# else:
		# 	if horzDist >  

		max_action = np.argmax(self.Q_table[(vertbin,horzbin,botbin,velbin,self.g)])
		action = self.Q_table[(vertbin,horzbin,botbin,velbin,self.g)][max_action]


		if len(self.stateSeq) != 0:
			if self.last_action == 0 and self.gassigned == False:
				self.g = self.lastV - vel
				self.gassigned = True

		self.stateSeq.append(((vertbin,horzbin,botbin,velbin,self.g), max_action))

		if self.last_reward != None and self.last_reward < 0:
			self.rewards.reverse()
			self.stateSeq.reverse()
			for i, (S, A) in enumerate(self.stateSeq):
				if i==0:
					pass
				elif i == 1:
					self.Q_table[S][A] = (1 - self.alpha) * self.Q_table[S][A] + self.alpha * self.rewards[i-1]
					prevState = S
				else:
					self.Q_table[S][A] = (1 - self.alpha) * self.Q_table[S][A] + self.alpha * (self.rewards[i-1] + 0.9 * max(self.Q_table[prevState]))
					prevState = S
	
		self.last_action = max_action

		if np.random.random() < self.epsilon:
			self.last_action = -1 * (self.last_action - 1)
			self.stateSeq[-1] = ((vertbin,horzbin,botbin,velbin,self.g),self.last_action)
		self.lastV = vel
		# self.last_state  = (vertbin,horzbin,botbin,velbin,self.g)

		return self.last_action

	def reward_callback(self, reward):
		'''This gets called so you can see what reward you get.'''
		self.rewards.append(reward)
		self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
	'''
	Driver function to simulate learning by having the agent play a sequence of games.
	'''
	gravity = []
	for ii in range(iters):
		# Make a new monkey object.
		swing = SwingyMonkey(sound=False,                  # Don't play sounds.
							 text="Epoch %d" % (ii),       # Display the epoch on screen.
							 tick_length = t_len,          # Make game ticks super fast.
							 action_callback=learner.action_callback,
							 reward_callback=learner.reward_callback)

		# Loop until you hit something.
		while swing.game_loop():
			pass
		
		# Save score history.
		hist.append(swing.score)
		gravity.append(swing.gravity)
		# Reset the state of the learner.
		learner.reset()
	pg.quit()
	return gravity


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	gravity = run_games(agent, hist, 500, 1)
	print(np.mean(hist))
	print(hist)
	# np.save('nograv',np.array(gravity))
	# Save history. 
	np.save('histcomp8',np.array(hist))


