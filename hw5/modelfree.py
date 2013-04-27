from random import *
import throw
import darts
 
# The default player aims for the maximum score, unless the
# current score is less than the number of wedges, in which
# case it aims for the exact score it needs. 
#  
# You may use the following functions as a basis for 
# implementing the Q learning algorithm or define your own 
# functions.

def start_game():

  return(throw.location(throw.INNER_RING, throw.NUM_WEDGES)) 

def get_target(score):

  if score <= throw.NUM_WEDGES: return throw.location(throw.SECOND_PATCH, score)
  
  return(throw.location(throw.INNER_RING, throw.NUM_WEDGES))


# Exploration/exploitation strategy one.
def ex_strategy_one():
  return 0


# Exploration/exploitation strategy two.
def ex_strategy_two():
  return 1


# The Q-learning algorithm:
def Q_learning(gamma, num_games):
	# initialize Q's
	Q = []
	states = darts.get_states()
	actions = darts.get_actions()
	for s in states:
		for a in range(len(actions)):
			Q[s][a] = 0
			# A = {a:0}
			# Q[s] = A

	# run Q-learning
	for g in range(1, num_games + 1):
    
  	# run a single game
    s = throw.START_SCORE
    while s > 0:
      	# The following two statements implement two exploration-exploitation
        # strategies. Comment out the strategy that you wish not to use.
			
    	to_explore = ex_strategy_one(num_iterations)
      #to_explore = ex_strategy_two(num_iterations)

      if to_explore:
       	# explore
       	a = random.randint(0, len(actions)-1)
       	action = actions[a]
      else:
        # exploit
        q_values = Q[s]
       	a = q_values.index(max(q_values))
       	action = actions[a]

  return






