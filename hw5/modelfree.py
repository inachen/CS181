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
EPSILON = 0.5
def ex_strategy_one(num_iterations):
  if random.random() <= float(EPSILON/num_iterations):
    return 1
  return 0


# Exploration/exploitation strategy two.
TIME = 5
def ex_strategy_two(num_iterations):
  if num_iterations < TIME:
    return 1
  return 0


# The Q-learning algorithm:
# <<<<<<< HEAD
# def Q_learning(gamma, num_games):
# 	# initialize Q's
# 	Q = []
# 	states = darts.get_states()
# 	actions = darts.get_actions()
# 	for s in states:
# 		for a in range(len(actions)):
# 			Q[s][a] = 0
# 			# A = {a:0}
# 			# Q[s] = A

# 	# run Q-learning
# 	for g in range(1, num_games + 1):
    
#   	# run a single game
#     s = throw.START_SCORE
#     while s > 0:
#       	# The following two statements implement two exploration-exploitation
#         # strategies. Comment out the strategy that you wish not to use.
			
#     	to_explore = ex_strategy_one(num_iterations)
#       #to_explore = ex_strategy_two(num_iterations)

#       if to_explore:
#        	# explore
#        	a = random.randint(0, len(actions)-1)
#        	action = actions[a]
#       else:
#         # exploit
#         q_values = Q[s]
#        	a = q_values.index(max(q_values))
#        	action = actions[a]
# =======
def Q_learning(gamma, learning_rate, num_games):

	# store all actions (targets on dartboard) in actions array
    actions = darts.get_actions()
    states = darts.get_states()


    g = 0
    num_actions = {}
    num_transitions = {}
    T_matrix = {}
    q_values = {}
    
    
    # Initialize all arrays to 0 except the policy, which should be assigned a random action for each state.
    for s in states:
        num_actions[s] = {}
        num_transitions[s] = {}
        T_matrix[s] = {}
        q_values[s] = {}
        
        for a in range(len(actions)):
            num_actions[s][a] = 0
            q_values[s][a] = 0

        for s_prime in states:
            num_transitions[s][s_prime] = {}
            T_matrix[s][s_prime] = {}
            for a in range(len(actions)):
                num_transitions[s][s_prime][a] = 0
                T_matrix[s][s_prime][a] = 0

    for g in range(1, num_games + 1):
    
    	# run a single game
        s = throw.START_SCORE
        while s > 0:

        	# which strategy to use
        	to_explore = ex_strategy_one(num_iterations)
    	    #to_explore = ex_strategy_two(num_iterations)
    		
            if to_explore:
            	# explore
            	a = random.randint(0, len(actions)-1)
            	action = actions[a]
            else:
            	# exploit
            	a = pi_star[s]
            	action = actions[a]

  return






