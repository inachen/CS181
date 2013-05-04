import random
import throw
import darts
 
# The default player aims for the maximum score, unless the
# current score is less than the number of wedges, in which
# case it aims for the exact score it needs. 
#  
# You may use the following functions as a basis for 
# implementing the Q learning algorithm or define your own 
# functions.


# store all actions (targets on dartboard) in actions array

q_values = []

def start_game(gamma, learning_rate, num_games):

    actions = darts.get_actions()
    states = darts.get_states()

    Q_learning(gamma, learning_rate, num_games)

    a = q_values[throw.START_SCORE].index(max(q_values[throw.START_SCORE]))

    return actions[a]

def get_target(score):

    actions = darts.get_actions()
    
    a = q_values[score].index(max(q_values[score]))

    return actions[a]



# Define your first exploration/exploitation strategy here. Return 0 to exploit and 1 to explore. 
# You may want to pass arguments from the modelbased function. 
# Strategy 1: epsilon greedy
EPSILON = 0.5
def ex_strategy_one(num_iterations):
    if random.random() <= float(EPSILON/num_iterations):
        return 1
    return 0


# Define your first exploration/exploitation strategy here. Return 0 to exploit and 1 to explore. 
# You may want to pass arguments from the modelbased function.
# Strategy 2: Explore for a while before exploting

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


    g = 0
    num_iterations = 0

	# store all actions (targets on dartboard) in actions array
    # actions = darts.get_actions()
    # states = darts.get_states()


    actions = darts.get_actions()

    states = darts.get_states()
    
    # Initialize all arrays to 0 except the policy, which should be assigned a random action for each state.
    for s in states:
        list_a = []
        for a in range(len(actions)):
            list_a.append(0)
        q_values.append(list_a)

    for g in range(1, num_games + 1):

    	# run a single game
        s = throw.START_SCORE
        while s > 0:

            num_iterations += 1

        	# which strategy to use
            #to_explore = ex_strategy_one(num_iterations)
            to_explore = ex_strategy_two(num_iterations)

            if to_explore:
            	# explore
                a = random.randint(0, len(actions)-1)
                action = actions[a]
            else:
            	# exploit
                a = q_values[s].index(max(q_values[s]))
                action = actions[a]

    # g = 0
    # num_actions = {}
    # num_transitions = {}
    # T_matrix = {}
    # q_values = {}
    
    
    # # Initialize all arrays to 0 except the policy, which should be assigned a random action for each state.
    # for s in states:
    #     num_actions[s] = {}
    #     num_transitions[s] = {}
    #     T_matrix[s] = {}
    #     q_values[s] = {}
        
    #     for a in range(len(actions)):
    #         num_actions[s][a] = 0
    #         q_values[s][a] = 0

    #     for s_prime in states:
    #         num_transitions[s][s_prime] = {}
    #         T_matrix[s][s_prime] = {}
    #         for a in range(len(actions)):
    #             num_transitions[s][s_prime][a] = 0
    #             T_matrix[s][s_prime][a] = 0

    # for g in range(1, num_games + 1):
    
    # 	# run a single game
    #     s = throw.START_SCORE
    #     while s > 0:

    #     	# which strategy to use
    #     	to_explore = ex_strategy_one(num_iterations)
    # 	    #to_explore = ex_strategy_two(num_iterations)
    		
    #         if to_explore:
    #         	# explore
    #         	a = random.randint(0, len(actions)-1)
    #         	action = actions[a]
    #         else:
    #         	# exploit
    #         	a = pi_star[s]
    #         	action = actions[a]

  return


           	# Get result of throw from dart thrower; update score if necessary
            loc = throw.throw(action) 
            s_prime = int(s - throw.location_to_score(loc))
            if s_prime < 0:
                s_prime = s

            a_prime = q_values[s_prime].index(max(q_values[s_prime]))
            action_prime = actions[a_prime]
            # Update q value for the action we just performed
            q_values[s][a] = q_values[s][a] + learning_rate * (darts.R(s, actions[a]) + gamma * q_values[s_prime][a_prime] - q_values[s][a])

            # Next state becomes current state 
            s = s_prime

    return

   
