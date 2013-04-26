

# Components of a darts player. #

# 
 # Modify the following functions to produce a player.
 # The default player aims for the maximum score, unless the
 # current score is less than or equal to the number of wedges, in which
 # case it aims for the exact score it needs.  You can use this
 # player as a baseline for comparison.
 #

from random import *
import throw
import darts

# make pi global so computation need only occur once
PI = {}
EPSILON = .001

# define probabilities for hitting regions
PROBWEDGE = 0.4
PROBW1 = 0.2
PROBW2 = 0.1

PROBRING = 0.4
PROBR1 = 0.2
PROBR2 = 0.1


# actual
def start_game(gamma):

  infiniteValueIteration(gamma)
  for ele in PI:
    print "score: ", ele, "; ring: ", PI[ele].ring, "; wedge: ", PI[ele].wedge
  
  return PI[throw.START_SCORE]

def get_target(score):

  return PI[score]

# define transition matrix/ function
def T(a, s, s_prime):
  # takes an action a, current state s, and next state s_prime
  # returns the probability of transitioning to s_prime when taking action a in state s
  possible_rings = []
  ring_prob = []
  if (a.ring == throw.CENTER):
    possible_rings = [throw.CENTER,throw.INNER_RING,throw.FIRST_PATCH]
    ring_prob = [PROBRING,2*PROBR1,2*PROBR2]
  elif (a.ring == throw.INNER_RING):
    possible_rings = [throw.CENTER,throw.INNER_RING,throw.FIRST_PATCH,throw.MIDDLE_RING]
    ring_prob = [PROBR1,PROBRING+PROBR1,PROBR1,PROBR2]
  elif (a.ring == throw.FIRST_PATCH):
    possible_rings = [throw.CENTER,throw.INNER_RING,throw.FIRST_PATCH,throw.MIDDLE_RING,throw.SECOND_PATCH]
    ring_prob = [PROBR2,PROBR1,PROBRING,PROBR1,PROBR2]
  elif (a.ring == throw.MIDDLE_RING):
    possible_rings = [throw.INNER_RING,throw.FIRST_PATCH,throw.MIDDLE_RING,throw.SECOND_PATCH,throw.OUTER_RING]
    ring_prob = [PROBR2,PROBR1,PROBRING,PROBR1,PROBR2]
  elif (a.ring == throw.SECOND_PATCH):
    possible_rings = [throw.FIRST_PATCH,throw.MIDDLE_RING,throw.SECOND_PATCH,throw.OUTER_RING,throw.MISS]
    ring_prob = [PROBR2,PROBR1,PROBRING,PROBR1,PROBR2]
  elif (a.ring == throw.OUTER_RING):
    possible_rings = [throw.MIDDLE_RING,throw.SECOND_PATCH,throw.OUTER_RING,throw.MISS]
    ring_prob = [PROBR2,PROBR1,PROBRING,PROBR1+PROBR2]
  elif (a.ring == throw.OUTER_RING):
    possible_rings = [throw.MIDDLE_RING,throw.SECOND_PATCH,throw.OUTER_RING,throw.MISS]
    ring_prob = [PROBR2,PROBR1,PROBRING,PROBR1+PROBR2]
  elif (a.ring == throw.MISS):
    possible_rings = [throw.SECOND_PATCH,throw.OUTER_RING,throw.MISS]
    ring_prob = [PROBR2,PROBR1,PROBRING+PROBR1+PROBR2]

  w_index = throw.wedges.index(a.wedge)
  possible_wedges = [(a.wedge),(throw.wedges[(w_index+1)%throw.NUM_WEDGES]),(throw.wedges[(w_index-1)%throw.NUM_WEDGES]),(throw.wedges[(w_index+2)%throw.NUM_WEDGES]),(throw.wedges[(w_index-2)%throw.NUM_WEDGES])]
  wedge_prob = [PROBWEDGE,PROBW1,PROBW1,PROBW2,PROBW2]

  final_prob = 0

  for i in range(len(possible_rings)):
    for j in range(len(possible_wedges)):
      myloc = throw.location(possible_rings[i],possible_wedges[j])
      if (s - (throw.location_to_score(myloc))) == s_prime:
          final_prob = final_prob + (ring_prob[i]*wedge_prob[j])
  return final_prob



def infiniteValueIteration(gamma):
  # takes a discount factor gamma and convergence cutoff epislon
  # returns

  V = {}
  Q = {}
  V_prime = {}
  
  states = darts.get_states()
  actions = darts.get_actions()

  notConverged = True

  # intialize value of each state to 0
  for s in states:
    V[s] = 0
    Q[s] = {}

  # until convergence is reached
  while notConverged:

    # store values from previous iteration
    for s in states:
      V_prime[s] = V[s]

    # update Q, pi, and V
    for s in states:
      for a in actions:

        # given current state and action, sum product of T and V over all states
        summand = 0
        for s_prime in states:
          summand += T(a, s, s_prime)*V_prime[s_prime]

        # update Q
        Q[s][a] = darts.R(s, a) + gamma*summand

      # given current state, store the action that maximizes V in pi and the corresponding value in V
      PI[s] = actions[0]                                                        
      V[s] = Q[s][PI[s]]                                                        
      for a in actions:                                                         
        if V[s] <= Q[s][a]:                                                     
          V[s] = Q[s][a]                                                        
          PI[s] = a  

    notConverged = False
    for s in states:
      if abs(V[s] - V_prime[s]) > EPSILON:
        notConverged = True
      
# test_score = 9
# test_action = throw.location(throw.OUTER_RING,4)

# print T(test_action,test_score,5)