from dtree import *
import sys

def scoreTree(learner, dataset):
    "Give a score for a tree on a dataset between 0 and 1"

    # retrieve examples
    examples = dataset.examples[:]

    # keep track of the number it trains correctly
    numCorrect = 0
    for i in xrange(len(examples)):
        if learner.predict(examples[i]) == examples[i].attrs[dataset.target]:
            numCorrect = numCorrect + 1

    return numCorrect / float(len(examples))

# =========================
# Test Trees
# =========================
# a = DecisionTree(DecisionTree.LEAF, classification=1)
# b = DecisionTree(DecisionTree.LEAF, classification=2)
# c = DecisionTree(DecisionTree.LEAF, classification=0)

# branch1 = {1 : a, 2 : b, 3 : c}

# a2 = DecisionTree(DecisionTree.LEAF, classification=6)
# b2 = DecisionTree(DecisionTree.LEAF, classification=7)
# c2 = DecisionTree(DecisionTree.NODE, attr=5, branches = branch1)

# branch2 = {1 : a2, 2 : b2, 3 : c2}

# a3 = DecisionTree(DecisionTree.LEAF, classification=1)
# b3 = DecisionTree(DecisionTree.LEAF, classification=2)
# c3 = DecisionTree(DecisionTree.NODE)

# branch3 = [a3, b3, c3]

# tree3 = DecisionTree(DecisionTree.NODE, attr=4, branches = branch2)
# tree3.display()

# =========================
# FindEndNodes testing
# =========================

def findEndNodes (tree, attrlist, attrlists, curAttr=None):
  "returns list of end nodes (nodes with only leaves attached) "
  #attrlists.append(attrlist)
  #print attrlists
  if tree.nodetype == DecisionTree.NODE:
    #print tree.attr
    if curAttr != None:
      attrlist.append(curAttr)
    #print attrlist
    # print 'node'
    # print attrlists
    # print tree.branches.values()
    if every(lambda x: x.nodetype == DecisionTree.LEAF, tree.branches.values()):
      # print 'end node'
      #print attrlist
      #print attrlists
      #holder = copy.deepcopy(attrlist)
      attrlists.append(attrlist)

    else:
      # print 'not end node'
      # print tree.branches.keys() 
      for a in tree.branches.keys():
        cplist = copy.deepcopy(attrlist)
        findEndNodes(tree.branches[a], cplist, attrlists, a)
  return attrlists

def prune (learner, dataset):

  learner.dt.display()

  stumps = findEndNodes (learner.dt, [], [])

  # print stumps

  for s in stumps:
    pLearner = copy.deepcopy(learner)
    pLearner.dt.collapse(s)
    print scoreTree(pLearner, dataset)
    print scoreTree(learner, dataset)
    if scoreTree(pLearner, dataset) >= scoreTree(learner, dataset):
      print 'pruned'
      return pLearner

# leaves = []
# leaves = findEndNodes(tree3, [], [])
# leavesp = copy.deepcopy
# print leaves
# print leaves[0]
# tree3.collapse(leaves[0])
# tree3.display()
#ans = every(lambda x: x.branches.nodetype == DecisionTree.LEAF, tree)
#print ans

# =========================
# Prune testing
# =========================

example1 = Example([1,0,1,0,0])
example2 = Example([1,0,0,0,1])
example3 = Example([0,0,0,1,0])
example4 = Example([1,0,0,1,1])
example5 = Example([0,1,1,1,1])
example6 = Example([1,1,0,1,0])

examples1 = [example1,example2,example3,example4,example5,example6]

dataset1 = DataSet(examples1)

learner = DecisionTreeLearner()
learner.train(dataset1)

# learner.dt.display()
# print learner.dt
# print learner.dt.branches

# print findEndNodes(learner.dt, [],[])

prune(learner, dataset1)