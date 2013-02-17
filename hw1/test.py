from dtree import *
import sys

def findEndNodes (tree, attrlist, attrlists, curAttr=None):
  "returns list of end nodes (nodes with only leaves attached) "
  #attrlists.append(attrlist)
  #print attrlists
  if tree.nodetype == DecisionTree.NODE:
    #print tree.attr
    if curAttr != None:
      attrlist.append(curAttr)
    #print attrlist
    print 'node'
    #print attrlists
    print tree.branches.values()
    if every(lambda x: x.nodetype == DecisionTree.LEAF, tree.branches.values()):
    	print 'end node'
    	#print attrlist
    	#print attrlists
    	#holder = copy.deepcopy(attrlist)
    	attrlists.append(attrlist)

    else:
      print 'not end node'
      print tree.branches.keys() 
      for a in tree.branches.keys():
        findEndNodes(tree.branches[a], attrlist, attrlists, a)
  return attrlists

# how to see if reach all leaves
a = DecisionTree(DecisionTree.LEAF, classification=1)
b = DecisionTree(DecisionTree.LEAF, classification=2)
c = DecisionTree(DecisionTree.LEAF, classification=0)

branch1 = {1 : a, 2 : b, 3 : c}

a2 = DecisionTree(DecisionTree.LEAF, classification=6)
b2 = DecisionTree(DecisionTree.LEAF, classification=7)
c2 = DecisionTree(DecisionTree.NODE, attr=5, branches = branch1)

branch2 = {1 : a2, 2 : b2, 3 : c2}

a3 = DecisionTree(DecisionTree.LEAF, classification=1)
b3 = DecisionTree(DecisionTree.LEAF, classification=2)
c3 = DecisionTree(DecisionTree.NODE)

branch3 = [a3, b3, c3]

tree3 = DecisionTree(DecisionTree.NODE, attr=4, branches = branch2)
tree3.display()
leaves = []
leaves = findEndNodes(tree3, [], [])
leavesp = copy.deepcopy
print leaves
print leaves[0]
tree3.collapse(leaves[0])
tree3.display()
#ans = every(lambda x: x.branches.nodetype == DecisionTree.LEAF, tree)
#print ans
