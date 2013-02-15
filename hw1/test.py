from dtree import *
import sys

# how to see if reach all leaves
a = DecisionTree(DecisionTree.LEAF, classification=1)
b = DecisionTree(DecisionTree.LEAF, classification=2)
c = DecisionTree(DecisionTree.NODE)

branch1 = [a, b, c]

a2 = DecisionTree(DecisionTree.LEAF, classification=1)
b2 = DecisionTree(DecisionTree.LEAF, classification=2)
c2 = DecisionTree(DecisionTree.NODE)

branch2 = [a2, b2, c2]

a3 = DecisionTree(DecisionTree.LEAF, classification=1)
b3 = DecisionTree(DecisionTree.LEAF, classification=2)
c3 = DecisionTree(DecisionTree.NODE)

branch3 = [a3, b3, c3]

tree = DecisionTree(DecisionTree.NODE, branches = branch1)

ans = every(lambda x: x.branches.nodetype == DecisionTree.LEAF, tree)
print ans
