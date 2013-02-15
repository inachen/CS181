# testing.py
# -------
# Kathy lin, Ina Chen

from dtree import *
from main import *
import sys

# =========================
# Toy data for testing
# =========================
example1 = Example([1,0,1,0,0])
example2 = Example([1,0,0,0,1])
example3 = Example([0,0,0,1,0])
example4 = Example([1,0,0,1,1])
example5 = Example([0,1,1,1,1])
example6 = Example([1,1,0,1,0])

# same examples, but with opposite labels
xexample1 = Example([1,0,1,0,1])
xexample2 = Example([1,0,0,0,0])
xexample3 = Example([0,0,0,1,1])
xexample4 = Example([1,0,0,1,0])
xexample5 = Example([0,1,1,1,0])
xexample6 = Example([1,1,0,1,1])

# make some weighted data
wexample1 = Example([1,0,1,0,0])
wexample1.weight = 0.5
wexample2 = Example([1,0,0,0,1])
wexample2.weight = 0.1
wexample3 = Example([0,0,0,1,0])
wexample3.weight = 0.0
wexample4 = Example([1,0,0,1,1])
wexample4.weight = 0.1
wexample5 = Example([0,1,1,1,1])
wexample5.weight = 0.2
wexample6 = Example([1,1,0,1,0])
wexample6.weight = 0.1

# same examples, but with opposite labels
wxexample1 = Example([1,0,1,0,1])
wxexample1.weight = 0.1
wxexample2 = Example([1,0,0,0,0])
wxexample2.weight = 0.2
wxexample3 = Example([0,0,0,1,1])
wxexample3.weight = 0.3
wxexample4 = Example([1,0,0,1,0])
wxexample4.weight = 0.2
wxexample5 = Example([0,1,1,1,0])
wxexample5.weight = 0.1
wxexample6 = Example([1,1,0,1,1])
wxexample6.weight = 0.1

examples1 = [example1,example2,example3,example4,example5,example6]
xexamples1 = [xexample1,xexample2,xexample3,xexample4,xexample5,xexample6]
halfexamples1 = [xexample1,xexample2,xexample3,example4,example5,example6]
wexamples = [wexample1,wexample2,wexample3,wexample4,wexample5,wexample6]
wxexamples = [wxexample1,wxexample2,wxexample3,wxexample4,wxexample5,wxexample6]


dataset1 = DataSet(examples1)
xdataset1 = DataSet(xexamples1)
halfdataset1 = DataSet(halfexamples1)
wdataset = DataSet([wexample1,wexample2,wexample3,wexample4,wexample5,wexample6])
wxdataset = DataSet([wxexample1,wxexample2,wxexample3,wxexample4,wxexample5,wxexample6])

# =========================
# Testing for scoreTree
# =========================
learned = DecisionTreeLearner()
learned.train(dataset1)
assert(scoreTree(learned, dataset1) == 1)
assert(scoreTree(learned, xdataset1) == 0)
assert(scoreTree(learned, halfdataset1) == 0.5)

# =========================
# Testing for vote
# =========================
learned = DecisionTreeLearner()
learned.train(dataset1)

xlearned = DecisionTreeLearner()
xlearned.train(xdataset1)

halflearned = DecisionTreeLearner()
halflearned.train(halfdataset1)

assert(vote([[learned,0],[xlearned,0.5]], example1) == 1)
assert(vote([[learned,0],[xlearned,6]], example1) == 1)
assert(vote([[learned,0.5],[xlearned,0.5]], example1) == 0)
assert(vote([[learned,0.49],[xlearned,0.5]], example1) == 1)
assert(vote([[learned,0.5],[xlearned,0.5],[halflearned,0.5]], example1) == 1)
assert(vote([[learned,0.5],[xlearned,0.5],[halflearned,0.5]], example5) == 1)

# =========================
# Testing for weightHyp
# =========================

whalfdataset = DataSet([wxexample1,wxexample2,wxexample3,wexample4,wexample5,wexample6])

learned = DecisionTreeLearner()
learned.train(wdataset)

assert(weightHyp(learned, wdataset) == sys.maxint)
assert(weightHyp(learned, wxdataset) == 0)
assert(abs(weightHyp(learned, whalfdataset) - 0.5 * math.log((0.4)/0.6)) < 0.0001)

# =========================
# Testing for weightData
# =========================

whalfdataset = DataSet([wxexample1,wxexample2,wxexample3,wexample4,wexample5,wexample6])

learned = DecisionTreeLearner()
learned.train(wdataset)

weightData(learned, whalfdataset, 1.5)
assert(abs(wxexample1.weight - 0.1*math.exp(1.5)/2.778265506) < 0.00001)
assert(abs(wxexample2.weight - 0.2*math.exp(1.5)/2.778265506) < 0.00001)
assert(abs(wxexample3.weight - 0.3*math.exp(1.5)/2.778265506) < 0.00001)
assert(abs(wexample4.weight - 0.1*math.exp(-1.5)/2.778265506) < 0.00001)
assert(abs(wexample5.weight - 0.2*math.exp(-1.5)/2.778265506) < 0.00001)
assert(abs(wexample6.weight - 0.1*math.exp(-1.5)/2.778265506) < 0.00001)

# =========================
# Testing for splitData
# =========================

dataset2 = copy.deepcopy(dataset1)
dataset3 = copy.deepcopy(dataset1)
dataset4 = copy.deepcopy(dataset1)

splits1 = splitData(dataset2,1)
t1 = splits1[0].examples
v1 = splits1[1].examples
splits2 = splitData(dataset3,3)
t2 = splits2[0].examples
v2 = splits2[1].examples
splits3 = splitData(dataset4,5)
t3 = splits3[0].examples
v3 = splits3[1].examples

assert(len(t1) == 5)
assert(len(v1) == 1)
assert(len(t2) == 3)
assert(len(v2) == 3)
assert(len(t3) == 1)
assert(len(v3) == 5)
#print [e.attrs for e in t2], [f.attrs for f in v2]


# ===================================
# Testing for countTree (in dtree.py)
# ===================================
tree1 = DecisionTree(DecisionTree.LEAF,classification=0)
tree2 = DecisionTree(DecisionTree.NODE)
tree2.add(0,tree1)
tree3 = DecisionTree(DecisionTree.NODE)
tree3.add(0,tree2)
tree3.add(1,tree1)
tree3.add(2,tree2)
tree4 = DecisionTree(DecisionTree.NODE)
tree4.add(0,tree3)
tree4.add(1,tree2)
tree4.add(2,tree1)

assert(tree1.countTree(0) == 0)
assert(tree2.countTree(0) == 1)
assert(tree3.countTree(0) == 2)
assert(tree4.countTree(0) == 3)
print learned.countTree()
print learned.dt

