# main.py
# -------
# Kathy lin, Ina Chen

from dtree import *
import sys

#import matplotlib.pyplot as plt
#from pylab import *

class Globals:
    noisyFlag = False
    pruneFlag = False
    valSetSize = 0
    dataset = None


##Classify
#---------

def classify(decisionTree, example):
    return decisionTree.predict(example)

##Learn
#-------
def learn(dataset):
    learner = DecisionTreeLearner()
    learner.train( dataset)
    return learner.dt

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
    parseArgs([ 'main.py', '-n', '-p', 5 ]) = { '-n':True, '-p':5 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
    args_map = parseArgs(args)
    valSetSize = 0
    noisyFlag = False
    pruneFlag = False
    boostRounds = -1
    maxDepth = -1
    if '-n' in args_map:
      noisyFlag = True
    if '-p' in args_map:
      pruneFlag = True
      valSetSize = int(args_map['-p'])
    if '-d' in args_map:
      maxDepth = int(args_map['-d'])
    if '-b' in args_map:
      boostRounds = int(args_map['-b'])
    return [noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds]

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
# Testing for scoreTree
# =========================
# learned = DecisionTreeLearner()
# learned.train(dataset1)
# assert(scoreTree(learned, dataset1) == 1)
# assert(scoreTree(learned, xdataset1) == 0)
# assert(scoreTree(learned, halfdataset1) == 0.5)


def vote(trees, example):
    "Takes a list of tuples of trees and their weights. They vote on the dataset"

    average = 0.0;

    for i in xrange(len(trees)):
        average += (trees[i][1])*(trees[i][0].predict(example))

    average = average/sum(t[1] for t in trees)

    if average > 0.5:
        return 1
    else:
        return 0

# =========================
# Testing for vote
# =========================
# learned = DecisionTreeLearner()
# learned.train(dataset1)

# xlearned = DecisionTreeLearner()
# xlearned.train(xdataset1)

# halflearned = DecisionTreeLearner()
# halflearned.train(halfdataset1)

# assert(vote([[learned,0],[xlearned,0.5]], example1) == 1)
# assert(vote([[learned,0],[xlearned,6]], example1) == 1)
# assert(vote([[learned,0.5],[xlearned,0.5]], example1) == 0)
# assert(vote([[learned,0.49],[xlearned,0.5]], example1) == 1)
# assert(vote([[learned,0.5],[xlearned,0.5],[halflearned,0.5]], example1) == 1)
# assert(vote([[learned,0.5],[xlearned,0.5],[halflearned,0.5]], example5) == 1)


def scoreWeakTrees(learners,dataset):
    "Score a set of weak learners"

    # retrieve examples
    examples = dataset.examples

    # keep track of the number they train correctly
    numCorrect = 0
    for i in xrange(len(examples)):
        if vote(learners, examples[i]) == examples[i].attrs[dataset.target]:
            numCorrect = numCorrect + 1

    return numCorrect / float(len(examples))



def weightHyp(learner, dataset):
    "Finds the weight of a hypothesis"

    # retrive examples
    examples = dataset.examples[:]

    # find the error
    error = 0.0
    for i in xrange(len(examples)):
        if learner.predict(examples[i]) != examples[i].attrs[dataset.target]:
            error = error + examples[i].weight

    if error == 0:
        return sys.maxint

    if error == 1:
        return 0

    else:
        return 0.5 * math.log((1-error)/error)

# =========================
# Testing for weightHyp
# =========================

# whalfdataset = DataSet([wxexample1,wxexample2,wxexample3,wexample4,wexample5,wexample6])

# learned = DecisionTreeLearner()
# learned.train(wdataset)

# assert(weightHyp(learned, wdataset) == sys.maxint)
# assert(weightHyp(learned, wxdataset) == 0)
# assert(abs(weightHyp(learned, whalfdataset) - 0.5 * math.log((0.4)/0.6)) < 0.0001)

def weightData(learner, dataset, alpha):
    "reweights the data based on how well the algorithm did on each data point"
    mydata = dataset
    for e in mydata.examples:
        if learner.predict(e) != e.attrs[mydata.target]:
            e.weight = e.weight * math.exp(alpha)
        else:
            e.weight = e.weight * math.exp((-1.0*alpha))

    # Normalize
    s = sum([e.weight for e in mydata.examples])
    for e in mydata.examples:
        e.weight = e.weight / s


# =========================
# Testing for weightData
# =========================

# whalfdataset = DataSet([wxexample1,wxexample2,wxexample3,wexample4,wexample5,wexample6])

# learned = DecisionTreeLearner()
# learned.train(wdataset)

# weightData(learned, whalfdataset, 1.5)
# assert(abs(wxexample1.weight - 0.1*math.exp(1.5)/2.778265506) < 0.00001)
# assert(abs(wxexample2.weight - 0.2*math.exp(1.5)/2.778265506) < 0.00001)
# assert(abs(wxexample3.weight - 0.3*math.exp(1.5)/2.778265506) < 0.00001)
# assert(abs(wexample4.weight - 0.1*math.exp(-1.5)/2.778265506) < 0.00001)
# assert(abs(wexample5.weight - 0.2*math.exp(-1.5)/2.778265506) < 0.00001)
# assert(abs(wexample6.weight - 0.1*math.exp(-1.5)/2.778265506) < 0.00001)

def boosting(dataset,numrounds,maxdepth):
    learners = []
    mydata = dataset
    for i in xrange(numrounds):
        learner = DecisionTreeLearner()
        learner.train(dataset,cutoff=maxdepth)
        alpha = weightHyp(learner,dataset)
        weightData(learner,mydata,alpha)
        learners.append([learner,alpha])
    return learners

def splitData(dataset,size):
    training = dataset.examples
    if size >= len(training):
        return [None,dataset]
    if size <= 0:
        return [dataset,None]
    validation = []
    for i in xrange(size):
        num = random.randint(0,len(training)-1)
        validation.append(training[num])
        del training[num]
    return [DataSet(training,values=dataset.values),DataSet(validation,values=dataset.values)]

# =========================
# Testing for splitData
# =========================

# dataset2 = copy.deepcopy(dataset1)
# dataset3 = copy.deepcopy(dataset1)
# dataset4 = copy.deepcopy(dataset1)

# splits1 = splitData(dataset2,1)
# t1 = splits1[0].examples
# v1 = splits1[1].examples
# splits2 = splitData(dataset3,3)
# t2 = splits2[0].examples
# v2 = splits2[1].examples
# splits3 = splitData(dataset4,5)
# t3 = splits3[0].examples
# v3 = splits3[1].examples

# assert(len(t1) == 5)
# assert(len(v1) == 1)
# assert(len(t2) == 3)
# assert(len(v2) == 3)
# assert(len(t3) == 1)
# assert(len(v3) == 5)
# print [e.attrs for e in t2], [f.attrs for f in v2]

def main():
    arguments = validateInput(sys.argv)
    noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds = arguments
    print noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds

    # Read in the data file
    
    if noisyFlag:
        f = open("noisy.csv")
    else:
        f = open("data.csv")

    data = parse_csv(f.read(), " ")
    dataset = DataSet(data)

    a = splitData(dataset,10)
    #print len(a[1].examples)
    
    # Copy the dataset so we have two copies of it
    examples = dataset.examples[:]
 
    dataset.examples.extend(examples)
    dataset.max_depth = maxDepth
    if boostRounds != -1:
      dataset.use_boosting = True
      dataset.num_rounds = boostRounds

    # ====================================
    # WRITE CODE FOR YOUR EXPERIMENTS HERE
    # ====================================

    # =========================
    # Ten-fold Cross-Validation
    # =========================

    # Divide data into 10 chunks
    fold = 10

    dataLength = len(examples)
    chunkLength = dataLength/fold

    # for each chunk, train on the remaining data and test on the chunk
    runningAverage = 0
    for i in range(fold):
        learner = DecisionTreeLearner()
        training = DataSet(dataset.examples[(i*chunkLength):(i+fold-1)*chunkLength], values=dataset.values)
        validation = DataSet(dataset.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
        learner.train(training)
        runningAverage += scoreTree(learner, validation)

    # print the average score
    print "The average cross-validation score is", (runningAverage / fold)

    # =========================
    # Validation Set Pruning
    # =========================

    if pruneFlag == True:
        plt.clf()
        xs = range(5)
        ys = [3, 5, 1, 10, 8]
        p1 = plt.plot(xs, ys, color='b')
        plt.title('sample graph')
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.axis([0, 4, 0, 12])
        #prune()

    # ========
    # AdaBoost
    # ========

    if boostRounds > 0 and maxDepth > 0:
        
        # split data into training and validation randomly
        split = splitData(dataset,(len(dataset.examples))/10)
        training = split[0]
        validation = split[1]

        # get all the weak learners
        learners = boosting(training, boostRounds, maxDepth)





main()


    
