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


def vote(trees, example):
    "Takes a list of tuples of trees and their weights. They vote on the dataset"

    average = 0.0;

    for i in xrange(len(trees)):
        average += (trees[i][1])*(trees[i][0].predict(example))

    #print "sum", sum(t[1] for t in trees)
    average = average/sum(t[1] for t in trees)

    if average > 0.5:
        return 1
    else:
        return 0

def scoreWeakTrees(learners,dataset):
    "Score a set of weak learners"

    # retrieve examples
    examples = dataset.examples

    # keep track of the number they train correctly
    numCorrect = 0
    for i in xrange(len(examples)):
        #print examples[i].attrs
        if vote(learners, examples[i]) == examples[i].attrs[dataset.target]:
            numCorrect = numCorrect + 1
            #print "correct!"
            #print numCorrect

    return numCorrect / float(len(examples))

def weightHyp(learner, dataset):
    "Finds the weight of a hypothesis"

    # retrive examples
    examples = dataset.examples[:]

    # find the error
    error = 0.0
    for i in xrange(len(examples)):
        #print learner.predict(examples[i])
        #print examples[i].attrs[dataset.target]
        if learner.predict(examples[i]) != examples[i].attrs[dataset.target]:
            #print "wrong"
            error = error + examples[i].weight
            #print error
        # else:
        #     print i

    if error == 0:
        return sys.maxint

    if error == 1:
        return 0

    else:
        return 0.5 * math.log((1-error)/error)

def weightData(learner, dataset, alpha):
    "reweights the data based on how well the algorithm did on each data point"
    #print alpha
    mydata = dataset
    for e in mydata.examples:
        if learner.predict(e) != e.attrs[mydata.target]:
            e.weight = e.weight * math.exp(alpha)
        else:
            e.weight = e.weight * math.exp((-1.0*alpha))

    # Normalize
    s = float(sum([e.weight for e in mydata.examples]))
    #print [e.weight for e in mydata.examples]
    for e in mydata.examples:
        e.weight = e.weight / s

# wexample1 = Example([1,0,1,0,0])
# wexample1.weight = 0.5
# wexample2 = Example([1,0,0,0,1])
# wexample2.weight = 0.1
# wexample3 = Example([0,0,0,1,0])
# wexample3.weight = 0.0
# wexample4 = Example([1,0,0,1,1])
# wexample4.weight = 0.1
# wexample5 = Example([0,1,1,1,1])
# wexample5.weight = 0.2
# wexample6 = Example([1,1,0,1,0])
# wexample6.weight = 0.1

# wdataset = DataSet([wexample1,wexample2,wexample3,wexample4,wexample5,wexample6])

# wxexample1 = Example([1,0,1,0,1])
# wxexample1.weight = 0.1
# wxexample2 = Example([1,0,0,0,0])
# wxexample2.weight = 0.2
# wxexample3 = Example([0,0,0,1,1])
# wxexample3.weight = 0.3
# wxexample4 = Example([1,0,0,1,0])
# wxexample4.weight = 0.2
# wxexample5 = Example([0,1,1,1,0])
# wxexample5.weight = 0.1
# wxexample6 = Example([1,1,0,1,1])
# wxexample6.weight = 0.1

# whalfdataset = DataSet([wxexample1,wxexample2,wxexample3,wexample4,wexample5,wexample6])

# learned = DecisionTreeLearner()
# learned.train(wdataset)
# weightData(learned,whalfdataset,2)
# print [e.weight for e in whalfdataset.examples]

def boosting(dataset,numrounds,maxdepth):
    learners = []
    mydata = copy.deepcopy(dataset)
    for i in xrange(numrounds):
        learner = DecisionTreeLearner()
        learner.train(mydata,cutoff=maxdepth)
        #learner.dt.display()
        alpha = weightHyp(learner,mydata)
        print "alpha", alpha
        if alpha == sys.maxint:
            return [[learner,1]]
        weightData(learner,mydata,alpha)
        print [e.weight for e in mydata.examples]
        #print [e.weight for e in mydata.examples]
        #print "score", scoreTree(learner,mydata)
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

#def findEndNodes (tree):
#  "returns list of end nodes (nodes with only leaves attached) "



def prune (pLearner, origLearner, validation):
  nodelist = []
  classlist = []
  for a, c in pLearner.dt.branches.iteritems():
    nodelist.append(c.nodetype)
    if c.nodetype == DecisionTree.LEAF:
      classlist.append(c.classification)
  print len(nodelist)
  if every(lambda x: x == DecisionTree.LEAF, nodelist):
    hold = copy.deepcopy(pLearner)
    pLearner.dt.nodeType = DecisionTree.LEAF
    pLeanrer.dt.classification = mode(classlist)
    if scoreTree(pLearner, validation) > scoreTree(origLearner, validation):
      return pLearner.dt
    else:
      return hold.dt
  else:
    for a, c in pLearner.dt.branches.iteritems():
      if c.nodetype == DecisionTree.NODE:
        pLearner.dt = c
        pLearner.dt = prune(pLearner, origLearner, validation)


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
    
    # Copy the dataset so we have two copies of it
    examples = dataset.examples[:]
 
    dataset.examples.extend(examples)
    dataset.max_depth = maxDepth
    if boostRounds != -1:
      dataset.use_boosting = True
      dataset.num_rounds = boostRounds

    # ====================================
    # WRITE CODE YOUR EXPERIMENTS HERE
    # ====================================

    # =========================
    # Ten-fold Cross-Validation
    # =========================
    learner = DecisionTreeLearner()
    learner.train(dataset)
    learner.dt.display()

    # # Divide data into 10 chunks
    # fold = 10

    # dataLength = len(examples)
    # chunkLength = dataLength/fold

    # for each chunk, train on the remaining data and test on the chunk
    runningAverage = 0
    for i in range(fold):
        learner = DecisionTreeLearner()
        training = DataSet(dataset.examples[(i*chunkLength):(i+fold-1)*chunkLength], values=dataset.values)
        validation = DataSet(dataset.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
        learner.train(training)

        # make pruning tree
        pruneLearner = copy.deepcopy(learner)

#        prune(pruneLearner, learner)

        runningAverage += scoreTree(learner, validation)

    # # for each chunk, train on the remaining data and test on the chunk
    # runningAverage = 0
    # for i in range(fold):
    #     learner = DecisionTreeLearner()
    #     training = DataSet(dataset.examples[(i*chunkLength):(i+fold-1)*chunkLength], values=dataset.values)
    #     validation = DataSet(dataset.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
    #     learner.train(training)
    #     runningAverage += scoreTree(learner, validation)

    # # print the average score
    # print "The average cross-validation score is", (runningAverage / fold)

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

        if valSetSize >= len(examples):
            print "Please make sure your validation set size is smaller than your data set"

        else:
            # weight the data so that they all add to one
            for e in examples:
                e.weight = 1.0/len(examples)

            dataset2 = DataSet(examples)
            dataset2.examples.extend(examples)

            # split data into training and validation randomly
            # split = splitData(dataset2,valSetSize)
            # training = split[0]
            # validation = split[1]

            runningAverage = 0
            for i in range(1):
                learner = DecisionTreeLearner()
                training = DataSet(dataset2.examples[(i*chunkLength):(i+fold-1)*chunkLength], values=dataset2.values)
                validation = DataSet(dataset2.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
                # get all the weak learners
                learners = boosting(training, boostRounds, maxDepth)
                runningAverage += scoreWeakTrees(learners, validation)

            # # get all the weak learners
            # learners = boosting(training, boostRounds, maxDepth)

            print "The score for the AdaBoost algorithm with",boostRounds, "rounds and a max depth of",maxDepth, "is",runningAverage/fold





main()


    
