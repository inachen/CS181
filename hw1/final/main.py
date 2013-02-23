# main.py
# -------
# Kathy lin, Ina (Weilin) Chen

import matplotlib.pyplot as plt
from pylab import *
from dtree import *
import sys

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
        if learner.predict(examples[i]) != examples[i].attrs[dataset.target]:
            error = error + examples[i].weight

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

def boosting(dataset,numrounds,maxdepth):
    "Performs boosting given the max depth of the tree and the number of rounds"
    learners = []
    mydata = copy.deepcopy(dataset)
    for i in xrange(numrounds):
        #mydata = copy.deepcopy(dataset)
        learner = DecisionTreeLearner()
        learner.train(mydata,cutoff=maxdepth)
        #learner.dt.display()
        alpha = weightHyp(learner,mydata)
        #print "alpha", alpha
        if alpha == sys.maxint:
            return [[learner,1]]
        weightData(learner,mydata,alpha)
        learners.append([learner,alpha])
    return learners

def splitData(dataset,size):
    "Takes the data and randomly splits into two smaller ones based on 'size'"
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

def findEndNodes (tree, attrlist, attrlists, curAttr=None):
  "returns list of end nodes (nodes with only leaves attached) "
  if tree.nodetype == DecisionTree.NODE:

    if curAttr != None:
      attrlist.append(curAttr)

    if every(lambda x: x.nodetype == DecisionTree.LEAF, tree.branches.values()):
      attrlists.append(attrlist)
    else:
      for a in tree.branches.keys():
        cplist = copy.deepcopy(attrlist)
        findEndNodes(tree.branches[a], cplist, attrlists, a)
  return attrlists

def prune (learner, dataset):
  "prunes a learner by looking at all the stumps until there is no more error decrease"
  stumps = findEndNodes (learner.dt, [], [])
  for s in stumps:
    pLearner = copy.deepcopy(learner)
    pLearner.dt.collapse(s)
    if scoreTree(pLearner, dataset) >= scoreTree(learner, dataset):
      return prune(pLearner, dataset)
  return learner

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
    # Fold Cross-Validation
    # =========================

    # Divide data into however many chunks
    fold = 10

    dataLength = len(examples)
    chunkLength = dataLength/fold

    if pruneFlag == False:

      # for each chunk, train on the remaining data and test on the chunk [ for unpruned Cross Validation]
      runningAverage = 0
      for i in range(fold):
          learner = DecisionTreeLearner()
          training = DataSet(dataset.examples[(i*chunkLength):(i+fold-1)*chunkLength], values=dataset.values)

          
          testing = DataSet(dataset.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
          learner.train(training)

          runningAverage += scoreTree(learner, testing)
      #print the average score
      print "The average cross-validation score is", (runningAverage / fold)


      # =========================
      # Pruning
      # =========================

    else: 
      # keep track of scores for unpruned learner
      runningAverageNP = 0

      # keep track of scores for pruned learner
      runningAverageP = 0
      for i in range(fold):
        # divide data into training, validation, and testing sets
        learner = DecisionTreeLearner()
        training = DataSet(dataset.examples[(i*chunkLength):((i + fold - 1) * chunkLength - valSetSize)], values=dataset.values)
        validation = DataSet(dataset.examples[((i + fold - 1) * chunkLength - valSetSize): (i+fold-1)*chunkLength])
        testing = DataSet(dataset.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
        learner.train(training)

        # print 'learner'
        # learner.dt.display()
        
        plearner = prune(learner, validation)
        # print 'pruned learner'
        # plearner.dt.display()

        runningAverageNP += scoreTree(learner, testing)
        runningAverageP += scoreTree(plearner, testing)

      print "The average cross-validated score without pruning is:", (runningAverageNP/fold)
      print "The average cross-validated score with pruning is:", (runningAverageP/fold)

      # =========================
      # Validation Set Pruning Graphs
      # =========================

      # 2bi

    # file1 = open("noisy.csv")

    # dataNoisy = parse_csv(file1.read(), " ")
    # datasetN = DataSet(dataNoisy)

    # examplesN = datasetN.examples[:]

    # datasetN.examples.extend(examplesN)
    # datasetN.max_depth = maxDepth

    # file2 = open("data.csv")

    # dataFull = parse_csv(file2.read(), " ")
    # datasetF = DataSet(dataFull)

    # examplesF = datasetF.examples[:]

    # datasetF.examples.extend(examplesF)
    # datasetF.max_depth = maxDepth

    # Plist = []
    # NPlist = []
    # for valsize in range(1,81):
    #   NPcount = 0
    #   # keep track of scores for pruned learner
    #   Pcount = 0
    #   for i in range(fold):
    #     # divide data into training, validation, and testing sets
    #     learner = DecisionTreeLearner()
    #     training = DataSet(datasetF.examples[(i*chunkLength):((i + fold - 1) * chunkLength - valsize)], values=datasetF.values)
    #     validation = DataSet(datasetF.examples[((i + fold - 1) * chunkLength - valsize): (i+fold-1)*chunkLength])
    #     testing = DataSet(datasetF.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
    #     learner.train(training)
        
    #     plearner = prune(learner, validation)

    #     NPcount += scoreTree(learner, testing)
    #     Pcount += scoreTree(plearner, testing)

    #   NPavg = NPcount/fold
    #   Pavg = Pcount/fold

    #   NPlist.append(NPavg)
    #   Plist.append(Pavg)

    # PlistN = []
    # NPlistN = []
    # for valsize in range(1,81):
    #   NPcount = 0
    #   # keep track of scores for pruned learner
    #   Pcount = 0
    #   for i in range(fold):
    #     # divide data into training, validation, and testing sets
    #     learner = DecisionTreeLearner()
    #     training = DataSet(datasetN.examples[(i*chunkLength):((i + fold - 1) * chunkLength - valsize)], values=datasetN.values)
    #     validation = DataSet(datasetN.examples[((i + fold - 1) * chunkLength - valsize): (i+fold-1)*chunkLength])
    #     testing = DataSet(datasetN.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
    #     learner.train(training)
        
    #     plearner = prune(learner, validation)

    #     NPcount += scoreTree(learner, testing)
    #     Pcount += scoreTree(plearner, testing)

    #   NPavg = NPcount/fold
    #   Pavg = Pcount/fold

    #   NPlistN.append(NPavg)
    #   PlistN.append(Pavg)


    # plt.clf()
    # xs = range(1, 81)
    # ys = Plist
    # ys2 = NPlist
    # # ys3 = PlistN
    # # ys4 = NPlistN
    # p1, = plt.plot(xs, ys, color='b')
    # p2, = plt.plot(xs, ys2, color='r')
    # # p3, = plt.plot(xs, ys3, color='g')
    # # p4, = plt.plot(xs, ys4, color='r')
    # plt.title('Pruning: Cross Validation Performance vs Validation Set Size (Full Dataset) ')
    # plt.xlabel('Validation Set Size')
    # plt.ylabel('Cross Validation Performance')
    # plt.axis([0, 90, 0.6, 1])
    # # plt.legend((p1,), ('pruned',), 'lower right')
    # plt.legend((p1,p2), ('pruned','not pruned'), 'lower right')
    # # plt.legend((p1,p3), ('full data','noisy data'), 'lower right')
    # # plt.legend((p3,p4), ('pruned','not pruned'), 'lower right')
    # savefig('figure.pdf') # save the figure to a file
    # plt.show() # show the figure

    # # ========
    # # AdaBoost
    # # ========
    
    # Divide data into chunks

    dataLength = len(examples)
    chunkLength = dataLength/fold

    if boostRounds > 0 and maxDepth > 0:

        if chunkLength >= len(examples):
            print "Please make sure your validation set size is smaller than your data set"

        else:
            myexamples = copy.deepcopy(examples)

            # weight the data so that they all add to one
            for e in myexamples:
                e.weight = 1.0/len(myexamples)

            dataset2 = DataSet(myexamples)
            dataset2.examples.extend(myexamples)

            # keep track of the score for each validation 
            runningAverage = 0
            for i in range(fold):
                learner = DecisionTreeLearner()
                training = DataSet(dataset2.examples[(i*chunkLength):(i+fold-1)*chunkLength], values=dataset2.values)
                validation = DataSet(dataset2.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
                # get all the weak learners
                learners = boosting(training, boostRounds, maxDepth)
                runningAverage += scoreWeakTrees(learners, validation)

            print "The score for the AdaBoost algorithm with",boostRounds, "rounds and a max depth of",maxDepth, "is",runningAverage/fold

    # =========================================================================
    # Graphing Boosting for a range of round-sizes for noisy and non-noisy data
    # Uncomment to run
    # =========================================================================

    # # run for 1 through 30 rounds with maxDepth = 1
    # # do for both noisy and non-noisy data
    # # graph results

    # rounds = 30

    # results = []
    # # weight the data so that they all add to one
    # myexamples = copy.deepcopy(examples)
    # # weight the data so that they all add to one
    # for e in myexamples:
    #     e.weight = 1.0/len(myexamples)

    # dataset3 = DataSet(myexamples)
    # dataset3.examples.extend(myexamples)

    # # keep track of the score for each validation 
    # for roundnum in range(1,rounds + 1):
    #     runningAverage = 0
    #     for i in range(fold):
    #         learner = DecisionTreeLearner()
    #         training = DataSet(dataset3.examples[(i*chunkLength):(i+fold-1)*chunkLength], values=dataset3.values)
    #         validation = DataSet(dataset3.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
    #         # get all the weak learners
    #         learners = boosting(training, roundnum, 1)
    #         runningAverage += scoreWeakTrees(learners, validation)

    #     results.append(runningAverage/fold)

    # # now repeat for noisy data
    # g = open("noisy.csv")

    # datanoise = parse_csv(g.read(), " ")
    # datasetnoise = DataSet(datanoise)

    # resultsnoise = []
    # # weight the data so that they all add to one
    # examplesnoise = copy.deepcopy(datanoise)
    # # weight the data so that they all add to one
    # for e in examplesnoise:
    #     e.weight = 1.0/len(examplesnoise)

    # dataset4 = DataSet(examplesnoise)
    # dataset4.examples.extend(examplesnoise)

    # # keep track of the score for each validation 
    # for roundnum in range(1,rounds + 1):
    #     runningAverage = 0
    #     for i in range(fold):
    #         learner = DecisionTreeLearner()
    #         training = DataSet(dataset4.examples[(i*chunkLength):(i+fold-1)*chunkLength], values=dataset4.values)
    #         validation = DataSet(dataset4.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
    #         # get all the weak learners
    #         learners = boosting(training, roundnum, 1)
    #         runningAverage += scoreWeakTrees(learners, validation)

    #     resultsnoise.append(runningAverage/fold)

    # plt.clf()
    # xs = range(1,rounds + 1)
    # ys = results
    # ys2 = resultsnoise
    # p1, = plt.plot(xs, ys, color='b')
    # p2, = plt.plot(xs, ys2, color='r')
    # plt.title('Cross-validated test performance vs. number of boosting rounds')
    # plt.xlabel('Number of Boosting Rounds')
    # plt.ylabel('Test Performance')
    # plt.axis([0, rounds+2, 0.77, 1])

    # plt.legend([p1,p2], ['non-noisy','noisy'], 'lower right')
    # savefig('figure.jpg') # save the figure to a file
    # plt.show() # show the figure





    # =======================================================================
    # Graphing Boosting for a range of round-sizes for test and training data
    # Uncomment to run
    # =======================================================================

    # # run for rounds 1 through 15 on non-noisy data 
    # # get both training and test data
    # rounds = 15

    # resultstest = []
    # resultstraining = []

    # # weight the data so that they all add to one
    # myexamples = copy.deepcopy(examples)
    # # weight the data so that they all add to one
    # for e in myexamples:
    #     e.weight = 1.0/len(myexamples)

    # dataset1 = DataSet(myexamples)
    # dataset1.examples.extend(myexamples)

    # # keep track of the score for each validation 
    # for roundnum in range(1,rounds + 1):
    #     testaverage = 0
    #     trainingaverage = 0
    #     for i in range(fold):
    #         learner = DecisionTreeLearner()
    #         training = DataSet(dataset1.examples[(i*chunkLength):(i+fold-1)*chunkLength], values=dataset1.values)
    #         validation = DataSet(dataset1.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
    #         # get all the weak learners
    #         learners = boosting(training, roundnum, 1)
    #         testaverage += scoreWeakTrees(learners, validation)
    #         trainingaverage += scoreWeakTrees(learners, training)

    #     resultstest.append(testaverage/fold)
    #     resultstraining.append(trainingaverage/fold)

    # # plot 
    # plt.clf()
    # xs = range(1,rounds + 1)
    # ys = resultstest
    # ys2 = resultstraining
    # p1, = plt.plot(xs, ys, color='b')
    # p2, = plt.plot(xs, ys2, color='r')
    # plt.title('Cross-validated performance vs. number of boosting rounds')
    # plt.xlabel('Number of Boosting Rounds')
    # plt.ylabel('Performance')
    # plt.axis([0, rounds+2, 0.77, 1])

    # plt.legend([p1,p2], ['test performance','training performance'], 'lower right')
    # savefig('figure2.jpg') # save the figure to a file
    # plt.show() # show the figure



main()


    
