# main.py
# -------
# YOUR NAME HERE

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

# Give a score for a tree on a dataset between 0 and 1
def scoreTree(learner, dataset):

    # retrive examples
    examples = dataset.examples[:]

    # keep track of the number it trains correctly
    numCorrect = 0
    for i in xrange(len(examples)):
        if learner.predict(examples[i]) == examples[i].attrs[dataset.target]:
            numCorrect = numCorrect + 1

    return numCorrect / float(len(examples))


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

main()


    
