# lists = []
# list2 = None
# list1 = [1,3]
# if list2 == None:
# 	print 'yes'
# lists.append(list2)
# print lists


#   def traverse (tree, examples, attrs, attrlist):
    
#     if tree.attr == None:
#       return
#     else:
#       attrlist.append(tree.attr)
#       attrs = removall(tree.attr, attrs)
#       for a, c in tree.branches.iteritems():
#         if c.nodetype == DecisionTree.NODE:
#           traverse(tree.branches[a], examples, attrs, attrlist)
#       #for a in attrlist



#   nodelist = []
#   classlist = []
#   for a, c in pLearner.dt.branches.iteritems():
#     nodelist.append(c.nodetype)
#     if c.nodetype == DecisionTree.LEAF:
#       classlist.append(c.classification)
#   print len(nodelist)
#   if every(lambda x: x == DecisionTree.LEAF, nodelist):
#     hold = copy.deepcopy(pLearner)
#     pLearner.dt.nodeType = DecisionTree.LEAF
#     pLeanrer.dt.classification = mode(classlist)
#     if scoreTree(pLearner, validation) > scoreTree(origLearner, validation):
#       return pLearner.dt
#     else:
#       return hold.dt
#   else:
#     for a, c in pLearner.dt.branches.iteritems():
#       if c.nodetype == DecisionTree.NODE:
#         pLearner.dt = c
#         pLearner.dt = prune(pLearner, origLearner, validation)

runningAverage = 0
    for i in range(fold):
    	# divide data into training, validation, and testing sets
        learner = DecisionTreeLearner()
        training = DataSet(dataset.examples[(i*chunkLength):((i + fold - 1) * chunkLength - valSetSize)], values=dataset.values)
        validation = DataSet(dataset.examples[((i + fold - 1) * chunkLength - valSetSize + 1): (i+fold-1)*chunkLength], values=dataset.values)
        testing = DataSet(dataset.examples[(i+fold-1)*chunkLength:(i+fold)*chunkLength])
        learner.train(training)
        

        runningAverage += scoreTree(learner, testing)
