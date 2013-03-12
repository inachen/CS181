# clust.py
# -------
# YOUR NAME HERE

import sys
import random
import math

DATAFILE = "adults.txt"

#validateInput()

def validateInput():
    if len(sys.argv) != 3:
        return False
    if sys.argv[1] <= 0:
        return False
    if sys.argv[2] <= 0:
        return False
    return True


#-----------


def parseInput(datafile):
    """
    params datafile: a file object, as obtained from function `open`
    returns: a list of lists

    example (typical use):
    fin = open('myfile.txt')
    data = parseInput(fin)
    fin.close()
    """
    data = []
    for line in datafile:
        instance = line.split(",")
        instance = instance[:-1]
        data.append(map(lambda x:float(x),instance))
    return data


def printOutput(data, numExamples):
    for instance in data[:numExamples]:
        print ','.join([str(x) for x in instance])

# Makes a responsibility matrix
def makeR(length, position):
    response = []
    for i in xrange(length):
        if i == position:
            response.append(1)
        else:
            response.append(0)
    return response

# finds the distance between 2 points
def distancesqr(point1, point2):
    assert(len(point1) == len(point2)), "The two points must be in the same dimension"
    sums = 0
    for i in range(len(point1)):
        sums += math.pow(float(point1[i] - point2[i]),2)
    return sums

# multiplies a list all the way through by a number
def listmult(list, num):
    return [float(x * num) for x in list]

# adds 2 lists element by element
def listadd(list1, list2):
    assert(len(list1) == len(list2)), "The two lists must be the same length"
    return [float(i+j) for i,j in zip(list1, list2)]



# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    # Validate the inputs
    if(validateInput() == False):
        print "Usage: clust numClusters numExamples"
        sys.exit(1);

    numClusters = int(sys.argv[1])
    numExamples = int(sys.argv[2])

    #Initialize the random seed
    
    random.seed()

    #Initialize the data

    
    dataset = file(DATAFILE, "r")
    if dataset == None:
        print "Unable to open data file"


    data = parseInput(dataset)
    
    
    dataset.close()
    printOutput(data,numExamples)

    # ==================== #
    # WRITE YOUR CODE HERE #
    # ==================== #

    # find the dimension of the data
    dimension = len(data[1])

    # find the initial prototypes
    prototypes = []
    index = []

    for i in xrange(numClusters):
        rand = random.randint(0,numExamples-1)
        prototypes.append(data[rand])
        index.append(rand)
    print "index", index

    print len(prototypes)

    errors = []
    counter = 0
    responsibilities = [None]*numExamples

    # repeat until the error stops decreasing
    while True:

        # assign responsibilities to data
        for i in xrange(numExamples):
            distances = map(lambda n: distancesqr(data[i],n), prototypes)
            responsibilities[i] = makeR(numClusters,distances.index(min(distances)))
        print responsibilities

        print "RESPONSIBILITIES DONE\n"

        # find new error
        error = 0
        for j in range(numExamples):
            for k in range(numClusters):
                error += responsibilities[j][k]*distancesqr(data[j],prototypes[k])

        print "Error:",counter,":",error

        # quit if error isn't improving
        if counter > 10 and error >= errors[counter-9]:
            break

        # else, updates the prototypes
        errors.append(error)
        counter += 1

        print "STARTING UPDATING"
        for l in range(numClusters):
            topsum = [0]*dimension
            bottomsum = 0
            for m in range(numExamples):
                myrespon = responsibilities[m][l]
                topsum = listadd(topsum, listmult(data[m],myrespon)) 
                bottomsum += responsibilities[m][l]
                if bottomsum == 0:
                    print "Cluster number",l,"is obsolete"
            prototypes[l] = listmult(topsum,float(1/bottomsum))
            print "proto", prototypes

        # print "new prototypes", prototypes


if __name__ == "__main__":
    validateInput()
    main()
