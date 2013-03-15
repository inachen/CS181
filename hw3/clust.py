# clust.py
# -------
# YOUR NAME HERE

import sys
import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import random
import math
from utils import *


DATAFILE = "adults.txt"

#validateInput()

def validateInput():
    if len(sys.argv) != 4:
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

# Given responsibilites, find index of the 1
def findIndex(respon):
    for i in range(len(respon)):
        if respon[i] == 1:
            return i

# multiplies a list all the way through by a number
def listmult(list, num):
    return [(float(x * num)) for x in list]


# adds 2 lists element by element
def listadd(list1, list2):
    if list1 == None:
        return list2
    if list2 == None:
        return list1
    assert(len(list1) == len(list2)), "The two lists must be the same length"
    return [float(i+j) for i,j in zip(list1, list2)]

# performs k means clustering on a set of data
def kmeans(data,numExamples,numClusters):

    # find the dimension of the data
    dimension = len(data[1])

    # find the initial prototypes
    prototypes = []
    index = []

    for i in xrange(numClusters):
        rand = random.randint(0,numExamples-1)
        # make sure there are no repeats
        while rand in index:
            rand = random.randint(0,numExamples-1)
        prototypes.append(data[rand])
        index.append(rand)

    errors = []
    counter = 0
    responsibilities = [None]*numExamples

    # repeat until the error stops decreasing
    while True:

        # assign responsibilities to data
        for i in xrange(numExamples):
            distances = map(lambda n: utils.squareDistance(data[i],n), prototypes)
            responsibilities[i] = makeR(numClusters,distances.index(min(distances)))

        # find new error
        error = 0
        for j in range(numExamples):
            for k in range(numClusters):
                error += responsibilities[j][k]*utils.squareDistance(data[j],prototypes[k])/numExamples

        # print "Error:",counter,":",errors, error

        # quit if error isn't improving
        if counter > 10 and error >= errors[counter-9]:
            return error
            break

        # else, updates the prototypes
        errors.append(error)
        counter += 1

        for l in range(numClusters):
            topsum = [0]*dimension
            bottomsum = 0
            for m in range(numExamples):
                myrespon = responsibilities[m][l]
                topsum = listadd(topsum, listmult(data[m],myrespon))
                bottomsum += responsibilities[m][l]
            if bottomsum == 0:
                print "Cluster number",l,"is obsolete"
            prototypes[l] = listmult(topsum,float(1.0/bottomsum))


#===========#
# Autoclass #
#===========#

def autoclass(data,numExamples,numClusters):
    numAttrs = len(data[1])

    # record all the log liklihoods
    log_likely = []

    # record mean cluster values
    meanClusterValues = [[0]*numAttrs]*numClusters

    # initialize responsibilities
    responsibilities = [None]*numExamples

    # set all time high log likelihood
    high = -sys.maxint

    threshold = 0.000001

    # process data so that all data attributes have values 0 or 1
    cutoffs = reduce(listadd,data[0:numExamples],None)
    cutoffs = listmult(cutoffs,float(1.0/numExamples))

    mydata = []
    for i in range(numExamples):
        attributes = []
        for j in range(numAttrs):
            if data[i][j] >= cutoffs[j]:
                attributes.append(1)
            else:
                attributes.append(0)
        mydata.append(attributes)

    # initialize starting parameters to 1/numClusters and random values in [0,1]
    # thetas is a list (length numClusters) of lists (length numAttrs)
    thetas = []
    for i in range(numClusters):
        thetarow = [float(1.0/numClusters)]
        for j in range(numAttrs):
            thetarow.append(random.random())
        thetas.append(thetarow)

    counter = 0
    while True:
        # Expectation step

        # lists of expectations for all the k's
        E_N = [0]*numClusters
        E_D1 = [[0]*numClusters]*numAttrs

        # iterate over all the data
        for i in range(numExamples):
            pvalues = []
            # calculate the p values for each cluster
            for j in range(numClusters):
                dproduct = 1.0
                for k in range(numAttrs):
                    dproduct *= pow(thetas[j][k+1],mydata[i][k])*pow((1-thetas[j][k+1]),(1-mydata[i][k]))
                pvalue = thetas[j][0]*dproduct
                pvalues.append(pvalue)
                largestpIndex = pvalues.index(max(pvalues))
            # largest p corresponds to most likely being in that cluster
            responsibilities[i] = makeR(numClusters,largestpIndex)
            total = sum(pvalues)
            gammas = map(lambda n: n/float(total), pvalues)
            
            # increment E_N
            E_N = [a + b for a,b in zip(E_N,gammas)]

            # for each attribute, if x_nd = 1, increment E(N_d1(k))
            for l in range(numAttrs):
                if mydata[i][l] == 1:
                    E_D1[l] = [a + b for a,b in zip(E_D1[l],gammas)]

        # for each data point, for each k, calculate all p_k
        #     increment E(Nk)

        # Maximation
        # update the theta_c's
        for i in range(numClusters):
            thetas[i][0] = float(E_N[i])/numExamples
            for j in range(numAttrs):
                thetas[i][j+1] = float(E_D1[j][i])/E_N[i]

        counter += 1


        # Calculate log likelihoods
        sums = [0]*numClusters
        for i in range(numExamples):
            clust = findIndex(responsibilities[i])
            insidesums = 0
            for j in range(numAttrs):
                if thetas[clust][j+1] > 0 and thetas[clust][j+1] < 1:
                    insidesums += (mydata[i][j]*(log(thetas[clust][j+1]))) + ((1-mydata[i][j])*log(1-thetas[clust][j+1]))
            sums[clust] += log(thetas[clust][0]) + insidesums
        likely = sum(sums)

        log_likely.append(likely)
        print "log:",counter, ",",likely

        if likely > high + threshold:
            high = likely
        else:
            break

    # determine the mean cluster values
    for i in range(numClusters):
        mean = [0]*numAttrs
        counter = 0
        for j in range(numExamples):
            if responsibilities[j][i] == 1:
                mean = [a + b for a,b in zip(mean,mydata[j])]
                counter += 1
        meanClusterValues[i] = [a/float(counter) for a in mean]

    for i in range(numClusters):
        print "CLUSTER",i+1,"MEANS:",meanClusterValues[i]

    # plot the data
    plt.clf()
    xs = range(1,len(log_likely)+1)
    ys = log_likely
    p1, = plt.plot(xs, ys, color='b')
    plt.title('Log likelihood versus number of iterations (K=%s)' % numClusters)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Los likelihood')
    plt.axis([0, len(log_likely)+5, -10000, -15000])

    savefig('log_likely%s.jpg' % numClusters) # save the figure to a file
    # plt.show() # show the figure

#===========#
# HAC       #
#===========#
# performs HAC on set of data
def hac(data, numExamples, numClusters, dfunc):

    index = []

    datasample = random.sample(data, numExamples)

    clusterset = []
    for d in datasample:
        clusterset.append([d])
    # print "cluster start"
    # print clusterset

    # find data with smallest distance
    while len(clusterset) > numClusters:
        # stores indices of smallest distance
        curr_i = 0
        curr_j = 1
        for i in range(len(clusterset)):
            for j in range(len(clusterset)):
                if i !=j:
                    # print "i=", i
                    # print "j=", j
                    if dfunc(clusterset[i], clusterset[j], squareDistance) < dfunc(clusterset[curr_i], clusterset[curr_j], squareDistance):
                        curr_i = i
                        curr_j = j
        clust1 = clusterset[curr_i]
        clust2 = clusterset[curr_j]
        clusterset.append(clusterset[curr_i] + clusterset[curr_j])
        # print "cluster added"
        # print clusterset
        # print "pops"
        print clusterset.remove(clust1)
        print clusterset.remove(clust2)
        # print "cluster popped"
        # print clusterset

    return clusterset

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def main():
    # Validate the inputs
    if(validateInput() == False):
        print "Usage: clust numClusters numExamples algo"
        sys.exit(1);

    numClusters = int(sys.argv[1])
    numExamples = int(sys.argv[2])
    # 0 for kmeans, 1 for HAC
    algo = int(sys.argv[3])


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

    # KMEANS
    errors = []
    if algo == 0:
        for i in range(2,11):
            errors.append(kmeans(data,numExamples,i))

    # HAC
    clusterset = []
    if algo == 1:
        clusterset = hac(data, numExamples, numClusters, ccent)
        print "Cluster lengths"
        for l in clusterset:
            print len(l)

    # Autoclass
    if algo == 2:
        for i in range(9):
            autoclass(data,numExamples,i+2)

    #===============#
    # plot the data #
    #===============#

    # KMEANS
    # print errors

    if algo == 0:
        plt.clf()
        xs = range(2,11)
        ys = errors
        p1, = plt.plot(xs, ys, color='b')
        plt.title('Error versus number of clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Error')
        plt.axis([0, 12, 0.8, 1.8])

        savefig('errorkmeans.jpg') # save the figure to a file
        plt.show() # show the figure

    # HAC
    if algo == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colorlist = ['r', 'g', 'b', 'y']
        for c in range(len(clusterset)):
            # xs = []
            # ys = []
            # zs = []
            for l in clusterset[c]:
                # print "l is"
                # print l
                x = l[0]
                y = l[1]
                z = l[2]
                # xs.append(l[0])
                # ys.append(l[1])
                # zs.append(l[2])
                ax.scatter(x, y, z, c=colorlist[c])
        plt.title("HAC on 200 examples, Clusters = 4, Metric = centroid")
        ax.set_xlabel('Age')
        ax.set_ylabel('Education')
        ax.set_zlabel('Income')


        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot(xs, ys, zs, c=color, label='hac-cmin-100')
        # ax.legend()

        savefig('hac-cmin-100.jpg') # save the figure to a file
        plt.show() # show the figure

if __name__ == "__main__":
    validateInput()
    main()
