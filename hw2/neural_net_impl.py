from neural_net import NeuralNetwork, NetworkFramework
from neural_net import Node, Target, Input
import random


# <--- Problem 3, Question 1 --->

def FeedForward(network, input):
  """
  Arguments:
  ---------
  network : a NeuralNetwork instance
  input   : an Input instance

  Returns:
  --------
  Nothing

  Description:
  -----------
  This function propagates the inputs through the network. That is,
  it modifies the *raw_value* and *transformed_value* attributes of the
  nodes in the network, starting from the input nodes.

  Notes:
  -----
  The *input* arguments is an instance of Input, and contains just one
  attribute, *values*, which is a list of pixel values. The list is the
  same length as the number of input nodes in the network.

  i.e: len(input.values) == len(network.inputs)

  This is a distributed input encoding (see lecture notes 7 for more
  informations on encoding)

  In particular, you should initialize the input nodes using these input
  values:

  network.inputs[i].raw_value = input[i]
  """
  network.CheckComplete()
  numInputs = len(input.values)
  #print input.values
  # 1) Assign input values to input nodes
  for i in range(numInputs):
    network.inputs[i].raw_value = input.values[i]
    network.inputs[i].transformed_value = input.values[i]
  # 2) Propagates to hidden layer
  for i in range(len(network.hidden_nodes)):
    raw = NeuralNetwork.ComputeRawValue(network.hidden_nodes[i])
    network.hidden_nodes[i].raw_value = raw
    network.hidden_nodes[i].transformed_value = NeuralNetwork.Sigmoid(raw)
  # 3) Propagates to the output layer
  for i in range(len(network.outputs)):
    raw = NeuralNetwork.ComputeRawValue(network.outputs[i])
    network.outputs[i].raw_value = raw
    network.outputs[i].transformed_value = NeuralNetwork.Sigmoid(raw)

  #  network.outputs[3].raw_value

#< --- Problem 3, Question 2

def Backprop(network, input, target, learning_rate):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  input         : an Input instance
  target        : a target instance
  learning_rate : the learning rate (a float)

  Returns:
  -------
  Nothing

  Description:
  -----------
  The function first propagates the inputs through the network
  using the Feedforward function, then backtracks and update the
  weights.

  Notes:
  ------
  The remarks made for *FeedForward* hold here too.

  The *target* argument is an instance of the class *Target* and
  has one attribute, *values*, which has the same length as the
  number of output nodes in the network.

  i.e: len(target.values) == len(network.outputs)

  In the distributed output encoding scenario, the target.values
  list has 10 elements.

  When computing the error of the output node, you should consider
  that for each output node, the target (that is, the true output)
  is target[i], and the predicted output is network.outputs[i].transformed_value.
  In particular, the error should be a function of:

  target[i] - network.outputs[i].transformed_value
  
  """
  network.CheckComplete()

  # 1) We first propagate the input through the network
  FeedForward(network,input)

  delta_out = []

  # 2) Then we compute the errors and update the weigths starting with the last layer
  for j in range(len(network.outputs)):
    myoutput = network.outputs[j]
    y = target[j]
    s = myoutput.transformed_value
    e = y - s
    delta = (e * s * (1 - s))
    delta_out.append(delta)
    for m in range(len(myoutput.inputs)):
      myoutput.weights[m].value = myoutput.weights[m].value + (learning_rate*myoutput.inputs[m].transformed_value*delta)
  
  # 3) We now propagate the errors to the hidden layer, and update the weights there too
  for j in range(len(network.hidden_nodes)):
    mynode = network.hidden_nodes[j]
    e = sum(map(lambda n: mynode.forward_weights[n].value*delta_out[n], range(len(mynode.forward_weights))))
    s = mynode.transformed_value
    delta = e * s * (1 - s)
    for m in range(len(mynode.inputs)):
      mynode.weights[m].value = mynode.weights[m].value + (learning_rate*mynode.inputs[m].transformed_value*delta)
  

# <--- Problem 3, Question 3 --->

def Train(network, inputs, targets, learning_rate, epochs):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  inputs        : a list of Input instances
  targets       : a list of Target instances
  learning_rate : a learning_rate (a float)
  epochs        : a number of epochs (an integer)

  Returns:
  -------
  Nothing

  Description:
  -----------
  This function should train the network for a given number of epochs. That is,
  run the *Backprop* over the training set *epochs*-times
  """
  network.CheckComplete()

  for i in range(epochs):
    for j in range(len(inputs)):
      Backprop(network, inputs[j], targets[j], learning_rate)
      if j == 0:
        first = [n.value for n in network.weights]
      if j == 1:
        second = [n.value for n in network.weights]

  


# <--- Problem 3, Question 4 --->

class EncodedNetworkFramework(NetworkFramework):
  def __init__(self):
    """
    Initializatio.
    YOU DO NOT NEED TO MODIFY THIS __init__ method
    """
    super(EncodedNetworkFramework, self).__init__() # < Don't remove this line >
    
  # <--- Fill in the methods below --->

  def EncodeLabel(self, label):
    """
    Arguments:
    ---------
    label: a number between 0 and 9 (ARE THESE INTS??!!)

    Returns:
    ---------
    a list of length 10 representing the distributed
    encoding of the output.

    Description:
    -----------
    Computes the distributed encoding of a given label.

    Example:
    -------
    0 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    3 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Notes:
    ----
    Make sure that the elements of the encoding are floats.
    
    """
    # Replace line below by content of function
    assert(isinstance(label,int)), "label must be an integer"
    assert(label > -1 and label < 10), "label must be between 0 and 9"
    array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    array[label] = float(1)
    return array

  def GetNetworkLabel(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    the 'best matching' label corresponding to the current output encoding

    Description:
    -----------
    The function looks for the transformed_value of each output, then decides 
    which label to attribute to this list of outputs. The idea is to 'line up'
    the outputs, and consider that the label is the index of the output with the
    highest *transformed_value* attribute

    Example:
    -------

    # Imagine that we have:
    map(lambda node: node.transformed_value, self.network.outputs) => [0.2, 0.1, 0.01, 0.7, 0.23, 0.31, 0, 0, 0, 0.1, 0]

    # Then the returned value (i.e, the label) should be the index of the item 0.7,
    # which is 3
    
    """
    # Replace line below by content of function
    lst = map(lambda node: node.transformed_value, self.network.outputs)
    label = lst.index(max(lst))
    return label

  def Convert(self, image):
    """
    Arguments:
    ---------
    image: an Image instance

    Returns:
    -------
    an instance of Input

    Description:
    -----------
    The *image* arguments has 2 attributes: *label* which indicates
    the digit represented by the image, and *pixels* a matrix 14 x 14
    represented by a list (first list is the first row, second list the
    second row, ... ), containing numbers whose values are comprised
    between 0 and 256.0. The function transforms this into a unique list
    of 14 x 14 items, with normalized values (that is, the maximum possible
    value should be 1).
    
    """
    # Replace line below by content of function
    # function goes through image pixels first through rows then through columns
    dim = 14
    assert (dim == len(image.pixels)), "dimension mismatch"

    inputs = Input()

    for i in range(dim):
      for j in range(dim):
        newpixel = float(image.pixels[i][j]/256.0)
        inputs.values.append(newpixel)

    return inputs

  def InitializeWeights(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes the weights with random values between [-0.01, 0.01].

    Hint:
    -----
    Consider the *random* module. You may use the the *weights* attribute
    of self.network.
    
    """
    # replace line below by content of function
    for weight in self.network.weights:
      weight.value = random.uniform(-0.01, 0.01)

# Problem 3, Q5: Since our output values are between 0 and 1, normalizing the input values
# prevents us from having to normalize in later functions

#<--- Problem 3, Question 6 --->

class SimpleNetwork(EncodedNetworkFramework):
  def __init__(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a simple network, with 196 input nodes,
    10 output nodes, and NO hidden nodes. Each input node
    should be connected to every output node.
    """
    super(SimpleNetwork, self).__init__() # < Don't remove this line >
    
    # 1) Adds an input node for each pixel. 
    # defines the dimensions of the image
    DIM = 14
    DIGITS = 10
    newinputs = []
    for i in range(DIM*DIM):
      newin = Node()
      newinputs.append(newin)
      self.network.AddNode(newin,self.network.INPUT)

    # 2) Add an output node for each possible digit label.
    for j in range(DIGITS):
      newout = Node()
      for k in range(DIM*DIM):
        newout.AddInput(newinputs[k],None,self.network)
      self.network.AddNode(newout,self.network.OUTPUT)




#<---- Problem 3, Question 7 --->

class HiddenNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=15):
    """
    Arguments:
    ---------
    number_of_hidden_nodes : the number of hidden nodes to create (an integer)

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a network with a hidden layer. The network
    should have 196 input nodes, the specified number of
    hidden nodes, and 10 output nodes. The network should be,
    again, fully connected. That is, each input node is connected
    to every hidden node, and each hidden_node is connected to
    every output node.
    """
    super(HiddenNetwork, self).__init__() # < Don't remove this line >

    # 1) Adds an input node for each pixel
    # from above
    DIM = 14
    DIGITS = 10
    newinputs = []
    newhidden = []
    for i in range(DIM*DIM):
      newin = Node()
      newinputs.append(newin)
      self.network.AddNode(newin,self.network.INPUT)
    # 2) Adds the hidden layer
    for i in range(number_of_hidden_nodes):
      newhid = Node()
      for k in range(DIM*DIM):
        newhid.AddInput(newinputs[k],None,self.network)
      newhidden.append(newhid)
      self.network.AddNode(newhid,self.network.HIDDEN)
    # 3) Adds an output node for each possible digit label.
    for j in range(DIGITS):
      newout = Node()
      for k in range(number_of_hidden_nodes):
        newout.AddInput(newhidden[k],None,self.network)
      self.network.AddNode(newout,self.network.OUTPUT)
    

#<--- Problem 3, Question 8 ---> 

class CustomNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=10):
    """
    Arguments:
    ---------
    number_of_hidden_nodes : the number of hidden nodes to create (an integer) for each quadrant

    Returns:
    --------
    Nothing

    Description:
    -----------
    Has a hidden layer that is not fully connected. The image is split into 4 quadrants, which are 
    trained on separately before combined
    0   1   2   3   4   5   6  | 7   8   9   10  11  12  13
    14  15  16  17  18  19  20 | 21  22  23  24  25  26  27
    28  29  30  31  32  33  34 | 35  36  37  38  39  40  41
    42  43  44  45  46  47  48 | 49  50  51  52  53  54  55
    56  57  58  59  60  61  62 | 63  64  65  66  67  68  69
    70  71  72  73  74  75  76 | 77  78  79  80  81  82  83 
    84  85  86  87  88  89  90 | 91  92  93  94  95  96  97
    ---------------------------|---------------------------  
    98  99  100 ...            |
    ...                        |
    182 183 184 185 186 187 188| 189 190 191 192 193 194 195

    quadrant 1: i < 98, i%14 < 7
    quadrant 2: i < 98, i%14 > 6
    quadrant 3: i > 97, i%14 < 7
    quadrant 4: i > 97, i%14 > 6


    """
    super(CustomNetwork, self).__init__() # <Don't remove this line>
    
    # 1) Adds an input node for each pixel
    # from above
    DIM = 14
    DIGITS = 10
    q1inputs = []
    q2inputs = []
    q3inputs = []
    q4inputs = []
    q1hidden = []
    q2hidden = []
    q3hidden = []
    q4hidden = []
    for i in range(DIM*DIM):
      newin = Node()
      if i < 98 and i%14 < 7:
        q1inputs.append(newin)
      elif i < 98 and i%14 > 6:
        q2inputs.append(newin)
      elif i > 97 and i%14 < 7:
        q3inputs.append(newin)
      else:
        q4inputs.append(newin)
      self.network.AddNode(newin,self.network.INPUT)

    # 2) Adds the hidden layer
    for i in range(number_of_hidden_nodes):
      newq1 = Node()
      newq2 = Node()
      newq3 = Node()
      newq4 = Node()
      for k in range(DIM*DIM/4):
        newq1.AddInput(q1inputs[k],None,self.network)
        newq2.AddInput(q2inputs[k],None,self.network)
        newq3.AddInput(q3inputs[k],None,self.network)
        newq4.AddInput(q4inputs[k],None,self.network)
      q1hidden.append(newq1)
      q2hidden.append(newq2)
      q3hidden.append(newq3)
      q4hidden.append(newq4)
      self.network.AddNode(newq1,self.network.HIDDEN)
      self.network.AddNode(newq2,self.network.HIDDEN)
      self.network.AddNode(newq3,self.network.HIDDEN)
      self.network.AddNode(newq4,self.network.HIDDEN)
    # 3) Adds an output node for each possible digit label.
    for j in range(DIGITS):
      newout = Node()
      for k in range(number_of_hidden_nodes):
        newout.AddInput(q1hidden[k],None,self.network)
        newout.AddInput(q2hidden[k],None,self.network)
        newout.AddInput(q3hidden[k],None,self.network)
        newout.AddInput(q4hidden[k],None,self.network)
      self.network.AddNode(newout,self.network.OUTPUT)
