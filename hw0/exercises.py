# Import the math module, so that we have access to the functions sqrt and pow
import math

def factorial(x):
  """Return x!, assuming that x is a non-negative integer."""
  fac = 1
  while x > 0:
    fac = fac * x
    x -= 1
  return fac
    
def sumFile(filename):
  """Each line of filename contains a float.  Return the sum of all lines in the
  file."""
  infile = open('test_sum_file.txt', 'r')
  total = 0
  for num in infile:
    total += float(num)
  return total
  

# This is the syntax for a Python class.
class Point():
  """Class that encapsulates a single point in the x-y plane."""

  # This is the constructor for the class.  By convention, the first argument to
  # any method of the class is self, referring to the variable itself.  This is
  # similar to the "this" variable in other programming languages.
  def __init__(self, x_coord, y_coord):
    self.x = x_coord
    self.y = y_coord

  def distanceFromOrigin(self):
    d = math.sqrt(self.x ** 2 + self.y ** 2)
    return d
