class Root():
	"""node for a tree"""

	def __init__(self, value):
		self.value = value
		self.children = []

	def returnChildren(self):
		return self.children

	def returnValue(self):
		return self.value

	def addChild(self, child):
		self.children.append(child)

	def removeChild(self, child):
		if child in self.children:
			self.children.remove(child)
			return false
		else:
			return true

class Node(Root):
	def __init__(self, value, parent):
		Root.__init__(self,value)
		self.parent = parent

	def returnParent(self):
		return self.parent

	def returnChildren(self):
		Root.returnChildren(self)

	def returnValue(self):
		Root.returnValue(self)

	def addChild(self, child):
		Root.addChild(self, child)

	def removeChild(self, child):
		Root.removeChild(self, child)

tree = Root(0)
tree.addChild(Node(1,tree))
print(tree.children[0].value)

