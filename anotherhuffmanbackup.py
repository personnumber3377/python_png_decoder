
# This file implements huffman coding and huffman trees.

from helpers import *
import bitreader

# These are completely arbitrary

RIGHT = 1
LEFT = 0

class Node:

	def __init__(self) -> None: # Constructor, which just initializes stuff.
		# These values are going to be modified externally
		self.left = None
		self.right = None
		self.parent = None
		self.value = None # This will be only set on leaf nodes

	def isLeaf(self) -> bool: # This checks if this node is a leaf of the tree.
		if self.value != None:
			return True
		return False

	def isTrunk(self) -> bool: # Check if this node is not a leaf. this is equal to "not isLeaf()" .
		return not self.isLeaf()

	def add_node(self, direction: int, child) -> None: # This adds a child node to this node. direction is left or right (0 is left and 1 is right)
		assert self.value == None # Because we are adding a child node, this node can not be a leaf, and therefore must not have a value associated.
		assert direction == RIGHT or direction == LEFT
		if direction == RIGHT:
			self.right = child
		else:
			self.left = child
		# Set the parent of the child node as this node.
		child.parent = self
		return




class HuffmanTree:

	def __init__(self) -> None: # Constructor, which just initializes some values.
		self.root = Node() # The root of the tree.

	def traverse_tree(self, path_int: int, n: int, add_nodes: bool = False): # This functions traverses the tree by following the code path_int which is of length n bits
		# First convert the path to DEFLATE compliant bytes and then read it bit by bit and traverse the tree.
		path_bytes = int_to_bytes(path_int, n)
		# Now make a reader
		r = bitreader.Bitreader(path_bytes)
		cur_node = self.root # Start from the root.
		for _ in range(n):
			bit_val = r.read_bit()
			if bit_val == LEFT:
				cur_node = cur_node.left
			else:
				cur_node = cur_node.right
			# If add_nodes is true, then we are currently adding a leaf to the node, so
			assert cur_node != None # We should not try to traverse downwards a nonexistent path.
		# Now we have reached the leaf. Verify that it actually is a leaf.
		assert cur_node.isLeaf()
		# Return the leaf
		return cur_node

	def get_val(self, path_int: int, n: int): # This function is identical to traverse_tree, but returns the value of the node instead of the node. This is a convenience function such that we do not have to type ".value" every time.
		return (self.traverse_tree(path_int, n)).value

	def add_node(self, node, path_int: int, n: int) -> None:
		node = self.root

		path_bytes = int_to_bytes(path_int, n)
		# Now make a reader
		r = bitreader.Bitreader(path_bytes)

		# Now traverse the tree and add nodes if needed.
		for _ in range(n):
			bit_val = r.read_bit() # Get the current bit...
			if bit_val == RIGHT:
				child = node.right
				if child == None: # Check if that node is nonexistent...
					child = Node() # Create it.
					# def add_node(direction: int, child: Node)
					node.add_node(bit_val, child)
			else: # LEFT
				child = node.left
				if child == None: # Check if that node is nonexistent...
					child = Node() # Create it.
					# def add_node(direction: int, child: Node)
					node.add_node(bit_val, child)
			# Set the current node to the child node (aka traverse down the tree)
			node = child
		# Set the value of the new node.
		#node.value = value # This is done in the calling add_value function

	def add_value(self, path_int: int, n: int, value) -> None: # A wrapper around add_node
		new_node = Node()
		new_node.value = value
		self.add_node(new_node, path_int, n)
		return

	def read_symbol(self, reader: bitreader.Bitreader): # This function reads bits from the reader until a valid path is found and then returns the value of that path.
		cur_node = self.root # Start from the root.
		while True: # Loop.
			bit_val = reader.read_bit()
			if bit_val == LEFT:
				cur_node = cur_node.left
			else:
				cur_node = cur_node.right
			# If add_nodes is true, then we are currently adding a leaf to the node, so
			# Check if the current node is a leaf, if yes, then return that value...
			if cur_node.isLeaf():
				assert cur_node.value != None # Should have a value assigned to it.
				return cur_node.value 



