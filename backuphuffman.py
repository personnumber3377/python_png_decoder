
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

	def add_node(direction: int, child: Node) -> None: # This adds a child node to this node. direction is left or right (0 is left and 1 is right)
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
			#
			assert cur_node != None # We should not try to traverse downwards a nonexistent path.
		# Now we have reached the leaf. Verify that it actually is a leaf.
		assert cur_node.isLeaf()
		# Return the leaf
		return cur_node

	def get_val(self, path_int: int, n: int): # This function is identical to traverse_tree, but returns the value of the node instead of the node. This is a convenience function such that we do not have to type ".value" every time.
		return (self.traverse_tree(path_int, n)).value

	def add_node(self, node, path_int: int, n: int) -> None:
		# This adds a node at the path "path_int". This functions assumes that the path to the parent child of the node which is going to be added exists already. For example if the path 0b11 does not exist and you try to call this function with 0b111 , then this function will error out and will NOT create the 0b11 node for you.

		# First get the node and then call add_node on that.
		parent_node = self.traverse_tree(path_int << 1, n) # Here we need to discard the very last bit of the path, because that will be the thing which we are adding
		# The direction is just the last bit of path_int.
		direction = path_int & 0b1
		assert direction == RIGHT or direction == LEFT
		# Then just call add_node on that.
		parent_node.add_node(direction, node)
		return




