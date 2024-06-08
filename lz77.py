
# Implements the LZ77 compression algorithm https://en.wikipedia.org/wiki/LZ77_and_LZ78#LZ77

from huffman import * # This is needed for the huffman trees.
from helpers import * # Some helper functions.
from bitreader import *

def bitlengths_to_tree(bitlengths: list, alphabet: list) -> HuffmanTree:
	# Converts a list of bitlengths to a huffman tree with the given alphabet (the alphabet is assumed to already be in order)

	# This function returns a so called "canonical" huffman tree.

	max_bit_len = max(bitlengths) # Get the maximum length of all the lengths.

	bit_length_counts = [bitlengths.count(x) if x != 0 else 0 for x in range(max_bit_len+1)] # This is a list of the count of all of the lengths, for example the value at index 0 of this list is the amount of occurences of the value of zero in the bitlengths list. then the value at index 1 is the total amount of the value 1 which is in the bitlength list and so on...


	# "Next, we compute next_code such that next_code[n] is the smallest codeword with code length n."

	# First initialize next_code to [0,0], because, when n == 0, aka code length is zero, then the smallest codeword is nothing, so it doesn't really matter, then on n == 1 case the smallest codeword is 0b0 , because the other possibility is 0b1, which is of course larger, so we automatically know the first two.

	next_code = [0, 0]

	for i in range(2, max_bit_len+1): # Go through all of the bitlengths.

		# I don't really understand how this works, but let's return to this later. DONOTUNDERSTAND

		next_code.append((next_code[i-1] + bit_length_counts[i-1]) << 1)

		'''
		
		next_code = [0, 0]
		for bits in range(2, MAX_BITS+1):
			next_code.append((next_code[bits-1] + bl_count[bits-1]) << 1)
		print(next_code) # [0, 0, 2, 6]

		'''

	#print("Here is the value of next_code: "+str(next_code))

	output_tree = HuffmanTree()

	for c, bitlength in zip(alphabet, bitlengths):
		if bitlength != 0:
			# def add_value(self, path_int: int, n: int, value) -> None: # A wrapper around add_node
			#print("Now trying to add "+str(c)+" to bit path "+str(bin(next_code[bitlength])[2:]))
			output_tree.add_value(next_code[bitlength], bitlength, c) # Add the alphabet to the thing.
			next_code[bitlength] += 1 # This works, because initially next_code[n] is the smallest codeword with bitlength zero, and we need to add one to this, because then when we encounter another code with the same bitlength, we add the new value (aka the value which we added one to). This assumes that the data is not malformed, because for example we could have many codes with the same bitlength, and then when we add to this number, we can flow over and next_code[n] is of bitlength n+1 instead and that messes up our decoding.

	return output_tree # Return the decoded tree object...

'''

However, the DEFLATE spec adds yet another twist. Rather than just specifying the code lengths directly, for "even greater compactness", the code length sequences themselves are compressed using a Huffman code! The alphabet for the code lengths is as follows:

'''

# code_length_orders = [16, 17, 18, 0, 8, 7, 9, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15] # This is the order of the the bitlengths when decoding the code length sequences.

code_length_orders = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15] # This should be correct maybe????

def decode_trees(r) -> tuple: # This shit decodes the trees from the bitreader object r
	HLIT = r.read_n_bits(5) + 257
	HDIST = r.read_n_bits(5) + 1
	HCLEN = r.read_n_bits(4) + 4

	
	code_length_tree_bitlengths = [0 for _ in range(19)] # Just assign zeros for now. This makes it easy to use the CodeLengthsCodesOrder to actually put the values at the correct indexes.

	for i in range(HCLEN):
		code_length_tree_bitlengths[code_length_orders[i]] = r.read_n_bits(3)

	print("code_length_tree_bitlengths == "+str(code_length_tree_bitlengths))

	# Construct the codelength tree. This will be used to actually get the codelengths.

	code_length_tree = bitlengths_to_tree(code_length_tree_bitlengths, [x for x in range(19)])

	print("Here is the code length tree: "+str(code_length_tree))

	# Read literal/length + distance code length list, this will take advantage of the previously generated code_length_tree .

	bitlengths = [] # This will be the list passed to bitlengths_to_tree which will give us the very final tree used to decode the data.


	'''

	Symbol	Meaning

	0 - 15	Represent code lengths of 0 - 15
	16	    Copy the previous code length 3 - 6 times. The next 2 bits indicate repeat length ( 0 = 3, ..., 3 = 6)
	17	    Repeat a code length of 0 for 3 - 10 times. (3 bits of length)
	18	    Repeat a code length of 0 for 11 - 138 times. (7 bits of length)

	'''

	while len(bitlengths) < HLIT + HDIST: # Decode the literal/(length + distance) trees.
		symbol = code_length_tree.read_symbol(r) # Read a symbol from the code length tree.
		print("Decoding this symbol while decoding the trees: "+str(symbol))
		if 0 <= symbol <= 15:
			bitlengths.append(symbol)
		elif symbol == 16: # This means to copy the previous code length 3 - 6 times.
			repeat_count = r.read_n_bits(2) + 3 # Read how many times to repeat
			what_to_repeat = bitlengths[-1] # What to repeat
			bitlengths += [what_to_repeat for _ in range(repeat_count)] # Repeat code
		elif symbol == 17: # repeat 3..10 times .
			repeat_count = r.read_n_bits(3) + 3 # Read how many times to repeat
			bitlengths += [0 for _ in range(repeat_count)] # Repeat code
		elif symbol == 18: # repeat 11..138 times.
			repeat_count = r.read_n_bits(7) + 11 # Read how many times to repeat
			bitlengths += [0 for _ in range(repeat_count)] # Repeat code
		else:
			print("Invalid symbol when decoding length tree: "+str(symbol))
			assert False

	print("Here are the bitlengths when decoding the trees: "+str(bitlengths))

	# Final tree construction. Now that we have the bitlengths, we can finally get our literal/(length + distance) trees and the distance tree
	literal_length_distance_tree = bitlengths_to_tree(bitlengths[:HLIT], [x for x in range(286)]) # alphabet is to 0-285 inclusive
	distances_tree = bitlengths_to_tree(bitlengths[HLIT:], [x for x in range(30)]) # this is the tree used to decode the backwards distances.
	return literal_length_distance_tree, distances_tree

# These are the tables used in the decompression of the lz77 data backwards lengths

length_extra_bits = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0] # These are the extra bits when reading the length codes from the bitstream.
length_bases = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258] # These are the length bases. We are going to add the integer to these when reading the bitstream.
# These are the tables used in the decoding of the backwards distances...
backwards_distance_extra_bits = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]

backwards_distance_bases = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577]

'''
[0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]




[0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]

'''

# Decodes a block from the bitstream with the given literal/length and distance trees.

def lz77_decode_block(r: Bitreader, literal_length_tree: HuffmanTree, distance_tree: HuffmanTree, output: list) -> None: # The output is the list of bytes to output. This function modifies it in-place.
	while True: # Main decoding loop.
		val = literal_length_tree.read_symbol(r) # Get value
		print("Decoded this value: "+str(val))
		if val < 256: # Literal value
			output.append(val)
		elif val == 256: # End of block
			return output # Return the final data
		else: # The value encodes the length portion.
			symbol = val - 257
			print("length_bases[symbol] == "+str(length_bases[symbol]))
			print("length_extra_bits[symbol] == "+str(length_extra_bits[symbol]))
			# Now read the extra bits. and add it to the baselength to get the final length
			final_length = r.read_n_bits(length_extra_bits[symbol]) + length_bases[symbol]
			# Now read the distance amount in a similar fashion
			distance_symbol = distance_tree.read_symbol(r)
			
			print("backwards_distance_extra_bits[distance_symbol] == "+str(backwards_distance_extra_bits[distance_symbol]))
			print("backwards_distance_bases[distance_symbol] == "+str(backwards_distance_bases[distance_symbol]))
			final_distance = r.read_n_bits(backwards_distance_extra_bits[distance_symbol]) + backwards_distance_bases[distance_symbol]
			# Now we have the final <length, distance> pair decoded from the bitstream. add to the output.
			# Take advantage of pythons ability to access with negative indexes. Note that this works, because the index [-n] changes as we are appending to the list.
			print("final_length == "+str(final_length))
			print("final_distance == "+str(final_distance))
			for _ in range(final_length):
				#print("final_distance == "+str(final_distance))
				output.append(output[-1*final_distance])

	return output # Return the final byte list.





