
# This file implements various tests for this zlib decompression library. This file tests for example the bitreader for various operations etc etc.

import bitreader
from helpers import * # Test the helper functions too
from huffman import *
from lz77 import *

def test_bitreader() -> None:
	# Just verify that we read the same bytes as are in the thing.
	byte_input_bytes = [0x01, 0x02, 0x03, 0x04, 0xff, 0xdd]
	byte_inputs = bytes(byte_input_bytes)
	reader = bitreader.Bitreader(byte_inputs)
	for k in range(len(byte_input_bytes)):
		print("New byte...")
		cur_int = 0
		for i in range(8):
			bit_val = int(reader.read_bit())
			#cur_int <<= 1 # shift to the left once
			cur_int |= ( bit_val << i )
		# Check for the stuff
		assert byte_input_bytes[k] == cur_int
		print("Here is the value which we read: "+str(bin(cur_int)))


	'''
	r = BitReader(b'\x9d')
	print(r.read_bit()) # 1
	print(r.read_bit()) # 0
	print(r.read_bit()) # 1
	print(r.read_bit()) # 1
	print(r.read_bit()) # 1
	print(r.read_bit()) # 0
	print(r.read_bit()) # 0
	print(r.read_bit()) # 1
	print(r.read_bit()) # IndexError: index out of range
	'''

	reader = bitreader.Bitreader([0x9d])

	correct_answer = [1,0,1,1,1,0,0,1]

	for i in range(len(correct_answer)):
		print("Checking..")
		assert reader.read_bit() == correct_answer[i]

	print("Bit reader test passed!!!")

	return


# def int_to_bytes(integer: int, n: int) -> bytes: # Encodes "integer" to a bytestring of length "n" bits.


def test_int_to_bytes() -> None:
	correct_answer = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1] # We should receive these bits sequentially when reading
	r = bitreader.Bitreader(int_to_bytes(0b11101000001, 11)) # Here we convert it to bytes.
	for bit in correct_answer:
		assert r.read_bit() == bit # Should match
	print("test_int_to_bytes passed!!!")
	return


def test_huffman_tree() -> None:

	# First create the huffman tree...
	tree = HuffmanTree()

	# Insert some leaves and nodes...

	tree.add_value(0b01, 2, "A")
	tree.add_value(0b1, 1, "B")
	tree.add_value(0b000, 3, "C")
	tree.add_value(0b001, 3, "D")

	# Try reading some of the values from the tree.

	r = bitreader.Bitreader(int_to_bytes(0b11101000001, 11)) # Some stuff.

	correct_answers = ["B", "B", "B", "A", "C", "D"]

	for i in range(len(correct_answers)):
		sym = tree.read_symbol(r)
		print("Here is the current symbol: "+str(sym))
		assert sym == correct_answers[i]

	print("test_huffman_tree passed!!!")

	return


def test_bitlengths_to_tree() -> None:

	# bitlengths_to_tree(bitlengths, alphabet)
	alphabet = 'ABCD'
	bl = [2, 1, 3, 3]

	tree = bitlengths_to_tree(bl, alphabet)

	r = bitreader.Bitreader(int_to_bytes(0b00010110111, 11))

	correct_answers = ["B", "B", "B", "A", "C", "D"]

	for i in range(len(correct_answers)):
		# This should decode the correct answer...
		sym = tree.read_symbol(r)
		#print("We read this symbol: "+str(sym))
		assert sym == correct_answers[i]

	print("test_bitlengths_to_tree passed!!!")
	return 



def run_tests() -> int: # runs all of the tests

	test_bitreader()
	test_int_to_bytes()
	test_huffman_tree()
	test_bitlengths_to_tree()

	print("[+] All tests passed!")


	return 0

if __name__=="__main__":
	exit(run_tests())
