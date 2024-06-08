
import zlib
from bitreader import *
from helpers import *
from lz77 import *
from huffman import *




def inflate_no_compression(reader, output):
	# Implements noncompressed blocks.
	# The LENGTH and NLENGTH fields are byte aligned, therefore skip to the start of the next byte.
	reader.skip_to_start_of_next_byte()
	LENGTH = reader.read_n_bytes(2) # The first two bytes have the length
	NLENGTH = reader.read_n_bytes(2) # This is the ones complement of LENGTH
	# Sanity checks.
	print("Here is LENGTH: "+str(bin(LENGTH)))
	print("Here is NLENGTH: "+str(bin(NLENGTH)))

	assert LENGTH & NLENGTH == 0 # No bits should match.
	assert (~NLENGTH) & LENGTH == LENGTH # The complemented version of NLENGTH should be LENGTH
	assert ((~NLENGTH) & 0xffff) == LENGTH
	# Add the bytes.
	print("Here is the LENGTH: "+str(LENGTH))
	stuff_to_add = []
	for _ in range(LENGTH):
		byte_contents = reader.read_byte()
		print("Here is the current byte: "+str(chr(byte_contents)))
		stuff_to_add.append(byte_contents)

	output += stuff_to_add # add.

	return


'''
def inflate_fixed_block(a,b):
	assert False # Not implemented

def inflate_block_fixed(r, o):
    bl = ([8 for _ in range(144)] + [9 for _ in range(144, 256)] +
        [7 for _ in range(256, 280)] + [8 for _ in range(280, 288)])
    literal_length_tree = bl_list_to_tree(bl, range(286))

    bl = [5 for _ in range(30)]
    distance_tree = bl_list_to_tree(bl, range(30))

    inflate_block_data(r, literal_length_tree, distance_tree, o)

'''


def inflate_fixed_block(reader, output) -> None: # Fixed block. Uses a predetermined literal_length tree and a backwards distance tree.
	bl = ([8 for _ in range(144)] + [9 for _ in range(144, 256)] +
	[7 for _ in range(256, 280)] + [8 for _ in range(280, 288)])
	literal_length_tree = bitlengths_to_tree(bl, range(286))

	bl = [5 for _ in range(30)]
	distance_tree = bitlengths_to_tree(bl, range(30))

	# def bitlengths_to_tree(bitlengths: list, alphabet: list) -> HuffmanTree:

	lz77_decode_block(reader, literal_length_tree, distance_tree, output)
	return


def inflate_block_dynamic(reader, output) -> None:
	# First get the trees from the bitstream, then use these trees to decode the block data.
	# decode_trees(reader)
	literal_length_tree, distance_tree = decode_trees(reader) # Get the trees.
	# def lz77_decode_block(r: Bitreader, literal_length_tree: HuffmanTree, distance_tree: HuffmanTree, output: list) -> None:
	lz77_decode_block(reader, literal_length_tree, distance_tree, output) # This will modify output in-place.
	return




def inflate(reader) -> bytes:
	# Main decompression algorith

	# Loop over each block.

	FINAL_BLOCK = 0
	output = [] # This will be the output bytes.
	while not FINAL_BLOCK:
		FINAL_BLOCK = reader.read_bit()
		BLOCK_TYPE = reader.read_n_bits(2)
		print("Here is BLOCK_TYPE: "+str(BLOCK_TYPE))
		#assert BLOCK_TYPE == 0 # Should be uncompressed
		if BLOCK_TYPE == 0: # Block isn't compressed
			inflate_no_compression(reader, output)
		elif BLOCK_TYPE == 1:
			# A so called "fixed" block.
			inflate_fixed_block(reader, output)
		elif BLOCK_TYPE == 2:
			# A so called "dynamic" block.
			inflate_block_dynamic(reader, output)
		else:
			print("Invalid block type: "+str(BLOCK_TYPE))
			assert False
	return bytes(output) # Convert list of integers to a bytestring before returning


def our_decompress(input_data: bytes) -> bytes:
	# This is the main decompression function...
	# Create a reader from the data.
	reader = Bitreader(input_data)
	# Now try to read the four first bytes which should be equal to 8 in decimal
	CMF = reader.read_byte()

	print("CMF == "+str(CMF))
	CM = CMF & 15
	CINFO = ( CMF & 0b11110000 ) >> 4

	'''
	 Compression info. This is the base-2 logarithm of the LZ77 window size, minus eight. In other words, the window size is . The maximum window size that is allowed by the spec is 32768, which is . In other words, CINFO must be , and any other value should be treated as an error.

	'''

	# Must be less than equal to seven

	assert CINFO <= 7
	print("CINFO == "+str(CINFO))
	LZ77_WINDOW_SIZE = 2 ** ( CINFO + 8 )

	print("CM == "+str(CM))
	assert CM == 8
	print("CM test passed... (CM is equal to 8)")

	# The next byte is the flag byte (or FLG for short)

	FLG = reader.read_byte()

	# Bits 0 to 4: FCHECK --- Used as part of the  checksum. See below.

	FCHECK = FLG & 0b11111 # First four bytes
	FDICT = ( FLG & 0b100000 ) >> 5

	#  Compression level. This indicates whether the original data was compressed with the fastest/fast/default/max-compression compression level. It's not needed for de-compression at all, and is only there to indicate if recompression might be worthwhile. For our purposes, we can simply ignore it.

	FLEVEL = ( FLG >> 6 )

	# These flag values also act as a checksum of sorts.

	if ( CMF * 256 + FLG ) % 31 != 0:
		print("zlib header checksum failure")
		assert False

	output = inflate(reader) # Main decompression algorithm.

	CHECKSUM = reader.read_n_bytes(4)

	return output

def main() -> int: # Main function

	#example_data = bytes("Hello world!", encoding="ascii") # Example data

	# Now try to compress it using zlib.

	#compressed_data = zlib.compress(example_data)

	# Try with non-compressed data (for now)..

	#input_string = "aabaabaabaabaabaabaabaabaabbaaavabababaaaaaabaababaabaaaabaabaabaabaabaabaabaabaabbaaavabababaaaaaabaababaabaaaabaabaabaabaabaabaabaabaabbaaavabababaaaaaabaababaabaaaabaabaabaabaabaabaabaabaabbaaavabababaaaaaabaababaabaabbbbbaabaababaabaaaabaababaabaaaabaababaabaaabababaaaaaabbababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaababaaababaabaaaabaaabababaaaaaabaababaabaaaabaaabababaaaaaabaababaabaaaabaafefebbabababaaaabbaabababaaaabbaaababaaaabbaaababaaaabbababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababababaaaaaabaababaabaabbbbbaabaababaabaaaabaabababaa"

	input_string = "Hello World!"

	#compressed_data = zlib.compress(bytes(input_string, encoding="ascii"), level=0)

	compressed_data = zlib.compress(bytes(input_string, encoding="ascii"))

	print(compressed_data)

	# Try to use our decompressing function on the compressed input.

	result = our_decompress(compressed_data)

	print("Here is the result: "+str(result))

	# Decode the result to ascii. It should be the same as the original string.

	out_as_string = result.decode("ascii")

	print("Here is the decompressed output: "+str(out_as_string))


	# Sanity check.
	assert input_string == out_as_string

	return 0

if __name__=="__main__":
	exit(main())
