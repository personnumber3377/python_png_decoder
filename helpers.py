
import math

def type_check(input, type) -> bool: # Checks that the input is of type "type"
	assert isinstance(input, type)

def strict_val_check(a, b) -> bool:
	if type(a) != type(b):
		# differ by type
		return False
	if a != b:
		# differ by value
		return False
	return True # Match

def get_nth_bit(val: int, n: int) -> int: # Get the n'th bit from the value called "val".
	return ( val >> n ) & 0b1 # First shift to get the correct bit and then and to get that bit

def int_to_bytes(integer: int, n: int) -> bytes: # Encodes "integer" to a bytestring of length "n" bits.

	'''

	r = BitReader(code_to_bytes(0b11101000001, 11))
	print([r.read_bit() for _ in range(11)]) # [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1]

	'''

	output = [0] # Must be initialized with the zero value.

	cur_bit = 0
	for i in range(n-1, -1, -1): # Go from n-1 to 0 inclusive
		if cur_bit >= 8: # Skip over to the next byte.
			output.append(0)
			cur_bit = 0
		# Operate on the very last byte.
		output[-1] |= ((1 << cur_bit) if integer & (1 << i) else 0)
		cur_bit += 1
	return bytes(output)



