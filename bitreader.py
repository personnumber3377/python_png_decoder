
# Implements a bitreader class which is used to interpret the zlib data from the datastream.

from helpers import *

class Bitreader:
	
	def __init__(self, data: bytes): # Initializes the bitreader with the supplied data.
		self.current_byte = 0 # index of the current byte.
		self.current_bit = 0 # index of the current bit being read.
		self.data = data

	def read_bit(self) -> bool: # Reads one bit from the bitstream
		byte_val = self.data[self.current_byte] # Get the current byte in the data which we are working with.
		bit_val = ( byte_val >> self.current_bit ) & 0b00000001 # Shift to the current bit and then and it with 0b00000001 to get the current bit.
		#print("bit_val == "+str(bit_val))
		#print("type(bit_val) == "+str(type(bit_val)))
		assert strict_val_check(bit_val, 1) or strict_val_check(bit_val, 0) # the value which we read must be 1 or 0 and of type integer.
		# Update counters.
		self.current_bit += 1
		# Check if we skip to the next byte
		if self.current_bit == 8:
			self.current_bit = 0
			self.current_byte += 1
			#print("self.current_byte == "+str(self.current_byte))
			#print("self.data[self.current_byte] == "+str(self.data[self.current_byte]))
		return bool(bit_val) # convert to boolean type before returning

	def read_byte(self) -> int: # Reads a byte from the bitstream
		output = 0 # output integer
		for count in range(8): # 8 bits in a byte
			bit = self.read_bit()
			#print("Here is the bit: "+str(bit))
			#output <<= 1 # first shift to the left, because the bit which we just read will take up bit position 0
			output |= ( int(bit) << count ) # append bit to integer (must first convert to integer, because "bit" is actually of type bool)
		return int(output) # Return as integer
	
	def read_n_bytes(self, n) -> int: # Reads the next n bits as an integer
		output = 0
		for count in range(n): # read n bytes
			bit = self.read_byte()
			assert 0 <= bit <= 255
			#output <<= 8
			output |= ( int(bit) << (count*8) )
		return int(output) # Return as integer

	def read_n_bits(self, n) -> int: # Reads the next n bits as an integer
		output = 0
		for count in range(n): # read n bytes
			bit = self.read_bit()
			assert 0 <= bit <= 1
			#output <<= 1
			output |= (int(bit) << count )
		return int(output) # Return as integer

	def skip_to_start_of_next_byte(self) -> None: # This should skip to the start of the next byte.
		self.current_bit = 0 # First bit...
		self.current_byte += 1 # of the next byte.
		return

