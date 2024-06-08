
# Main PNG decoder entrypoint...

import sys
#import const
from const import * # Some constants
import struct # For reading binary data without having to worry about endianness etc etc..
import zlib # This is for zlib.crc32 only!
from own_zlib import *

'''

SOME NOTES

Length	4-byte unsigned integer giving the number of bytes in the chunk's data field. The length counts only the data field, not itself, the chunk type, or the CRC. Zero is a valid length.
Chunk Type	A sequence of 4 bytes defining the chunk type.
Chunk Data	The data bytes appropriate to the chunk type, if any. This field can be of zero length.
CRC	A 4-byte CRC (Cyclic Redundancy Code) calculated on the preceding bytes in the chunk, including the chunk type field and the chunk data fields, but not including the length field. The CRC can be used to check for corruption of the data.

'''

def read_chunks(data: bytes) -> list: # Returns a list of the binary chunks.
	chunks = []
	while True: # While there are chunks to be read.
		chunk_header = data[:SIZEOF_CHUNK_HEADER] # Read the chunk header.
		print("chunk_header == "+str(chunk_header))
		print("len(chunk_header) == "+str(len(chunk_header)))
		data = data[SIZEOF_CHUNK_HEADER:] # Advance the data.
		chunk_length, chunk_type = struct.unpack('>I4s', chunk_header) # Decode the chunk header
		# Now read the chunk contents.
		chunk_data = data[:chunk_length]
		data = data[chunk_length:] # Advance the data.
		calculated_checksum = zlib.crc32(chunk_data, zlib.crc32(struct.pack('>4s', chunk_type))) # Calculate checksum.
		crc_bytes = data[:CHUNK_CRC_SIZE]
		data = data[CHUNK_CRC_SIZE:] # Advance data pointer.
		chunk_crc, = struct.unpack('>I', crc_bytes) # The expected CRC.
		if chunk_crc != calculated_checksum: # CRC mismatch, therefore data is corrupted!
			print("File is corrupted!")
			exit(1)
		chunks.append(tuple((chunk_data, chunk_type))) # Add the chunk and the chunk type to the list.
		print("chunk_type == "+str(chunk_type))
		if chunk_type == IEND_CHUNK_IDENTIFIER: # The last chunk. Break now.
			break
	return chunks

# These are used in the different filters.

BYTES_PER_PIXEL = 4 # This is the hardcoded value for one byte RGBA images which is usually the case.

def Reconstruct_a(r,c,out,scanlines): # out is the current output array and scanlines are the.. ya know.. scanlines. :D
	# return Recon[r * stride + c - bytesPerPixel] if c >= bytesPerPixel else 0
	return out[r][c - BYTES_PER_PIXEL] if c >= BYTES_PER_PIXEL else 0 # "r * stride" is basically the current scanline and then the c is the current pixel thing.

def Reconstruct_b(r,c,out,scanlines):
	# return Recon[(r-1) * stride + c] if r > 0 else 0
	return out[r-1][c] if r > 0 else 0 # The same thing as the same pixel on the previous scanline.

def Reconstruct_c(r,c,out,scanlines):
	# return Recon[(r-1) * stride + c - bytesPerPixel] if r > 0 and c >= bytesPerPixel else 0
	return out[r-1][c - BYTES_PER_PIXEL] if r > 0 and c >= BYTES_PER_PIXEL else 0 # 

def PaethPredictor(a,b,c): # This is just a spec defined function
	p = a + b - c
	pa = abs(p - a)
	pb = abs(p - b)
	pc = abs(p - c)
	if pa <= pb and pa <= pc:
		Pr = a
	elif pb <= pc:
		Pr = b
	else:
		Pr = c
	return Pr


def read_png(data: bytes) -> None: # Just show as an image (for now).
	if data[:len(PNG_HEADER)] != PNG_HEADER: # Not a PNG file!
		print("File isn't a PNG file!")
		exit(1)

	global BYTES_PER_PIXEL # We may modify this when we read "colort" (the image type) . Up until then, this is assumed to be 4

	data = data[len(PNG_HEADER):] # Skip the PNG header for reading the chunks
	chunks = read_chunks(data)
	# Get the chunk types as a list.
	chunk_types = [chunk[1] for chunk in chunks]
	# The very first chunk should be a b'IHDR' chunk.
	#if chunk_types[0] != b'IHDR':
	#	print("Very first chunk should be a b'IHDR' chunk!")
	#	exit(1)
	assert chunk_types[0] == IHDR_CHUNK_IDENTIFIER # First chunk should be "IHDR"
	assert chunk_types[-1] == IEND_CHUNK_IDENTIFIER # Final chunk should be "IEND"
	assert IDAT_CHUNK_IDENTIFIER in chunk_types # There should be atleast one data chunk.

	# Now process the IHDR chunk:

	'''

	Field name	Field size	Description
	Width	4 bytes	4-byte unsigned integer. Gives the image dimensions in pixels. Zero is an invalid value.
	Height	4 bytes
	Bit depth	1 byte	A single-byte integer giving the number of bits per sample or per palette index (not per pixel). Only certain values are valid (see below).
	Color type	1 byte	A single-byte integer that defines the PNG image type. Valid values are 0 (grayscale), 2 (truecolor), 3 (indexed-color), 4 (greyscale with alpha) and 6 (truecolor with alpha).
	Compression method	1 byte	A single-byte integer that indicates the method used to compress the image data. Only compression method 0 (deflate/inflate compression with a sliding window of at most 32768 bytes) is defined in the spec.
	Filter method	1 byte	A single-byte integer that indicates the preprocessing method applied to the image data before compression. Only filter method 0 (adaptive filtering with five basic filter types) is defined in the spec.
	Interlace method	1 byte	A single-byte integer that indicates whether there is interlacing. Two values are defined in the spec: 0 (no interlace) or 1 (Adam7 interlace).

	'''

	ihdr_chunk, _ = chunks[0] # Get the IHDR chunk.

	# Unpack the values from that chunk...

	width, height, bitd, colort, compm, filterm, interlacem = struct.unpack('>IIBBBBB', ihdr_chunk)
	
	# Check for the image type. The usual case is the colort == 6 case (truecolor with alpha)

	# Let's check for the stuff.

	assert colort == 6 or colort == 2 # We only support truecolor with alpha (6) or truecolor (2).

	if colort == 2: # Truecolor without alpha channel, therefore set BYTES_PER_PIXEL to three instead of four.
		BYTES_PER_PIXEL = 3

	# Check for the bitdepth. (Must be 8 for now).

	assert bitd == 8

	if compm != 0:
		print('invalid compression method')
		exit(1)
	if filterm != 0:
		print('invalid filter method')
		exit(1)

	chunks = chunks[1:] # Get rid of the IHDR chunk.

	# Now go over each IDAT chunk...

	idat_data = b''.join(chunk[0] for chunk in chunks if chunk[1] == IDAT_CHUNK_IDENTIFIER)

	print("Here is the IDAT data concatenated: "+str(idat_data))

	# Now use our own version of zlib.decompress to decompress it...

	decompressed_data = our_decompress(idat_data)
	print("Here is the decompressed data: "+str(decompressed_data))
	#decompressed_data = zlib.decompress(idat_data)
	print("Here is the length of the data: "+str(len(decompressed_data)))
	

	'''
	# Thanks to https://www.geeksforgeeks.org/break-list-chunks-size-n-python/ !!!

	    for i in range(0, len(l), n):  
        yield l[i:i + n] 
	
	'''

	# Just assume picture is 8 bit RGBA for now.

	scanline_size = 1 + width * BYTES_PER_PIXEL # Four bytes per pixel times the amount of pixels plus one, because the very first byte is the filter type.

	scanlines = [decompressed_data[i:i + scanline_size] for i in range(0, len(decompressed_data), scanline_size)]

	out = [[0 for _ in range(len(scanlines[0]) - 1)] for _ in range(len(scanlines))] # Final RGBA image. Zero out first, such that we do not need to do shit with this later on. " - 1" , because the first byte of the scanline is the filter type.
	


	# Just print the filter types for each scanline...

	filter_types = [scanline[0] for scanline in scanlines] # Just show the filter types for now.

	# Go over each scanline and add the data to the output list.
	tot_values = 0
	for r, scanline in enumerate(scanlines):
		# Main loop

		filt_type = scanline[0]

		scanline = scanline[1:] # Cut out the filter type byte.
		print("scanline == "+str(scanline))
		print("len(scanline) == "+str(len(scanline)))
		print("BYTES_PER_PIXEL == "+str(BYTES_PER_PIXEL))
		print("colort == "+str(colort))
		assert len(scanline) == width * BYTES_PER_PIXEL
		for c, byte in enumerate(scanline): # Loop over each byte in the 


			tot_values += 1

			match filt_type: # Switch case basically. (This is only in python3.10 and upwards)
				case 0: # "None"
					reconstructed = byte
				case 1:
					reconstructed = byte + Reconstruct_a(r, c, out, scanlines)
				case 2:
					reconstructed = byte + Reconstruct_b(r, c, out, scanlines)
				case 3:
					reconstructed = byte + (Reconstruct_a(r, c, out, scanlines) + Reconstruct_b(r, c, out, scanlines)) // 2
				case 4:
					reconstructed = byte + PaethPredictor(Reconstruct_a(r, c, out, scanlines), Reconstruct_b(r, c, out, scanlines), Reconstruct_c(r, c, out, scanlines)) # Paeth stuff.
				case _: # Undefined filter type.
					print("Invalid filter type: "+str(filt_type))
					exit(1)

			out[r][c] = reconstructed & 0xff # Place the reconstructed byte into the output.

	#print(filter_types)

	print("tot_values == "+str(tot_values))

	# Now we have a reconstructed image in out.


	image_bytes = []

	for line in out:
		#image_bytes += line
		image_bytes.extend(line)
	#print("Here are the final bytes: "+str(image_bytes))


	# Now show the final output:

	import matplotlib.pyplot as plt
	import numpy as np
	plt.imshow(np.array(image_bytes).reshape((height, width, BYTES_PER_PIXEL)))
	plt.show()

	print("[+] Done!")
	return 0 # Return success

def main() -> int: # Main function
	# Read contents from the file supplied in argv[1]
	if len(sys.argv) < 2: # aka 1
		print("Usage: "+str(sys.argv[0])+" PNG_FILE")
		exit(1)
	filename = sys.argv[1]
	# Open file and read file as bytes
	fh = open(filename, "rb")
	data = fh.read()
	fh.close()
	# Now decode the data.
	read_png(data)
	return 0 # Return success


if __name__=="__main__":
	exit(main())
