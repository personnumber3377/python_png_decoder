
# This file defines some constants used by the PNG spec...

PNG_HEADER = b'\x89PNG\r\n\x1a\n' # Should be at the start of the file.
SIZEOF_CHUNK_HEADER = 8 # How many bytes each chunk has before the content bytes.
CHUNK_CRC_SIZE = 4 # How many bytes the chunk CRC is in length.
# IEND_CHUNK_IDENTIFIER = b'\x00\x00\x00\x00IEND' # Chunk which signifies the final chunk.

IHDR_CHUNK_IDENTIFIER = b'IHDR' # Header chunk
IDAT_CHUNK_IDENTIFIER = b'IDAT' # Data chunk.
IEND_CHUNK_IDENTIFIER = b'IEND' # Chunk which signifies the final chunk.

