import ctypes
import os
from enum import IntEnum

from ...Compression import decompressLZ4

zstd = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libzstd.dll")) # Zstandard python lib doesnt support gameloft zstd compression

zstd.ZSTD_decompress.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
zstd.ZSTD_decompress.restype = ctypes.c_size_t

zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
zstd.ZSTD_isError.restype = ctypes.c_uint

zstd.ZSTD_getErrorName.restype = ctypes.c_char_p
zstd.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]

def decompressZstd(data, decompressedDataSize):
	#if isinstance(data, bytearray):
	#	data = memoryview(data)
	
	decompressedDataBuffer = ctypes.create_string_buffer(decompressedDataSize)

	actualDecompressedDataSize = zstd.ZSTD_decompress(decompressedDataBuffer, decompressedDataSize, data, len(data))
	if zstd.ZSTD_isError(actualDecompressedDataSize):
		raise RuntimeError(zstd.ZSTD_getErrorName(actualDecompressedDataSize).decode())

	return decompressedDataBuffer.raw[:actualDecompressedDataSize]

class CompressionAlgorithm(IntEnum):
	NONE = 0
	LZ4 = 1
	ZSTD = 2
	BROTLI = 3
	ZIP = 7
	LZHAM = 8

def readAndDecompress(stream, isCompressionOptional=True, forcedCompressionAlgorithm=None):
	if isCompressionOptional and (dataLength := stream.readUInt32()):
		return stream.readBytes(dataLength)

	compressionAlgorithm = CompressionAlgorithm(stream.readUInt8()) if forcedCompressionAlgorithm is None else CompressionAlgorithm[forcedCompressionAlgorithm]
	if compressionAlgorithm == CompressionAlgorithm.NONE:
		return stream.buffer[stream.offset:]
	
	compressedDataLength = stream.readUInt32()
	decompressedDatalength = stream.readUInt32()
	compressedData = stream.readBytes(compressedDataLength)

	match compressionAlgorithm:
		case CompressionAlgorithm.ZSTD:
			return decompressZstd(compressedData, decompressedDatalength)
		case CompressionAlgorithm.LZ4:
			return decompressLZ4(compressedData, decompressedDatalength)
		case _:
			raise NotImplementedError(f"Unsupported compression algorithm: {compressionAlgorithm!r}")