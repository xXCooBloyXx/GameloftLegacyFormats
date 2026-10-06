from ... import ByteStream, PVRTexTool
from . import Compression

PVR_VERSION_3_MAGIC = b"PVR\x03"

def decode(fileData):
	if not fileData.startswith(PVR_VERSION_3_MAGIC):
		raise NotImplementedError(f"Only PVR textures of version 3 are supported. Expected {PVR_VERSION_3_MAGIC}, got {fileData[:4]}")

	stream = ByteStream(fileData)

	stream2 = ByteStream()
	stream2.writeBytes(stream.readBytes(44))

	mipmapCount = stream2.writeUInt32(stream.readUInt32())

	metadataSize = stream.readUInt32()
	stream2.writeUInt32(0) # Metadata Size

	if not metadataSize:
		decompressedFileData = stream.buffer
	else:
		mipmapsDataOffset = stream.offset + metadataSize
		compressionSignature = stream.readBytes(8)
		if compressionSignature == b"zstdjetc":
			compressionAlgorithm = "Zstd"
		elif compressionSignature == b"\x00JET\x00LZ4":
			compressionAlgorithm = "LZ4"
		else:
			raise NotImplementedError(f"Unsupported compression algorithm. Signature: {compressionSignature}")

		stream.readUInt32() # Mipmaps Metadata Size	

		for _ in range(mipmapCount):
			compressedDataOffset = mipmapsDataOffset + stream.readUInt32()
			compressedDataLength = stream.readUInt32()
			decompressedDataLength = stream.readUInt32()

			compressedData = fileData[compressedDataOffset:compressedDataOffset + compressedDataLength]
			decompressedData = getattr(Compression, "decompress" + compressionAlgorithm)(compressedData, decompressedDataLength)
			stream2.writeBytes(decompressedData)
		
		decompressedFileData = stream2.buffer

	return PVRTexTool.decode(decompressedFileData, True) # Returns PIL.Image