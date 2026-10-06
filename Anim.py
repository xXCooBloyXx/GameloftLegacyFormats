import numpy as np

from .. import BitStream, ByteStream
from .gameloft import Compression
from .PIG import NodeConfig, QuantizationLevel, readAttribute

"""
0(v4.2+)
-1(v? - v4.1) - compression type always lz4, dataCompressed flag
"""
version = 0

class Node:
	def read(self, stream):
		self.name = stream.readString()
		self.config = NodeConfig().read(stream)

		return self

class Material:
	def read(self, stream):
		self.name = stream.readString()
		self.textureCount = stream.readUInt16()

		return self

class FileConfig:
	def read(self, stream):
		bitStream = BitStream.fromUInt(stream.readUInt8(), 1)

		self.translationQuantized = bitStream.readBoolean()
		if version <= -1:
			self.dataCompressed = bitStream.readBoolean()
			assert bitStream.readUInt(6) == 0
		else:
			assert bitStream.readUInt(7) == 0

		return self

class KeyFrames:
	def read(self, stream, nodes, config):
		for node in nodes:
			positionFlags = stream.readUInt16()
			positionCount = positionFlags & 0x7FFF
			assert positionFlags >> 15 == 0 # Some kind of flag, same in rot and scale
			if positionCount:
				node.translationIndices = np.frombuffer(stream.readBytes(positionCount * 2), np.uint16)
				if config.translationQuantized:
					dataQuantizationLevel = QuantizationLevel(stream.readUInt8())
					unk1 = [stream.readFloat32() for _ in range(3)]
					unk2 = [stream.readFloat32() for _ in range(3)]

					node.translations = readAttribute(stream, positionCount, 3, dataQuantizationLevel, False)
					if dataQuantizationLevel != QuantizationLevel.NONE:
						node.translations *= unk1
						node.translations += unk2
				else:
					node.translations = readAttribute(stream, positionCount, 3, ("<f4", False), False)
			else:
				node.translationIndices = np.array([0], np.uint16)
				node.translations = np.array([[stream.readFloat32() for _ in range(3)]], np.float32)

		for node in nodes:
			rotationFlags = stream.readUInt16()
			rotationCount = rotationFlags & 0x7FFF
			assert rotationFlags >> 15 == 0
			if rotationCount:
				node.rotationIndices = np.frombuffer(stream.readBytes(rotationCount * 2), np.uint16)
				if node.config.unk1:
					xyz = readAttribute(stream, rotationCount, 3, QuantizationLevel.INT16, False)
					sqSum = np.sum(xyz ** 2, axis=1)
					w = np.sqrt(np.clip(1.0 - sqSum, 0.0, None))
					
					node.rotations = np.column_stack((xyz, w))
					
					norms = np.linalg.norm(node.rotations, axis=1, keepdims=True)
					node.rotations /= np.where(norms == 0, 1.0, norms)
				else:
					node.rotations = readAttribute(stream, rotationCount, 4, QuantizationLevel.INT16, False)
			else:
				node.rotationIndices = np.array([0], np.uint16)
				if node.config.unk1:
					xyz = readAttribute(stream, 1, 3, QuantizationLevel.INT16, False)
					sqSum = np.sum(xyz ** 2, axis=1)
					w = np.sqrt(np.clip(1.0 - sqSum, 0.0, None))
					node.rotations = np.column_stack((xyz, w))
					
					norms = np.linalg.norm(node.rotations, axis=1, keepdims=True)
					node.rotations /= np.where(norms == 0, 1.0, norms)
				else:
					node.rotations = readAttribute(stream, 1, 4, QuantizationLevel.INT16, False)
			node.rotations[:, :3] *= -1

		for node in nodes:
			scaleQuantized = stream.readBoolean()
			scaleFlags = stream.readUInt16()
			scaleCount = scaleFlags & 0x7FFF
			assert scaleFlags >> 15 == 0
			if scaleCount:
				node.scaleIndices = np.frombuffer(stream.readBytes(scaleCount * 2), np.uint16)
				
				if scaleQuantized:
					scaleQuantizationLevel = QuantizationLevel(stream.readUInt8())
					unk1 = [stream.readFloat32() for _ in range(3)]
					unk2 = [stream.readFloat32() for _ in range(3)]
					
					node.scales = readAttribute(stream, scaleCount, 3, scaleQuantizationLevel, False)
					if scaleQuantizationLevel != QuantizationLevel.NONE:
						node.scales *= unk1
						node.scales += unk2
				else:
					node.scales = readAttribute(stream, scaleCount, 3, ("<f4", False), False)
			else:
				node.scaleIndices = np.array([0], np.uint16)
				node.scales = np.array([[stream.readFloat32() for _ in range(3)]], np.float32)

class File:
	def __init__(self):
		self.frameRate = 30
		self.nodes = []
		self.materials = []

	def load(self, fileData):
		stream = ByteStream(fileData)

		self.frameRate = stream.readUInt32()

		self.nodes = [Node().read(stream) for _ in range(stream.readUInt16())]

		self.materials = [Material().read(stream) for _ in range(stream.readUInt16())]

		config = FileConfig().read(stream)
		
		if version <= -1:
			stream2 = ByteStream(Compression.readAndDecompress(stream, False, "LZ4")) if config.dataCompressed else stream
		else:
			stream2 = ByteStream(Compression.readAndDecompress(stream, False))

		KeyFrames().read(stream2, self.nodes, config)
		# materials, not implemented
		#stream2.readUInt8()

		return self