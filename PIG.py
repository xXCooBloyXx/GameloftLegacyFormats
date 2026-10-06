from enum import IntEnum

import numpy as np

from .. import BitStream, ByteStream
from .gameloft import Compression

""" I used negative versioning for easier future support of potential older versions
0(minion rush v4.2+, 10.5.0c [*])
-1(minion rush v? - v4.1) - compression type always lz4, dataCompressed flag
The oldest version ive found was in "The Adventures of Tintin: The Secret of the Unicorn"

Partial implementation of oldest minion rush version PIG files

def readAttribute(stream, elementCount, componentCount, quantizationLevel, areElementsAligned=True):
	stream.offset += stream.readUInt16() + 2
	#print("reading attribute at", stream.offset, "shape:", elementCount, "x", componentCount, quantizationLevel)
	if isinstance(quantizationLevel, QuantizationLevel):
		if quantizationLevel == QuantizationLevel.NONE:
			componentDataType = "<f4"
			normalize = False
		elif quantizationLevel == QuantizationLevel.INT16:
			componentDataType = "<i2"
			normalize = True
		elif quantizationLevel == QuantizationLevel.INT8:
			componentDataType = "<i1"
			normalize = True
	else:
		componentDataType, normalize = quantizationLevel
	componentDataType = np.dtype(componentDataType)
	
	elementSize = componentDataType.itemsize * componentCount
	if areElementsAligned:
		elementSize += -elementSize % 4

	result = np.lib.stride_tricks.as_strided(
		np.frombuffer(stream.readBytes(elementCount * elementSize), componentDataType),
		(elementCount, componentCount),
		(elementSize, componentDataType.itemsize)
	)
	
	if normalize:
		result = result.astype(np.float32)
		result /= np.iinfo(componentDataType).max
	
	return result

class SubMesh:
	def read(self, stream, skinned):
		assert stream.readUInt32() == 100
		
		vertexAttributesQuantization = VertexAttributesQuantization().read(stream)
		vertexAttributesConfig = VertexAttributesConfig().read(stream)
		
		self.unk = [stream.readFloat32() for _ in range(3)]
		if vertexAttributesQuantization.position != QuantizationLevel.NONE:
			self.translation = [stream.readFloat32() for _ in range(3)]
			self.scale = [stream.readFloat32() for _ in range(3)]
		else:
			self.translation = None
			self.scale = None
		
		vertexCount = stream.readUInt16()
		indexCount = stream.readUInt32()
		
		self.materialName = stream.readString()
		self.textures = [stream.readString() for _ in range(stream.readUInt16())]
		
		assert stream.readInt8() == 0 # Unused?
		
		stream2 = ByteStream(Compression.readAndDecompress(stream))

		try:
			self.positions = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.position) if vertexAttributesConfig.hasPosition else None
			self.normals = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.normal) if vertexAttributesConfig.hasNormal else None

			self.tangents0 = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.tangent) if vertexAttributesConfig.hasTangent0 else None
			self.tangents1 = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.tangent) if vertexAttributesConfig.hasTangent1 else None
			self.tangents2 = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.tangent) if vertexAttributesConfig.hasTangent2 else None
			self.tangents3 = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.tangent) if vertexAttributesConfig.hasTangent3 else None

			self.colors = readAttribute(stream2, vertexCount, 4, ("<u1", True)) if vertexAttributesConfig.hasColor else None

			self.texcoords0 = readAttribute(stream2, vertexCount, 2, vertexAttributesQuantization.texcoord0) if vertexAttributesConfig.hasTexcoord0 else None
			self.texcoords1 = readAttribute(stream2, vertexCount, 2, vertexAttributesQuantization.texcoord1) if vertexAttributesConfig.hasTexcoord1 else None
			self.texcoords2 = readAttribute(stream2, vertexCount, 2, vertexAttributesQuantization.texcoord2) if vertexAttributesConfig.hasTexcoord2 else None
			self.texcoords3 = readAttribute(stream2, vertexCount, 2, vertexAttributesQuantization.texcoord3) if vertexAttributesConfig.hasTexcoord3 else None

			self.jointIndices = readAttribute(stream2, vertexCount, 4, ("<u1", False)) if skinned else None
			self.weights = readAttribute(stream2, vertexCount, 4, ("<u1", True)) if skinned else None

			self.triangles = readAttribute(stream2, indexCount, 1, ("<i2", False), False).reshape(-1, 3)
		except EOFError:
			self.skip = True
			print("failed", self.materialName)
		
		self.skin = Skin().read(stream) if skinned else None
		
		return self

"""
version = 0

def readAttribute(stream, elementCount, componentCount, quantizationLevel, areElementsAligned=True):
	if isinstance(quantizationLevel, QuantizationLevel):
		if quantizationLevel == QuantizationLevel.NONE:
			componentDataType = "<f4"
			normalize = False
		elif quantizationLevel == QuantizationLevel.INT16:
			componentDataType = "<i2"
			normalize = True
		elif quantizationLevel == QuantizationLevel.INT8:
			componentDataType = "<i1"
			normalize = True
	else:
		componentDataType, normalize = quantizationLevel
	componentDataType = np.dtype(componentDataType)
	
	elementSize = componentDataType.itemsize * componentCount
	if areElementsAligned:
		elementSize += -elementSize % 4
	
	result = np.lib.stride_tricks.as_strided(
		np.frombuffer(stream.readBytes(elementCount * elementSize), componentDataType),
		(elementCount, componentCount),
		(elementSize, componentDataType.itemsize)
	)
	
	if normalize:
		result = result.astype(np.float32)
		result /= np.iinfo(componentDataType).max
	
	return result

class NodeConfig:
	def read(self, stream):
		bitStream = BitStream.fromUInt(stream.readUInt8(), 4)
		
		self.isDummy = bitStream.readBoolean()
		self.unk1 = bitStream.readBit()
		self.unk2 = bitStream.readBit()
		self.unk3 = bitStream.readBit()
		self.unk4 = bitStream.readBit()
		self.unk5 = bitStream.readBit()
		self.unk6 = bitStream.readBit()
		self.unk7 = bitStream.readBit() # Related to cameras
		
		return self

class NodeUnk2:
	def read(self, stream):
		raise Exception # I havent come across a case where this struct was used so...
		self.unk1 = stream.readString()
		self.unk2 = stream.readString()

class Node:
	def read(self, stream):
		assert stream.readUInt32() == 100
		
		self.name = stream.readString()
		self.config = NodeConfig().read(stream)
		self.parentIndex = stream.readInt16()
		
		self.translation = [stream.readFloat32() for _ in range(3)]
		# JET Engine uses reversed bone roll direction
		self.rotation = [-stream.readFloat32(), -stream.readFloat32(), -stream.readFloat32(), stream.readFloat32()]
		self.scale = [stream.readFloat32() for _ in range(3)]
		
		self.unk1 = stream.readFloat32() # wth is this?
		
		self.unks2 = [NodeUnk2().read(stream) for _ in range(stream.readUInt16())] # Another user datas?
		
		return self

class QuantizationLevel(IntEnum):
	NONE = 0
	INT16 = 1
	INT8 = 2

class VertexAttributesQuantization:
	def read(self, stream):
		bitStream = BitStream.fromUInt(stream.readUInt32() & ~0x80000000, 4)
		
		self.position = QuantizationLevel(bitStream.readUInt(2))
		self.normal = QuantizationLevel(bitStream.readUInt(2))
		self.tangents = QuantizationLevel(bitStream.readUInt(2))
		
		self.texcoord0 = QuantizationLevel(bitStream.readUInt(1))
		self.texcoord1 = QuantizationLevel(bitStream.readUInt(1))
		self.texcoord2 = QuantizationLevel(bitStream.readUInt(1))
		self.texcoord3 = QuantizationLevel(bitStream.readUInt(1))
		
		assert bitStream.readUInt(22) == 0 # Unused? highest bit is related to weights or something with meena?
		
		return self

class VertexAttributesConfig:
	def read(self, stream):
		flags = stream.readUInt32()
		if not flags: # Ive never found such file but its like this in game parser
			raise Exception
			stream.readInt32()
			stream.readInt32()
			flags = stream.readUInt32()
		
		bitStream = BitStream.fromUInt(flags, 4)
		
		self.hasPosition = bitStream.readBoolean()
		self.hasNormal = bitStream.readBoolean()
		
		self.hasTangent0 = bitStream.readBoolean()
		self.hasTangent1 = bitStream.readBoolean()
		self.hasTangent2 = bitStream.readBoolean()
		self.hasTangent3 = bitStream.readBoolean()
		
		self.hasColor = bitStream.readBoolean()
		
		self.hasTexcoord0 = bitStream.readBoolean()
		self.hasTexcoord1 = bitStream.readBoolean()
		self.hasTexcoord2 = bitStream.readBoolean()
		self.hasTexcoord3 = bitStream.readBoolean()
		
		assert bitStream.readUInt(21) == 0
		
		return self	

class Skin:
	def read(self, stream):
		self.unk1 = stream.readUInt16()
		jointCount = stream.readUInt16()
		
		stream2 = ByteStream(Compression.readAndDecompress(stream, forcedCompressionAlgorithm="LZ4" if version <= -1 else None))
			
		self.jointIndices = []
		self.unk2 = []
		for _ in range(jointCount):
			self.unk2.append([[stream.readFloat32() for _ in range(3)] for _ in range(2)]) # min and max of something
			unk3 = stream2.readUInt16()
			unk4 = stream2.readUInt16()
			assert unk3 == unk4 # Maybe index of the node and index used in bone indices?
			self.jointIndices.append(unk3)
		
		self.inverseBindMatrices = np.asarray([stream2.readMatrix() for _ in range(jointCount)]) # TODO: Numpy
		
		return self

class SubMesh:
	def read(self, stream, skinned):
		assert stream.readUInt32() == 100
		
		vertexAttributesQuantization = VertexAttributesQuantization().read(stream)
		vertexAttributesConfig = VertexAttributesConfig().read(stream)
		
		self.unk = [stream.readFloat32() for _ in range(3)]
		if vertexAttributesQuantization.position != QuantizationLevel.NONE:
			self.translation = [stream.readFloat32() for _ in range(3)]
			self.scale = [stream.readFloat32() for _ in range(3)]
		else:
			self.translation = None
			self.scale = None
		
		vertexCount = stream.readUInt16()
		indexCount = stream.readUInt32()
		
		self.materialName = stream.readString()
		self.textures = [stream.readString() for _ in range(stream.readUInt16())]
		
		assert stream.readInt8() == 0
		
		stream.align(16)
		stream2 = ByteStream(Compression.readAndDecompress(stream, forcedCompressionAlgorithm="LZ4" if version <= -1 else None))

		# TODO: Move this part to "Vertices" or smth?
		self.positions = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.position) if vertexAttributesConfig.hasPosition else None
		self.normals = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.normal) if vertexAttributesConfig.hasNormal else None
		
		self.tangents0 = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.tangents) if vertexAttributesConfig.hasTangent0 else None
		self.tangents1 = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.tangents) if vertexAttributesConfig.hasTangent1 else None
		self.tangents2 = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.tangents) if vertexAttributesConfig.hasTangent2 else None
		self.tangents3 = readAttribute(stream2, vertexCount, 3, vertexAttributesQuantization.tangents) if vertexAttributesConfig.hasTangent3 else None
		
		self.colors = readAttribute(stream2, vertexCount, 4, ("<u1", True)) if vertexAttributesConfig.hasColor else None
		
		self.texcoords0 = readAttribute(stream2, vertexCount, 2, vertexAttributesQuantization.texcoord0) if vertexAttributesConfig.hasTexcoord0 else None
		self.texcoords1 = readAttribute(stream2, vertexCount, 2, vertexAttributesQuantization.texcoord1) if vertexAttributesConfig.hasTexcoord1 else None
		self.texcoords2 = readAttribute(stream2, vertexCount, 2, vertexAttributesQuantization.texcoord2) if vertexAttributesConfig.hasTexcoord2 else None
		self.texcoords3 = readAttribute(stream2, vertexCount, 2, vertexAttributesQuantization.texcoord3) if vertexAttributesConfig.hasTexcoord3 else None
		
		self.jointIndices = readAttribute(stream2, vertexCount, 4, ("<u1", False)) if skinned else None
		self.weights = readAttribute(stream2, vertexCount, 4, ("<u1", True)) if skinned else None
		
		self.triangles = np.cumsum(readAttribute(stream2, indexCount, 1, ("<i2", False), False)).reshape(-1, 3)
		
		self.skin = Skin().read(stream) if skinned else None
		
		return self

class Mesh:
	def read(self, stream):
		self.lod = stream.readUInt8()
		
		assert stream.readUInt32() == 100
		
		skinned = stream.readBoolean()
		assert stream.readUInt8() == 0
		self.boundingBox = [[stream.readFloat32() for _ in range(3)] for _ in range(2)]
		
		self.subMeshes = [SubMesh().read(stream, skinned) for _ in range(stream.readUInt16())]
		
		return self

class MultiResMesh:
	def read(self, stream):
		assert stream.readUInt32() == 100
		
		self.nodeIndex = stream.readUInt32()
		self.meshes = [Mesh().read(stream) for _ in range(stream.readUInt16())]
		
		return self

class Camera:
	def read(self, stream):
		self.node = Node().read(stream)
		self.name = stream.readString()

		self.fovX = stream.readFloat32()
		self.aspectRatio = stream.readFloat32()
		self.nearPlane = stream.readFloat32()
		self.farPlane = stream.readFloat32()
		self.type = stream.readUInt8() # Ortographic or perspective, TODO:
		
		return self

class File:
	def __init__(self):
		self.nodes = []
		self.unk = 1
		self.multiResMeshes = []
		self.cameras = []
		self.userDatas = {}
	
	def load(self, fileData):
		stream = ByteStream(fileData)
		
		assert stream.readUInt32() == 100 # Useful fields that werent actually updated(probably)
		
		self.nodes = [Node().read(stream) for _ in range(stream.readUInt16())]
		
		self.unk = stream.readUInt8()
		assert self.unk in (1, 2) # Some kind of mode?
		
		self.multiResMeshes = [MultiResMesh().read(stream) for _ in range(stream.readUInt16())]
		self.cameras = [Camera().read(stream) for _ in range(stream.readUInt16())]
		if not stream.isAtEnd():
			self.userDatas = {stream.readString(): stream.readString() for _ in range(stream.readUInt16())}
		
		return self