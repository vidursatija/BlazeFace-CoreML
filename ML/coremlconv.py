# smpl export
import numpy as np
import pickle
import torch
from torch.nn import Module
import os
import math
from torch.autograd import Variable
from tqdm import tqdm
from blazeface import BlazeFace


class AnchorModel(Module):
	def __init__(self, anchors):
		super(AnchorModel, self).__init__()
		self.anchors = torch.nn.Parameter(anchors)

	def forward(self, raw_boxes):
		"""Converts the predictions into actual coordinates using
		the anchor boxes. Processes the entire batch at once.
		"""
		# print(raw_boxes.size())
		# boxes = torch.zeros(size=(896, 16)) # raw_boxes.clone() # torch.zeros_like(raw_boxes)

		x_center = (raw_boxes[:, 0] / 128.0) * self.anchors[:, 2] + self.anchors[:, 0]
		y_center = (raw_boxes[:, 1] / 128.0) * self.anchors[:, 3] + self.anchors[:, 1]

		w = (raw_boxes[:, 2] / 128.0) * self.anchors[:, 2]
		h = (raw_boxes[:, 3] / 128.0) * self.anchors[:, 3]

		concat_stuff = []
		concat_stuff.append(y_center - h / 2.0)
		concat_stuff.append(x_center - w / 2.0)
		concat_stuff.append(y_center + h / 2.0)
		concat_stuff.append(x_center + w / 2.0)

		# raw_boxes[:, 0] = y_center - h / 2.  # ymin
		# raw_boxes[:, 1] = x_center - w / 2.  # xmin
		# raw_boxes[:, 2] = y_center + h / 2.  # ymax
		# raw_boxes[:, 3] = x_center + w / 2.  # xmax


		for k in range(6):
		    offset = 4 + k*2
		    # raw_boxes[:, offset    ] = (raw_boxes[:, offset    ] / 128.0) * self.anchors[:, 2] + self.anchors[:, 0] # x
		    # raw_boxes[:, offset + 1] = (raw_boxes[:, offset + 1] / 128.0) * self.anchors[:, 3] + self.anchors[:, 1] # y
		    concat_stuff.append((raw_boxes[:, offset    ] / 128.0) * self.anchors[:, 2] + self.anchors[:, 0])
		    concat_stuff.append((raw_boxes[:, offset + 1] / 128.0) * self.anchors[:, 3] + self.anchors[:, 1])

		return torch.stack(concat_stuff, dim=-1)


class BlazeFaceScaled(Module):
	def __init__(self, bfModel, decodeBoxModel):
		super(BlazeFaceScaled, self).__init__()
		self.bfModel = bfModel
		self.decodeBoxModel = decodeBoxModel

	def forward(self, x):
		# x = x/127.5 - 1 # We do this because it's more convenient to scale in PyTorch than CoreML
		out = self.bfModel(x)
		c = out[1].sigmoid() # .squeeze() # 896
		r = self.decodeBoxModel(out[0].squeeze()).unsqueeze(0) # .view(896, 8, 2)
		return r, c # torch.cat([r, c], dim=-1) # 896, 17

import coremltools as ct
from coremltools.converters.onnx import convert

bfModel = BlazeFace()
bfModel.load_weights("./blazeface.pth")
bfModel.load_anchors("./anchors.npy")

decodeBoxModel = AnchorModel(bfModel.anchors)

bfs = BlazeFaceScaled(bfModel, decodeBoxModel)
bfs.eval()

traced_model = torch.jit.trace(bfs, torch.rand(1, 3, 128, 128), check_trace=True)
# print(traced_model)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=ct.Shape(shape=(1, 3, 128, 128,)), bias=[-1,-1,-1], scale=1/127.5)]
)
mlmodel.save('BlazeFaceScaled.mlmodel')

print(mlmodel)
# Save converted CoreML model


# result = mlmodel.predict({"betas_pose_trans": x, "v_personal": y}, usesCPUOnly=True)