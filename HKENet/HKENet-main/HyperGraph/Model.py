from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import pairPredict, contrastLoss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(131, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(131, args.latdim)))
		self.gcnLayer = GCNLayer()
		self.hgnnLayer = HGNNLayer()
		self.uHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		self.iHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))

		self.edgeDropper = SpAdjDropEdge()

	def forward(self, adj, keepRate=0.2):
		# embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		embeds = self.uEmbeds + self.iEmbeds
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		uuHyper = self.uEmbeds @ self.uHyper
		iiHyper = self.iEmbeds @ self.iHyper

		for i in range(args.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			# hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-keepRate), lats[-1][:131])
			# hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-keepRate), lats[-1][131:])
			hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1 - keepRate), lats[-1])
			hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1 - keepRate), lats[-1])
			gnnLats.append(temEmbeds)
			hyperLats.append(hyperULat + hyperILat)
			lats.append(temEmbeds + hyperLats[-1])

		embeds = sum(lats)
		return 0.0101*embeds, gnnLats, hyperLats
		# return 0.0101*embeds + self.uEmbeds + self.iEmbeds, gnnLats, hyperLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:131], embeds[131:]
		
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		sslLoss = 0
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = hyperEmbedsLst[i]
			sslLoss += contrastLoss(embeds1[:131], embeds2[:131], t.unique(ancs), args.temp) + contrastLoss(embeds1[131:], embeds2[131:], t.unique(poss), args.temp)
		return bprLoss, sslLoss
	
	def predict(self, adj):
		embeds, _, _ = self.forward(adj, 0.2)
		return embeds[:131], embeds[131:]

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=args.leaky)

	def forward(self, adj, embeds):
		return (t.spmm(adj, embeds))

class HGNNLayer(nn.Module):
	def __init__(self):
		super(HGNNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=args.leaky)
	
	def forward(self, adj, embeds):
		# lat = self.act(adj.T @ embeds)
		# ret = self.act(adj @ lat)
		lat = (adj.T @ embeds)
		ret = (adj @ lat)
		return ret

class SpAdjDropEdge(nn.Module):
	def __init__(self):
		super(SpAdjDropEdge, self).__init__()

	def forward(self, adj, keepRate):
		return adj
		if keepRate == 1.0:
			return adj
		vals = adj#._values()
		idxs = adj#._indices()
		edgeNum = vals.size()#.size()
		mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
		newVals = vals[mask] / keepRate

		# print(idxs.shape)
		# print(mask.shape)
		# newIdxs = idxs[:, mask]
		newIdxs = idxs[mask]
		# print(newIdxs.size())
		# print(newVals.size())
		# print(adj[0].size())
		return t.sparse.FloatTensor(newIdxs, newVals, adj[0].size())
