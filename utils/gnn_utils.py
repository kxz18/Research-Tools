#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
beta version
'''
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


@torch.no_grad()
def graph_to_batch(tensor, batch_id, padding_value=0, mask_is_pad=True):
    '''
    :param tensor: [N, D1, D2, ...]
    :param batch_id: [N]
    :param mask_is_pad: 1 in the mask indicates padding if set to True
    '''
    lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
    bs, max_n = lengths.shape[0], torch.max(lengths)
    batch = torch.ones((bs, max_n, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) * padding_value
    # generate pad mask: 1 for pad and 0 for data
    pad_mask = torch.zeros((bs, max_n + 1), dtype=torch.long, device=tensor.device)
    pad_mask[(torch.arange(bs, device=tensor.device), lengths)] = 1
    pad_mask = (torch.cumsum(pad_mask, dim=-1)[:, :-1]).bool()
    data_mask = torch.logical_not(pad_mask)
    # fill data
    batch[data_mask] = tensor
    mask = pad_mask if mask_is_pad else data_mask
    return batch, mask


@torch.no_grad()
def length_to_batch_id(S, lengths):
    # generate batch id
    batch_id = torch.zeros_like(S)  # [N]
    batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
    batch_id.cumsum_(dim=0)  # [N], item idx in the batch
    return batch_id


def _edge_dist(X, atom_pad_mask, src_dst):
    '''
    :param X: [N, n_channel, 3], coordinates
    :param atom_pad_mask: [N, n_channel], mark the padding positions as 1
    :param src_dst: [Ef, 2], all edges that needs distance calculation, represented in (src, dst)
    '''
    BIGINT = 1e10  # assign a large distance to invalid edges
    dist = X[src_dst]  # [Ef, 2, n_channel, 3]
    dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
    dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
    atom_pad = atom_pad_mask[src_dst]  # [Ef, 2, n_channel]
    atom_pad = torch.logical_or(atom_pad[:, 0].unsqueeze(2), atom_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
    dist = dist + atom_pad * BIGINT  # [Ef, n_channel, n_channel]
    dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
    return dist


def _knn_edges(X, AP, src_dst, atom_pos_pad_idx, k_neighbors, batch_info, given_dist=None):
    '''
    :param X: [N, n_channel, 3], coordinates
    :param AP: [N, n_channel], atom position with pad type need to be ignored
    :param src_dst: [Ef, 2], full possible edges represented in (src, dst)
    :param given_dist: [Ef], given distance of edges
    '''
    offsets, batch_id, max_n, gni2lni = batch_info

    BIGINT = 1e10  # assign a large distance to invalid edges
    N = X.shape[0]
    if given_dist is None:
        dist = _edge_dist(X, AP == atom_pos_pad_idx, src_dst)
    else:
        dist = given_dist
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]

    dist_mat = torch.ones(N, max_n, device=dist.device, dtype=dist.dtype) * BIGINT  # [N, max_n]
    dist_mat[(src_dst[0], gni2lni[src_dst[1]])] = dist
    del dist
    dist_neighbors, dst = torch.topk(dist_mat, k_neighbors, dim=-1, largest=False)  # [N, topk]

    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k_neighbors)
    src, dst = src.flatten(), dst.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)

    dst = dst + offsets[batch_id[src]]  # mapping from local to global node index

    edges = torch.stack([src, dst])  # message passed from dst to src

    return edges  # [2, E]


def _radial_edges(X, atom_pad_mask, src_dst, dist_cut_off, given_dist=None):
    '''
    :param X: [N, n_channel, 3], coordinates
    :param atom_pad_mask: [N, n_channel], mark the padding positions as 1
    :param src_dst: [Ef, 2], full possible edges represented in (src, dst)
    :param given_dist: [Ef], given distance of edges
    '''
    if given_dist is None:
        dist = _edge_dist(X, atom_pad_mask, src_dst)
    else:
        dist = given_dist
    is_valid = dist < dist_cut_off
    src_dst = src_dst[is_valid]
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]
    return src_dst


class BatchEdgeConstructor:
    '''
    Construct intra-segment edges (intra_edges) and inter-segment edges (inter_edges) with O(Nn) complexity,
    where n is the largest number of nodes of one graph in the batch.
    Additionally consider global nodes: 
        global nodes will connect to all nodes in its segment (global_normal_edges)
        global nodes will connect to each other regardless of the segments they are in (global_global_edges)
    Additionally consider edges between adjacent nodes in the sequence in the same segment (seq_edges)
    '''

    def __init__(self, global_node_id_vocab) -> None:
        self.global_node_id_vocab = copy(global_node_id_vocab)

        # buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.row = None
        self.col = None
        self.row_global = None
        self.col_global = None
        self.row_seg = None
        self.col_seg = None
        self.offsets = None
        self.max_n = None
        self.gni2lni = None
        self.not_global_edges = None

    def get_batch_edges(self, batch_id):
        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        return (row, col), (offsets, max_n, gni2lni)

    def _prepare(self, S, batch_id, segment_ids) -> None:
        (row, col), (offsets, max_n, gni2lni) = self.get_batch_edges(batch_id)

        # not global edges
        is_global = sequential_or(*[S == global_node_id for global_node_id in self.global_node_id_vocab]) # [N]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        
        # segment ids
        row_seg, col_seg = segment_ids[row], segment_ids[col]

        # add to buffer
        self.row, self.col = row, col
        self.offsets, self.max_n, self.gni2lni = offsets, max_n, gni2lni
        self.row_global, self.col_global = row_global, col_global
        self.not_global_edges = not_global_edges
        self.row_seg, self.col_seg = row_seg, col_seg

    def _construct_intra_edges(self, X, atom_pad_mask, batch_id):
        row, col = self.row, self.col
        # all possible ctx edges: same seg, not global
        select_edges = torch.logical_and(self.row_seg == self.col_seg, self.not_global_edges)
        intra_all_row, intra_all_col = row[select_edges], col[select_edges]
        return torch.stack([intra_all_row, intra_all_col])

    def _construct_inter_edges(self, X, atom_pad_mask, batch_id):
        row, col = self.row, self.col
        # all possible inter edges: not same seg, not global
        select_edges = torch.logical_and(self.row_seg != self.col_seg, self.not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        return torch.stack([inter_all_row, inter_all_col])

    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = torch.logical_and(self.row_global, self.col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    def _construct_seq_edges(self):
        row, col = self.row, self.col
        # add additional edge to neighbors in 1D sequence (except epitope)
        select_edges = sequential_and(
            torch.logical_or((row - col) == 1, (row - col) == -1),  # adjacent in the graph order
            self.not_global_edges  # not global edges (also ensure the edges are in the same segment)
        )
        seq_adj = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return seq_adj

    @torch.no_grad()
    def construct_edges(self, X, atom_pad_mask, S, batch_id, k_neighbors, segment_ids):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # prepare inputs
        self._prepare(S, batch_id, segment_ids)

        # intra-segment edges
        intra_edges = self._construct_intra_edges(X, atom_pad_mask, batch_id, k_neighbors)

        # inter-segment edges
        inter_edges = self._construct_inter_edges(X, atom_pad_mask, batch_id, k_neighbors)

        # edges between global nodes and normal/global nodes
        global_normal_edges, global_global_edges = self._construct_global_edges()

        # edges on the 1D sequence
        seq_edges = self._construct_seq_edges()

        self._reset_buffer()

        return intra_edges, inter_edges, global_normal_edges, global_global_edges, seq_edges


class KNNBatchEdgeConstructor(BatchEdgeConstructor):
    def __init__(self, global_node_id_vocab, k_neighbors) -> None:
        super().__init__(global_node_id_vocab)
        self.k_neighbors = k_neighbors

    def _construct_intra_edges(self, X, atom_pad_mask, batch_id):
        all_intra_edges = super()._construct_intra_edges(X, atom_pad_mask, batch_id)
        # knn
        intra_edges = _knn_edges(
            X, atom_pad_mask, all_intra_edges.T, self.k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return intra_edges
    
    def _construct_outer_edges(self, X, atom_pad_mask, batch_id):
        all_inter_edges = super()._construct_outer_edges(X, atom_pad_mask, batch_id)
        # knn
        inter_edges = _knn_edges(
            X, atom_pad_mask, all_inter_edges.T, self.k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inter_edges
    

class RadialBatchEdgeConstructor(BatchEdgeConstructor):
    def __init__(self, global_node_id_vocab, dist_cutoff) -> None:
        super().__init__(global_node_id_vocab)
        self.dist_cutoff = dist_cutoff
    
    def _construct_intra_edges(self, X, atom_pad_mask, batch_id):
        all_intra_edges = super()._construct_intra_edges(X, atom_pad_mask, batch_id)
        # radial (cutoff with distance)
        intra_edges = _radial_edges(X, atom_pad_mask, all_intra_edges.T, self.dist_cutoff)
        return intra_edges
    
    def _construct_outer_edges(self, X, atom_pad_mask, batch_id):
        all_inter_edges = super()._construct_outer_edges(X, atom_pad_mask, batch_id)
        # radial (cutoff with distance)
        inter_edges = _radial_edges(X, atom_pad_mask, all_inter_edges.T, self.dist_cutoff)
        return inter_edges
