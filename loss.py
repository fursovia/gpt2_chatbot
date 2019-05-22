import torch
import torch.nn as nn


def cosine_distance(x, y):
    """
    Args:
        x: tensor of shape [batch_size, emb_size]
        y: tensor of shape [batch_size, emb_size]
    Returns:
        cos_dists: tensor of shape [batch_size, batch_size]
    """

    cos_dists = None

    return cos_dists


def triplet_loss(context_embeddings, answer_embeddings, margin=0.2):

    cos_dists = cosine_distance(context_embeddings, answer_embeddings)
    anchor, positive, negative = sample(cos_dists)

    # TODO: fix
    loss = torch.relu(anchor - positive + negative + margin)

    return loss


def margin_loss(context_embeddings, answer_embeddings):
    pass


def contrastive_loss(context_embeddings, answer_embeddings):
    pass


def sample(distances, strategy='all'):
    # all, hard, weighted
    anchor, positive, negative = None, None, None

    return anchor, positive, negative
