import torch
import torch.nn.functional as F


change_weight = 3.0
no_change_weight = 1.0

# Define the weighted cross-entropy loss function
def weighted_cross_entropy_loss(predicted, target):
    # Squeeze target to remove the channel dimension
    target = target.squeeze(1).long()  # Convert target to long and [batch_size, height, width] format
    # Reshape weights to match predicted tensor
    weights_tensor = torch.tensor([no_change_weight, change_weight], device=target.device)
    # Compute cross-entropy loss with the weights
    # import pdb; pdb.set_trace()
    loss = torch.nn.functional.cross_entropy(predicted, target, weight=weights_tensor)
    return loss


def consistency_loss(pred1, pred2):
    return F.mse_loss(pred1, pred2)

def entropy_loss(pred):
    probs = torch.softmax(pred, dim=1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean()
