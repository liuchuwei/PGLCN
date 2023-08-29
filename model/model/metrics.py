import torch
from sklearn import metrics
import torch.nn.functional as F


def logit_BCE(preds, labels):


    if labels.size()[1]==1:
        y = labels.type(torch.int64)
        y_one_hot = torch.zeros(labels.size()[0],2).cuda()
        # y_one_hot = torch.zeros(labels.size()[0],2)
        y_one_hot = y_one_hot.scatter_(1,y,1)
        labels = y_one_hot

    # weights = torch.tensor([1, 100]).cuda()
    # loss = torch.nn.BCEWithLogitsLoss(weight=weights)(preds, labels)
    loss = torch.nn.BCEWithLogitsLoss()(preds, labels)

    return loss


def logit_accuracy(preds, labels):

    pred = torch.argmax(preds, dim=1).detach().cpu().numpy()
    acc = metrics.accuracy_score(labels.detach().cpu().numpy(), pred)

    return acc

def softmax_cross_entropy_with_logits(labels, logits, dim=-1):
    return (-labels * F.log_softmax(logits, dim=dim)).sum(dim=dim)

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    # loss = torch.nn.CrossEntropyLoss(reduction="none")(preds, labels)
    loss = softmax_cross_entropy_with_logits(labels, preds)
    # mask = mask.type(torch.float32)
    mask = mask/torch.mean(mask)
    # loss *= mask
    loss = loss * mask
    return torch.mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    if labels.size()[1]==1:
        y = labels.type(torch.int64)
        y_one_hot = torch.ones(labels.size()[0],2).cuda()
        y_one_hot = y_one_hot.scatter_(1,y,0)
        labels = y_one_hot

    correct_prediction = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1))
    accuracy_all = correct_prediction.type(torch.float32)
    # mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    accuracy_all *= mask
    return torch.mean(accuracy_all)
