import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.dice_loss import DiceLoss,ShiftDiceLoss
from loss.cross_entropy import CrossentropyLoss, TopKLoss, DynamicTopKLoss, LabelSmoothing

#---------------------------------seg loss---------------------------------
class CEPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus cross entropy
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(CEPlusDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        # print(predict.size())
        # print(target.size())
        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict,target)

        ce = CrossentropyLoss(weight=self.weight)
        ce_loss = ce(predict,target)
        
        total_loss = ce_loss + dice_loss

        return total_loss


class CEPlusTopkDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus cross entropy
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(CEPlusTopkDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        # print(predict.size())
        # print(target.size())
        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict,target)

        ce = CrossentropyLoss(weight=self.weight)
        ce_loss = ce(predict,target)
        
        total_loss = ce_loss + dice_loss

        return total_loss


class TopkCEPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus topk cross entropy 
    """
    def __init__(self, weight=None, ignore_index=None, alpha=1, beta=1, **kwargs):
        super(TopkCEPlusDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta

    def forward(self, predict, target):

        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index)
        dice_loss = dice(predict,target)

        topk = TopKLoss(weight=self.weight,**self.kwargs)
        topk_loss = topk(predict,target)
        
        total_loss = self.alpha*topk_loss + self.beta*dice_loss

        return total_loss


class TopkCEPlusTopkDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus topk cross entropy 
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(TopkCEPlusTopkDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):

        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict,target)

        topk = TopKLoss(weight=self.weight,**self.kwargs)
        topk_loss = topk(predict,target)
        
        total_loss = topk_loss + dice_loss

        return total_loss


class TopkCEPlusShiftDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus topk cross entropy 
    """
    def __init__(self, weight=None, shift=0.5, ignore_index=None, **kwargs):
        super(TopkCEPlusShiftDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.shift = shift
        self.ignore_index = ignore_index

    def forward(self, predict, target):

        assert predict.size() == target.size()
        dice = ShiftDiceLoss(weight=self.weight,shift=self.shift,ignore_index=self.ignore_index)
        dice_loss = dice(predict,target)

        topk = TopKLoss(weight=self.weight,**self.kwargs)
        topk_loss = topk(predict,target)
        
        total_loss = topk_loss + dice_loss

        return total_loss


class DynamicTopkCEPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus topk cross entropy 
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DynamicTopkCEPlusDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):

        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index)
        dice_loss = dice(predict,target)

        topk = DynamicTopKLoss(weight=self.weight,**self.kwargs)
        topk_loss = topk(predict,target)
        
        total_loss = topk_loss + dice_loss

        return total_loss


class CELabelSmoothingPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus cross entropy with label smoothing
    """
    def __init__(self, smoothing=0.0, weight=None, ignore_index=None, **kwargs):
        super(CELabelSmoothingPlusDice, self).__init__()
        self.kwargs = kwargs
        self.smoothing = smoothing
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        # print(predict.size())
        # print(target.size())
        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict,target)

        cels = LabelSmoothing(smoothing=self.smoothing,weight=self.weight)
        cels_loss = cels(predict,target)
        
        total_loss = cels_loss + dice_loss

        return total_loss