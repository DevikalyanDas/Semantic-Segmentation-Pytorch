import torch

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

class IoU(object):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None,classes=19):
        self.eps = eps
        self.threshold = threshold
        self.activation = torch.nn.Softmax(dim=1)
        self.classes = classes
    def __call__(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        y_pr = _threshold(y_pr, threshold=self.threshold)
        y_gt = make_one_hot(y_gt.long().unsqueeze(dim=1),self.classes)
        intersection = torch.sum(y_gt * y_pr)
        union = torch.sum(y_gt) + torch.sum(y_pr) - intersection + self.eps
        score = (intersection + self.eps) / union
        return score.item()

class Fscore(object):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None,classes=19):
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = torch.nn.Softmax(dim=1)
        self.classes = classes

    def __call__(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        y_pr = _threshold(y_pr, threshold=self.threshold)
        y_gt = make_one_hot(y_gt.long().unsqueeze(dim=1),self.classes)

        tp = torch.sum(y_gt * y_pr)
        fp = torch.sum(y_pr) - tp
        fn = torch.sum(y_gt) - tp

        score = ((1 +self.beta ** 2) * tp + self.eps) \
                / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.eps)

        return score.item()        

class Accuracy(object):

    def __init__(self, threshold=0.5, activation=None,classes=19):
        self.threshold = threshold
        self.activation = torch.nn.Softmax(dim=1)
        self.classes = classes
    def __call__(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        y_pr = _threshold(y_pr, threshold=self.threshold)
        y_gt = make_one_hot(y_gt.long().unsqueeze(dim=1),self.classes)

        tp = torch.sum(y_gt == y_pr, dtype=y_pr.dtype)
        score = tp / y_gt.view(-1).shape[0]
        return score.item()

class Sensitivity(object):
# Sensitivity
    def __init__(self, eps=1e-7,activation=None, threshold=0.5,classes=19):
        self.eps = eps
        self.threshold = threshold
        self.activation = torch.nn.Softmax(dim=1)
        self.classes = classes
    def __call__(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        y_pr = _threshold(y_pr, threshold=self.threshold)
        y_gt = make_one_hot(y_gt.long().unsqueeze(dim=1),self.classes)
        
        tp = torch.sum(y_gt * y_pr)
        fn = torch.sum(y_gt) - tp
        score = (tp + self.eps) / (tp + fn + self.eps)

        return score.item()


class Specificity(object):

    def __init__(self, eps=1e-7,activation=None, threshold=0.5,classes=19):
        self.eps = eps
        self.threshold = threshold
        self.activation = torch.nn.Softmax(dim=1)
        self.classes = classes

    def __call__(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        y_pr = _threshold(y_pr, threshold=self.threshold)
        y_gt = make_one_hot(y_gt.long().unsqueeze(dim=1),self.classes)     

        tn = torch.sum(y_gt == y_pr, dtype=y_pr.dtype)-torch.sum(y_gt * y_pr)
        tp = torch.sum(y_gt * y_pr)
        fp = torch.sum(y_pr) - tp

        score = (tn + self.eps) / (tn + fp + self.eps)

        return score.item()