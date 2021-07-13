import torch
import torch.nn as nn
import torch.nn.functional as F

from .softmax import softmax
from .amsoftmax import amsoftmax

class center_loss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        embedding_dim (int): embedding dimension.
        num_classes (int): number of classes.
    """
    def __init__(self, embedding_dim, num_classes, **kwargs):
        super(center_loss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.centers_table =  nn.Embedding(num_classes, embedding_dim)

    def forward(self, x, label):
        """
        Args:
            x: speaker embedding with shape (batch_size, num_shot, feat_dim).
            label: ground truth label with shape (batch_size).
        """
        assert len(x.shape) == 3
        label = label.repeat_interleave(x.shape[1])
        x = x.reshape(-1, self.embedding_dim)
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.embedding_dim

        centers = self.centers_table(label)
        loss = F.cosine_similarity(x, centers, dim=1, eps=1e-8)

        return loss.mean()


class center_softmax(nn.Module):
    """Center softmax.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        embedding_dim (int): embedding dimension.
        num_classes (int): number of classes.
    """
    def __init__(self, embedding_dim, num_classes, weight=0.05, **kwargs):
        super(center_softmax, self).__init__()
        self.weight = weight
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.center_loss = center_loss(embedding_dim, num_classes, **kwargs)
        self.softmax_loss = softmax(embedding_dim, num_classes, **kwargs)

        print('init center softmax with lambda {:.2f}'.format(weight))
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))

    def forward(self, x, label):
        """
        Args:
            x: speaker embedding with shape (batch_size, num_shot, feat_dim).
            label: ground truth label with shape (batch_size).
        """
        loss_c = self.center_loss(x, label)
        loss_s, acc1 = self.softmax_loss(x, label)
        loss = self.weight * loss_c + (1.0-self.weight) * loss_s

        return loss, acc1


class center_am(nn.Module):
    """Center softmax.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        embedding_dim (int): embedding dimension.
        num_classes (int): number of classes.
    """
    def __init__(self, embedding_dim, num_classes, weight=0.05, **kwargs):
        super(center_am, self).__init__()
        self.weight = weight
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.center_loss = center_loss(embedding_dim, num_classes, **kwargs)
        self.amsoftmax = amsoftmax(embedding_dim, num_classes, **kwargs)

        print('init center AM-softmax with lambda {:.2f}'.format(weight))
        print('Embedding dim is {}, number of speakers is {}'.format(embedding_dim, num_classes))

    def forward(self, x, label):
        """
        Args:
            x: speaker embedding with shape (batch_size, num_shot, feat_dim).
            label: ground truth label with shape (batch_size).
        """
        loss_c = self.center_loss(x, label)
        loss_s, acc1 = self.amsoftmax(x, label)
        loss = self.weight * loss_c + (1.0-self.weight) * loss_s

        return loss, acc1



if __name__ == "__main__":
    model = center_loss(10, 100)
    data = torch.randn((2, 1, 10))
    label = torch.tensor([0, 1])

    loss, acc = model(data, label)

    print(data.shape)
    print(loss)
    print(acc)

