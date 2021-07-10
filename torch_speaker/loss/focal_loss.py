import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import accuracy

class focal_loss(nn.Module):
    def __init__(self, embedding_dim, num_classes, gamma=0.01, **kwargs):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.criertion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x, label):
        assert len(x.shape) == 3
        label = label.repeat_interleave(x.shape[1])
        x = x.reshape(-1, self.embedding_dim)
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.embedding_dim

        x = self.fc(x)
 
        logp = self.criertion(x, label)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        prec1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]
        return loss.mean(), prec1

if __name__ == "__main__":
    model = focal_loss(10, 100)
    data = torch.ones((2, 1, 10))
    label = torch.tensor([0, 1])
    loss, acc = model(data, label)

    print(data.shape)
    print(loss)
    print(acc)

