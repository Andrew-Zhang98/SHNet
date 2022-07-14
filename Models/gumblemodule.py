import torch
import torch.nn.functional as F
from torch.autograd import Variable


class GumbelSoftmax2D(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbelSoftmax2D, self).__init__()
        self.hard = hard
        self.gpu = False

    def cuda(self):
        self.gpu = True

    def cpu(self):
        self.gpu = False

    def sample_gumbel(self, shape, eps=1e-10):
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return Variable(noise).cuda()
        else:
            return Variable(noise)

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        dim = logits.size(-1)
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, 1)
        return soft_samples

    def gumbel_softmax(self, logits, temperature, hard=False, gumbel=False):
        if gumbel:
            y = self.gumbel_softmax_sample(logits, temperature)
        else:
            y = F.softmax(logits, 1)
        if hard:
            _, max_value_indexes = y.data.max(1, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            y = Variable(y_hard - y.data) + y
        return y

    def forward(self, logits, gumbel=False, temp=1):
        b, c, h, w = logits.size()
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, c)
        logits = self.gumbel_softmax(logits, temperature=1, hard=self.hard, gumbel=gumbel)

        return logits.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
