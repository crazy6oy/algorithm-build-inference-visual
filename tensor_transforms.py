import torch
from torch.nn import functional


class Pad_Tensor(object):
    def __call__(self, input):
        n, c, h, w = input.shape
        h_pad = max(0, w - h)
        w_pad = max(0, h - w)
        return torch.nn.functional.pad(input, (0, w_pad, 0, h_pad))


class Permute_Tensor(object):
    def __call__(self, input):
        return input.permute(0, 3, 1, 2)


class Resize_Tensor(object):
    def __init__(self, outsize):
        self.outsize = outsize

    def __call__(self, input):
        return torch.nn.functional.interpolate(
            input, (self.outsize, self.outsize), mode="bicubic"
        )


class Normalize_Tensor(object):
    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, input):
        input /= 255.0
        for i in range(len(self.mean)):
            input[:, i] -= self.mean[i]
            input[:, i] /= self.std[i]
        return input


if __name__ == "__main__":
    a = torch.Tensor([[[[1, 2, 3, 4], [3, 4, 6, 7]]]])
    a = a.repeat_interleave(3, 1)
    print(a)
    pad = Pad_Tensor()
    res = Resize_Tensor(8)
    norm = Normalize_Tensor(mean=(100.0, 100.0, 100.0))
    a = pad.__call__(a)
    print(a.shape)
    print(a)
    a = res.__call__(a)
    print(a.shape)
    print(a)
    a = norm.__call__(a)
    print(a.shape)
    print(a)
