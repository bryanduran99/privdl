from pytorch_msssim import ssim


def neg_ssim(img0s, img1s):
    def norm(data):
        Min, Max = data.min(), data.max()
        return (data - Min) / (Max - Min)
    img0s, img1s = map(norm, [img0s, img1s])
    return -ssim(img0s, img1s, data_range=1.)
