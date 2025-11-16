import torch
import ssim

im = torch.randint(0, 255, (5, 3, 256, 256), dtype=torch.float, device='cuda')
img1 = im / 255
img2 = img1 * 0.5

losser = ssim.SSIM(data_range=1., channel=3).cuda()
loss = losser(img1, img2).mean()

losser2 = ssim.MS_SSIM(data_range=1., channel=3).cuda()
loss2 = losser2(img1, img2).mean()

print(loss.item())
print(loss2.item())