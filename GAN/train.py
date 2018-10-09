import tqdm
import torch as t
from torch.autograd import Variable
import torchvision as tv
import GAN.mymodel
import GAN.config
import os
from torchnet.meter import AverageValueMeter
import CNN.dataloader
from PIL import Image

opt=GAN.config.Config()

netg=GAN.mymodel.Generator()
netd=GAN.mymodel.Discriminator()

def default_loader(path):
    return Image.open(path)

transform = tv.transforms.Compose([tv.transforms.Resize((60, 180)),
                                    tv.transforms.ToTensor(),
                                   ])

path='/home/wang/topo/trainData/'
dataset = CNN.dataloader.myDataset(path, transform, default_loader)
data_loader = t.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=3)


# transforms = tv.transforms.Compose([
#     tv.transforms.Resize(opt.image_size),
#     tv.transforms.CenterCrop(opt.image_size),
#     tv.transforms.ToTensor(),
#     tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
# dataloader = t.utils.data.DataLoader(dataset,
#                                      batch_size=opt.batch_size,
#                                      shuffle=True,
#                                      num_workers=opt.num_workers,
#                                      drop_last=True
#                                      )

optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
criterion = t.nn.BCELoss()
true_labels = t.ones(opt.batch_size)
fake_labels = t.zeros(opt.batch_size)

fix_noises = t.randn(opt.batch_size, opt.nz)
fix_noises = t.autograd.Variable(fix_noises, volatile=True)

noises = t.randn(opt.batch_size, opt.nz)
noises = t.autograd.Variable(noises, volatile=True)

errord_meter = AverageValueMeter()
errorg_meter = AverageValueMeter()

epochs = range(opt.max_epoch)
for epoch in iter(epochs):
    for ii,(image, label, k, j, l) in tqdm.tqdm(enumerate(data_loader)):
        real_img = Variable(image)
        if (ii + 1) % opt.d_every == 0:
            optimizer_d.zero_grad()
            output = netd(real_img)
            error_d_real = criterion(output, true_labels)
            error_d_real.backward()
            noises.data.copy_(t.randn(opt.batch_size, 128))
            fake_img=netg(noises).detach()
            fake_output=netd(fake_img)
            error_d_fake=criterion(fake_output, fake_labels)
            error_d_fake.backward()
            optimizer_d.step()

        if (ii + 1) % opt.g_every == 0:
            optimizer_g.zero_grad()
            fake_img = netg(noises)
            fake_output=netd(fake_img)
            error_g=criterion(fake_output,true_labels)
            error_g.backward()
            optimizer_g.step()
        if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
            ## 可视化
            fix_fake_imgs = netg(fix_noises)
    if (epoch+1) % opt.save_every == 0:
        # 保存模型、图片
        fix_fake_imgs = netg(fix_noises)
        tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, epoch), normalize=True,
                                range=(-1, 1))
        t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
        t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
        errord_meter.reset()
        errorg_meter.reset()