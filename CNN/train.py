import CNN.model as model
import torchvision
import torch as t
from torch.autograd import Variable
import CNN.dataloader
import numpy

net = model.Generator()
optimizer = t.optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
criterion = t.nn.MSELoss()
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((60, 180)),
                                            torchvision.transforms.ToTensor(),
                                            #torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                            ])
path='/home/wang/桌面/top/testData/'
dataset = CNN.dataloader.myDataset(path, transform, CNN.dataloader.default_loader)
train_loader = t.utils.data.DataLoader(dataset=dataset,
                                       batch_size=1,
                                       shuffle=True)
def main():
    for epoch in range(1, 50):
        train(epoch)
    t.save(net, './model.pkl')

def train(epoch):
    i=0
    for img, img_path, volfrac, penal, rmin in train_loader:
        img= Variable(img)
        optimizer.zero_grad()
        p_volfrac, p_penal, p_rmin = net(img)
        loss = criterion(p_volfrac, t.tensor(numpy.array([[float(volfrac[0])]])).float())+\
               criterion(p_penal, t.tensor(numpy.array([[float(penal[0])]])).float())+\
               criterion(p_rmin, t.tensor(numpy.array([[float(rmin[0])]])).float())
        loss.backward()
        optimizer.step()
        i += 1
        if i % 20 == 0:
            print('Train Epoch: {} [{}]\tLoss: {:.6f}'.format(
                epoch, i, loss.item()))

# def test():
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         data, target = Variable(data), Variable(target)
#         output = model(data)
#         test_loss += F.nll_loss(output, target, size_average=False).item()
#         pred = output.data.max(1, keepdim=True)[1]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#         test_loss /= len(test_loader.dataset)
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             test_loss, correct, len(test_loader.dataset),
#             100. * correct / len(test_loader.dataset)))




if __name__ == '__main__':
    main()