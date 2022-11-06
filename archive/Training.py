import torch
import torch.nn as nn
import tqdm
from Losses import SSIM, cycle_consistency_loss, real_mse_loss, fake_mse_loss, psnr, acc
import os


class MyFrame():
    def __init__(self, Gxy, Dx, Gyx, Dy, learning_rate, device, beta1, beta2, load_untrained_model=False,
                 evalmode=False):
        self.Gxy = Gxy().to(device)
        self.Gyx = Gyx().to(device)
        self.Dx = Dx().to(device)
        self.Dy = Dy().to(device)
        # self.segmenter = UNet().to(device)
        # self.segmenter.load_state_dict(torch.load('/home/user3/Desktop/data/CT_IITD/Save_models/Unet_rib.pth'))
        self.g = torch.optim.Adam(params=list(self.Gxy.parameters()) + list(self.Gyx.parameters()), lr=learning_rate,
                                  betas=[beta1, beta2])
        self.dx = torch.optim.Adam(params=list(self.Dx.parameters()), lr=learning_rate, betas=[beta1, beta2])
        self.dy = torch.optim.Adam(params=list(self.Dy.parameters()), lr=learning_rate, betas=[beta1, beta2])
        self.lr = learning_rate
        self.imX = 0
        self.imY = 0

    def set_input(self, i1, i2, mX, mY):
        self.imX = i1
        self.imY = i2
        self.mapX = mX
        self.mapY = mY

    def dump(self):
        a, b, c = self.Gxy.forward(self.imX, self.mapX)
        return a

    def optimize(self, backprop=True):
        # Discriminator train:
        self.dx.zero_grad()
        # atmap = self.segmenter.forward(self.imX)
        outX = self.Dx.forward(self.imX)
        a, b, c = self.Gyx.forward(self.imY, self.mapY)
        a1, b1, c1 = self.Gxy.forward(self.imX, self.mapX)
        outY = self.Dy.forward(a)
        # outY = self.Dy.forward(self.Gyx.forward(self.imY))
        Ldx_real = real_mse_loss(outX)
        Ldx_fake = fake_mse_loss(outY)
        dx_loss = Ldx_real + Ldx_fake
        if backprop:
            dx_loss.backward()
            self.dx.step()

        self.dy.zero_grad()
        # atmap = self.segmenter.forward(self.imX)
        outY = self.Dy.forward(self.imY)
        outX = self.Dy.forward(a1)
        # outX = self.Dy.forward(self.Gxy.forward(self.imX))
        Ldy_real = real_mse_loss(outY)
        Ldy_fake = fake_mse_loss(outX)
        dy_loss = Ldy_real + Ldy_fake
        if backprop:
            dy_loss.backward()
            self.dy.step()

        # Generator train:
        self.g.zero_grad()
        # atmap = self.segmenter.forward(self.imX)
        gen1, gen11, gen12 = self.Gyx.forward(self.imY, self.mapY)
        imX1 = nn.MaxPool2d(2).forward(self.imX)
        imX2 = nn.MaxPool2d(2).forward(imX1)

        Ly1 = real_mse_loss(self.Dx(gen1))
        # Ls1 = cycle_consistency_loss(self.imX,gen1,10) + cycle_consistency_loss(imX1,gen11,10)
        # + cycle_consistency_loss(imX2,gen12,10)
        genY, genY1, genY2 = self.Gxy.forward(gen1, self.mapX)
        imY1 = nn.MaxPool2d(2).forward(self.imY)
        imY2 = nn.MaxPool2d(2).forward(imY1)

        Ly2 = cycle_consistency_loss(self.imY, genY, 10) + cycle_consistency_loss(imY1, gen11,
                                                                                  10) + cycle_consistency_loss(imY2,
                                                                                                               gen12,
                                                                                                               10)

        gen2, gen21, gen22 = self.Gxy.forward(self.imX, self.mapX)
        Lx1 = real_mse_loss(self.Dy(gen2))
        # Ls2 = cycle_consistency_loss(self.imY,gen2,10)
        genX, genX1, genX2 = self.Gyx.forward(gen2, self.mapY)
        Lx2 = cycle_consistency_loss(self.imX, genX, 10) + cycle_consistency_loss(imX1, genX1,
                                                                                  10) + cycle_consistency_loss(imX2,
                                                                                                               genX2,
                                                                                                               10)

        gLoss = Lx1 + Lx2 + Ly1 + Ly2  # + Ls1 + Ls2
        if backprop:
            gLoss.backward()
            self.g.step()
        return gLoss + dx_loss + dy_loss

    def save(self):
        torch.save(self.Gxy.state_dict(), r'E:\Projects\Deep learning\Rib Suppression\Log\Models\GXY_guided.pth')
        torch.save(self.Gyx.state_dict(), r'E:\Projects\Deep learning\Rib Suppression\Log\Models\GYX_guided.pth')
        torch.save(self.Dx.state_dict(), r'E:\Projects\Deep learning\Rib Suppression\Log\Models\DX_guided.pth')
        torch.save(self.Dy.state_dict(), r'E:\Projects\Deep learning\Rib Suppression\Log\Models\DY_guided.pth')

    def load(self, initial=True):
        self.Gxy.load_state_dict(torch.load(r'E:\Projects\Deep learning\Rib Suppression\Log\Models\GXY_guided.pth'))
        self.Gyx.load_state_dict(torch.load(r'E:\Projects\Deep learning\Rib Suppression\Log\Models\GYX_guided.pth'))
        self.Dx.load_state_dict(torch.load(r'E:\Projects\Deep learning\Rib Suppression\Log\Models\DX_guided.pth'))
        self.Dy.load_state_dict(torch.load(r'E:\Projects\Deep learning\Rib Suppression\Log\Models\DY_guided.pth'))

    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.lr / new_lr
        for param_group in self.g.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.dx.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.dy.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.lr, new_lr))
        print('update learning rate: %f -> %f' % (self.lr, new_lr))
        self.lr = new_lr

root_path = r'E:\Projects\Deep learning\Rib Suppression\Dataset\JSRT'#'/home/user3/Desktop/data/CT'
input_size = (3,256,256) #for kaggle 448
batch_size = 1
learning_rate = 0.000001
epochs = 1000
beta1= 0.5
beta2= 0.999

INITAL_EPOCH_LOSS = 10000
NUM_EARLY_STOP = 20
NUM_UPDATE_LR = 2

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#device = torch.device('cpu')

def trainer_chan(epoch, epochs, train_loader, train_dataset, solver):
    keep_training = True
    no_optim = 0
    train_epoch_best_loss = INITAL_EPOCH_LOSS
    print('Epoch {}/{}'.format(epoch, epochs))
    train_epoch_loss = 0
    train_epoch_acc = 0
    train_epoch_psnr = 0
    train_epoch_ssim = 0
    p_loss = 10
    test_index = 8

    # index = 0
    length = len(train_loader)
    iterator = enumerate(train_loader)#, total=length, leave=False, desc=f'Epoch {epoch}/{epochs}')
    ssim = SSIM()
    for index, (imX, imY, mapX, mapY) in iterator:
        print('Progress:', str((index + 1) * 100 / len(train_dataset)))
        imX = imX.to(device)
        imY = imY.to(device)
        mapX = mapX.to(device)
        mapY = mapY.to(device)

        solver.set_input(imX, imY, mapX, mapY)
        train_loss = solver.optimize()
        imYn = solver.dump()
        accuracy = acc(imYn, imY)
        psnr_met = psnr(imYn, imY)
        # ssim_met = ssim.forward(imYn, imY)
        train_loss = train_loss.detach().cpu().numpy()
        train_acc = accuracy.detach().cpu().numpy()
        train_psnr = psnr_met.detach().cpu().numpy()
        train_ssim = ssim.forward(imYn, imY).detach().cpu().numpy()
        # train_ssim = ssim_met.detach().cpu().numpy()

        train_epoch_loss += train_loss
        train_epoch_acc += train_acc
        train_epoch_psnr += train_psnr
        train_epoch_ssim += train_ssim
        # train_epoch_ssim += train_ssim
        # print(index, end = ' ')

        if index == test_index:
            test_imX = imX
            test_imY = imY
            test_pred = imYn

    train_epoch_loss = train_epoch_loss / len(train_dataset)
    train_epoch_acc = train_epoch_acc / len(train_dataset)
    train_epoch_ssim = train_epoch_ssim / len(train_dataset)
    train_epoch_psnr = train_epoch_psnr / len(train_dataset)
    print('train_loss:', train_epoch_loss)
    print('train_acc:', train_epoch_acc)
    print('train_psnr:', train_epoch_psnr)
    print('train_ssim:', train_epoch_ssim)

    print('---------------------------------------------')
    return train_epoch_loss, train_epoch_psnr, train_epoch_ssim, keep_training


def tester_chan(test_loader, test_dataset, test_best_loss, solver, num, smart=True):
    test_acc = 0
    test_loss = 0
    test_psnr = 0
    test_ssim = 0
    ssim = SSIM()
    with torch.no_grad():
        for index, (imX, imY, mapX, mapY) in enumerate(test_loader):
            print('Progress:', str((index + 1) * 100 / len(test_dataset)))
            imX = imX.to(device)
            imY = imY.to(device)
            mapX = mapX.to(device)
            mapY = mapY.to(device)
            solver.set_input(imX, imY, mapX, mapY)
            loss = solver.optimize(backprop=False)
            imYn = solver.dump()
            accuracy = acc(imYn, imY)
            psnr_met = psnr(imYn, imY)

            loss = loss.detach().cpu().numpy()
            a = accuracy.detach().cpu().numpy()
            p = psnr_met.detach().cpu().numpy()
            s = ssim.forward(imYn, imY).detach().cpu().numpy()

            test_loss += loss
            test_acc += a
            test_psnr += p
            test_ssim += s

        test_acc = test_acc / (len(test_dataset))
        test_loss = test_loss / (len(test_dataset))
        test_psnr = test_psnr / (len(test_dataset))
        test_ssim = test_ssim / (len(test_dataset))

        print('Test Accuracy : ', test_acc)
        print('Test loss : ', test_loss)
        print('Test SSIM : ', test_ssim)
        print('Test PSNR : ', test_psnr)
        if smart == True:
            if test_ssim <= test_best_loss:
                num += 1
                print('Not Saving model: Best Test SSIM = ', test_best_loss, ' Current test SSIM = ', test_ssim)
            else:
                num = 0
                print('Saving model...')
                solver.save()
                test_best_loss = test_ssim

            if num >= 5:
                num = 0
                print('Loading model...')
                solver.load()
                solver.update_lr(2, True)
                test_best_loss = 0
        print('-------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------')
        return test_loss, test_psnr, test_ssim, test_best_loss, num