from torch.utils.data import Dataset, DataLoader
from Training import root_path, batch_size, device, trainer_chan, tester_chan, MyFrame, epochs, learning_rate, beta1, beta2
from Losses import SSIM, Metric_manager
from Load_data import Eye_Dataset
from Utils import convert
from Models import Generator_v4, Discriminator

dataset = Eye_Dataset(root_path,'train')
train_dataset = Eye_Dataset(root_path,'train')
test_dataset = Eye_Dataset(root_path,'test')

train_dataset.images = []
train_dataset.labels = []
train_dataset.mapX = []
train_dataset.mapY = []

test_dataset.images = []
test_dataset.labels = []
test_dataset.mapX = []
test_dataset.mapY = []

train_dataset.images += dataset.images[0:392]
train_dataset.labels += dataset.labels[0:392]
test_dataset.images += dataset.images[392:]
test_dataset.labels += dataset.labels[392:]
train_dataset.mapX += dataset.mapX[0:392]
test_dataset.mapX += dataset.mapX[392:]
train_dataset.mapY += dataset.mapY[0:392]
test_dataset.mapY += dataset.mapY[392:]


train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

print('Training device: ', device)

solver = MyFrame(Generator_v4, Discriminator, Generator_v4, Discriminator, learning_rate, device, beta1, beta2)
#solver.load()
train_met = Metric_manager()
test_met = Metric_manager()
prev_test_loss = 0.0
num = 0

for epoch in range(1, epochs + 1):
    l, p, s, k = trainer_chan(epoch, epochs, train_loader, train_dataset, solver)
    if epoch != 1:
        train_met.load(r'E:\Projects\Deep learning\Rib Suppression\Log\trainmets.npy')
    train_met.register(l, p, s)
    train_met.save(r'E:\Projects\Deep learning\Rib Suppression\Log\trainmets.npy')

    l, p, s, pt_loss, num = tester_chan(test_loader, test_dataset, prev_test_loss, solver, num)
    if epoch != 1:
        test_met.load(r'E:\Projects\Deep learning\Rib Suppression\Log\testmets.npy')
    test_met.register(l, p, s)
    test_met.save(r'E:\Projects\Deep learning\Rib Suppression\Log\testmets.npy')

    if pt_loss >= prev_test_loss: prev_test_loss = pt_loss
    # tr_Loss += [l]

    # solver.save()
    if k:
        continue
    else:
        break