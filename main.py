import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as d_utils
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((64,64)),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
CelebA = torchvision.datasets.ImageFolder(r"path/to/dataset",transform=transform)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self.create_block(100,1024,4,1,0)
        self.block2 = self.create_block(1024,512,4,2,1)
        self.block3 = self.create_block(512,256,4,2,1)
        self.block4 = self.create_block(256,128,4,2,1)
        self.block5 = nn.Sequential(nn.ConvTranspose2d(128,3,4,2,1),nn.BatchNorm2d(3),nn.Tanh())

    def create_block(self,in_f,out_f,kernel,stride,pad):
        deconv = nn.ConvTranspose2d(in_f,out_f,kernel,stride,pad)
        batch_norm = nn.BatchNorm2d(out_f)
        relu = nn.ReLU()
        return nn.Sequential(deconv,batch_norm,relu)

    def forward(self,x):
        x = x.view(-1,100,1,1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self,leak_val):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3,128,4,2,1),nn.LeakyReLU(leak_val))
        self.block2 = self.create_block(128,256,4,2,1,leak_val)
        self.block3 = self.create_block(256,512,4,2,1,leak_val)
        self.block4 = self.create_block(512,1024,4,2,1,leak_val)
        self.block5 = nn.Sequential(nn.Conv2d(1024,1,4,1,0),nn.Sigmoid())
    
    def create_block(self,in_f,out_f,kernel,stride,pad,leak_val):
        conv = nn.Conv2d(in_f,out_f,kernel,stride,pad)
        batch_norm = nn.BatchNorm2d(out_f)
        l_relu = nn.LeakyReLU(leak_val)
        return nn.Sequential(conv,batch_norm,l_relu)
    
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.view(-1,1)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
CelebA = d_utils.DataLoader(CelebA,batch_size,True)

epochs = 500
lr = 0.0002

leak_val = 0.2

discrim = Discriminator(leak_val).to(device)
gen = Generator().to(device)

betas = (0.5,0.999)

discrim_optim = torch.optim.Adam(discrim.parameters(),lr,betas)
gen_optim = torch.optim.Adam(gen.parameters(),lr,betas)

criterion = nn.BCELoss()

noise_dim = 100
fixed_noise = torch.rand((batch_size,noise_dim)).to(device)

writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")

f_rl,r_rl,gen_rl,d_rl = 0,0,0,0
for epoch in range(1,epochs+1):
    for i,(real,_) in enumerate(CelebA):
        batch_size = real.shape[0]
        real = real.to(device)
        noise = torch.rand((batch_size,noise_dim)).to(device)
        
        fake = gen(noise).to(device)
        
        real_pred = discrim(real)
        fake_pred = discrim(fake.detach())
        r_loss,f_loss = criterion(real_pred,torch.ones((batch_size,1)).to(device)),criterion(fake_pred,torch.zeros((batch_size,1)).to(device))
        r_rl += r_loss
        f_rl += f_loss

        d_loss = (r_loss + f_loss) / 2
        d_rl += d_loss
        
        discrim_optim.zero_grad()
        d_loss.backward()
        discrim_optim.step()

        fake_pred = discrim(fake)
        gen_loss = criterion(fake_pred,torch.ones((batch_size,1)).to(device))
        gen_rl += gen_loss

        gen_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()

    if epoch:
        with torch.no_grad():
            gen.eval()
            fake = gen(fixed_noise).to(device)
            div_term = epoch*len(CelebA)
            print(f"d_loss:{d_rl/div_term},r_loss:{r_rl/div_term}\n,f_loss:{f_rl/div_term},gen_loss:{gen_rl/div_term}")
            gen.train()

            torch.save(gen.state_dict(),"generator.pth")
            torch.save(discrim.state_dict(),"discriminator.pth")

            fake = gen(fixed_noise)
            data = real

            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            writer_fake.add_image("CelebA Fake Images", img_grid_fake, global_step=epoch)
            writer_real.add_image("CelebA Real Images", img_grid_real, global_step=epoch)
