import torch.nn as nn 
import torch 
from torch.nn import init
import itertools
import random
from torch.autograd import Variable

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    
def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, gain=init_gain)
    return net


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True,k=4, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, k, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print('in', x.shape)
        # print('out', self.model(x).shape)
        return self.model(x)



class UNetMid(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride, dropout=0.0):
        super(UNetMid, self).__init__()
        layers = [
            nn.Conv3d(in_size, out_size, kernel, stride, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)


    def forward(self, x, skip_input):
        # print(x.shape)
        x = torch.cat((x, skip_input), 1)
        x = self.model(x)
        x =  nn.functional.pad(x, (1,0,1,0,1,0))

        return x

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, k=4):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, k, 2, 1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # print('new')
        # print(x.shape)
        # print(skip_input.shape)
        x = self.model(x)
        # print(x.shape)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, k=(4,3,4))
        self.mid1 = UNetMid(1024, 512, dropout=0.2, kernel = 4, stride = 1)
        self.mid2 = UNetMid(1024, 512, dropout=0.2, kernel = 4, stride = 1)
        self.mid3 = UNetMid(1024, 512, dropout=0.2, kernel = 4, stride = 1)
        self.mid4 = UNetMid(1024, 256, dropout=0.2, kernel = 4, stride = 1)
        self.up1 = UNetUp(256, 256, k=(4,3,4))
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        # self.us =   nn.Upsample(scale_factor=2)

        self.final = nn.Sequential(
            # nn.Conv3d(128, out_channels, 4, padding=1),
            # nn.Tanh(),
            nn.ConvTranspose3d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        m1 = self.mid1(d4, d4)
        m2 = self.mid2(m1, m1)
        m3 = self.mid3(m2, m2)
        m4 = self.mid4(m3, m3)
        u1 = self.up1(m4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        # u7 = self.up7(u6, d1)
        # u7 = self.us(u7)
        # u7 = nn.functional.pad(u7, pad=(1,0,1,0,1,0))
        # # print(self.final(u7).shape)
        return self.final(u3)
        #return m4


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()

        use_bias = True

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # nn.ZeroPad3d((1, 0, 1, 0)),
        )
        self.final = nn.Conv3d(512, 1, 4, padding=1, bias=False)

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        intermediate = self.model(img)
        pad = nn.functional.pad(intermediate, pad=(1,0,1,0,1,0))
        return self.final(pad)

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class CycleGAN(nn.Module):
    def __init__(self, lambda_A=10, lambda_B=10, lambda_identity=1, lambda_pix=0.5, lambda_co_A=0, lambda_co_B=0, lr=1e-5, beta1=0.5, pool_size=32):
        super(CycleGAN, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        print('Device:', self.device)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'cor_coe_GA', 'D_B', 'G_B', 'cycle_B', 'cor_coe_GB']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        self.lambda_A=lambda_A
        self.lambda_B=lambda_B
        self.lambda_identity=lambda_identity
        self.lambda_co_A=lambda_co_A
        self.lambda_co_B=lambda_co_B

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = init_net(GeneratorUNet()).to(self.device)
        self.netG_B = init_net(GeneratorUNet()).to(self.device)

        self.netD_A = init_net(PixelDiscriminator()).to(self.device)
        self.netD_B = init_net(PixelDiscriminator()).to(self.device)
        
        # define loss functions
        self.criterionGAN = GANLoss().to(self.device)
        self.criterionCycle = torch.nn.MSELoss()
        self.criterionIdt = torch.nn.MSELoss()
        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            lr=lr, betas=(beta1, 0.999))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def set_input(self, data):
        Tensor = torch.cuda.FloatTensor if self.device == torch.device("cuda") else torch.FloatTensor

        self.real_A = data[0].to(self.device)
        self.real_A = Variable(self.real_A.type(Tensor))
        self.real_B = data[1].to(self.device)
        self.real_B = Variable(self.real_B.type(Tensor))

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.lambda_identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B
        '''
        lambda_coA & lambda_coB
        '''
        lambda_co_A = self.lambda_co_A
        lambda_co_B = self.lambda_co_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.loss_idt_A = self.criterionIdt(self.fake_B, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.loss_idt_B = self.criterionIdt(self.fake_A, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = dict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def optimize_parameters(self):
        # forward
        self.forward() # Generates fake B, fake A and re-change fake B and fake A pour generer reconstructed A from fake B and reconstructed B from fake A
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G() # Updates the loss of the generators 
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A() # Updates the loss of the discriminator that tries to distinguish between reconstructed A and real A 
        self.backward_D_B() # Updates the loss of the discriminator that tries to distinguish between reconstructed B and real B 
        self.optimizer_D.step()