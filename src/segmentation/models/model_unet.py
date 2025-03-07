# adopted from https://github.com/astokholm/AI4ArcticSeaIceChallenge/blob/main/unet.py
import torch

class UNet(torch.nn.Module):
    """PyTorch U-Net Class. Uses unet_parts."""

    def __init__(self, options ,conv_filters , out_channels=3 , in_channels=3  ):
        super().__init__()
        #self.train_variables = options['sar_variables'] + options['amsr_env_variables']
        #self.unet_conv_filters = [16, 32, 64, 64] # from the provided code for winner team is different
        self.unet_conv_filters =  conv_filters # winners team config
        

        self.input_block = DoubleConv(options, input_n=in_channels, output_n=self.unet_conv_filters[0])

        self.encoder_blocks = torch.nn.ModuleList()
        self.decoder_blocks = torch.nn.ModuleList()


        for en_b in range(1, len(self.unet_conv_filters)):
            self.encoder_blocks.append(Encoderblock(options=options,
                                 input_n=self.unet_conv_filters[en_b - 1],
                                 output_n=self.unet_conv_filters[en_b]))  # only used to contract input patch.
            

        self.bridge = Encoderblock(options, input_n=self.unet_conv_filters[-1], output_n=self.unet_conv_filters[-1])

        self.decoder_blocks.append( Decoderblock(options=options, input_n=self.unet_conv_filters[-1], output_n=self.unet_conv_filters[-1]))

        for expand_n in range(len(self.unet_conv_filters), 1, -1):
            self.decoder_blocks.append(Decoderblock(options=options,
                                                     input_n=self.unet_conv_filters[expand_n - 1],
                                                     output_n=self.unet_conv_filters[expand_n - 2]))

        self.feature_out = OutConv(in_channels=self.unet_conv_filters[0], out_channels= out_channels)

    def forward(self, x):
        """Forward model pass."""
        x_contract = [self.input_block(x)]
        for contract_block in self.encoder_blocks:
            x_contract.append(contract_block(x_contract[-1]))
        x_expand = self.bridge(x_contract[-1])
        up_idx = len(x_contract)
        for expand_block in self.decoder_blocks:
            x_expand = expand_block(x_expand, x_contract[up_idx - 1])
            up_idx -= 1    

        x = self.feature_out(x_expand)
        return x
 

class DoubleConv(torch.nn.Module):
    """Class to perform a double conv layer in the U-NET architecture. Used in unet_model.py."""

    def __init__(self, options, input_n, output_n):
        super(DoubleConv, self).__init__()
        

        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_n,
                      out_channels=output_n,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      padding_mode='zeros',
                      bias=False),
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_n,
                      out_channels=output_n,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      padding_mode='zeros',
                      bias=False),
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU()
        )

    def forward(self, x):
        """Pass x through the double conv layer."""
        x = self.double_conv(x)

        return x



class Encoderblock(torch.nn.Module):
    """Class to perform downward pass in the U-Net."""

    def __init__(self, options, input_n, output_n):
        super(Encoderblock, self).__init__()

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.double_conv = DoubleConv(options, input_n, output_n)

    def forward(self, x):
        """Pass x through the downward layer."""
        
        x = self.max_pool(x)
        x = self.double_conv(x)

        return x

class Decoderblock(torch.nn.Module):
    """Class to perform upward layer in the U-Net."""

    def __init__(self, options, input_n, output_n):
        super(Decoderblock, self).__init__()

        self.padding_style = 'zeros'
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.double_conv = DoubleConv(options, input_n=input_n + output_n, output_n=output_n)

    def forward(self, x, x_skip):
        """Pass x through the upward layer and concatenate with opposite layer."""
        x = self.upsample(x)

        # Insure that x and skip H and W dimensions match.
        x = expand_padding(x, x_skip, padding_style=self.padding_style)
        x = torch.cat([x, x_skip], dim=1)

        return self.double_conv(x)
    
def expand_padding(x, x_contract, padding_style: str = 'constant'):
    """
    Insure that x and x_skip H and W dimensions match.
    Parameters
    ----------
    x :
        Image tensor of shape (batch size, channels, height, width). Expanding path.
    x_contract :
        Image tensor of shape (batch size, channels, height, width) Contracting path.
        or torch.Size. Contracting path.
    padding_style : str
        Type of padding.

    Returns
    -------
    x : ndtensor
        Padded expanding path.
    """
    # Check whether x_contract is tensor or shape.
    if type(x_contract) == type(x):
        x_contract = x_contract.size()

    # Calculate necessary padding to retain patch size.
    pad_y = x_contract[2] - x.size()[2]
    pad_x = x_contract[3] - x.size()[3]

    if padding_style == 'zeros':
        padding_style = 'constant'

    x = torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_style)

    return x

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv(x)
