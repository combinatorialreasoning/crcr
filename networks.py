import torch.nn as nn
import torch.nn.functional as f
import torch 
import gin


def small_init(layer, sd):
    if isinstance(layer, nn.Linear):
        # Initialize weights with a normal distribution
        nn.init.normal_(layer.weight, mean=0, std=sd)
        # Initialize biases to zero
        nn.init.zeros_(layer.bias)
    
@gin.configurable
class MRNNet(nn.Module):
    def __init__(self, input_size=54, hidden_size=128, repr_dim=64, depth=8, last_sd=None):
        super(MRNNet, self).__init__()
        assert repr_dim % 2 == 0, "In MRNNet the representation dim needs to be divisible by 2"

        self.phi = LNDenseNet(input_size, hidden_size, repr_dim//2, depth, last_sd)
        self.h = LNDenseNet(input_size, hidden_size, repr_dim//2, depth, last_sd)

        for m in self.parameters():
            self.initialize_weights(m)

        # self.initialize_weights
        self.log_lambda = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.eye(repr_dim))

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='silu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        phi1, psi1 = self.phi(x)
        phi2, psi2 = self.h(x)
        
        return torch.concat([phi1, phi2], axis=1), torch.concat([psi1,  psi2], axis=1)

@gin.configurable
class LNConvNet(nn.Module):
    def __init__(self, input_size=144, hidden_size=128, repr_dim=64, depth=8, last_sd=None, baseline=False):
        super(LNConvNet, self).__init__()
        modules = []
        res = []
        lns = []
        lns_res = []
        self.repr_dim = repr_dim
        assert depth % 2 == 0, "We expect the depth to be divisible by 2"
        self.input_layer = nn.Conv2d(in_channels=input_size, 
                                     out_channels=hidden_size,
                                     kernel_size=(3,3),
                                    stride=1,
                                    padding=1)
        self.first_ln = nn.BatchNorm2d(hidden_size)
        for _ in range(depth // 2):
            modules.append(nn.Conv2d(in_channels=hidden_size, 
                                     out_channels=hidden_size,
                                     kernel_size=(3,3),
                                    stride=1,
                                    padding=1))
            lns.append(nn.BatchNorm2d(hidden_size))
        
        for _ in range(depth // 2):
            res.append(nn.Conv2d(in_channels=hidden_size, 
                                     out_channels=hidden_size,
                                     kernel_size=(3,3),
                                    stride=1,
                                    padding=1))
            lns_res.append(nn.BatchNorm2d(hidden_size))

        self.output_layer = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            nn.Linear(
            in_features=hidden_size, 
            out_features=repr_dim              
                                )) #nn.Linear(hidden_size, repr_dim)

        if last_sd is not None:
            small_init(self.output_layer, last_sd)

        self.layers = nn.ModuleList(modules)
        self.res_layers = nn.ModuleList(res)
        self.lns = nn.ModuleList(lns)
        self.lns_res = nn.ModuleList(lns_res)
        self.baseline = baseline


        for m in self.parameters():
            self.initialize_weights(m)

        # self.initialize_weights
        self.log_lambda = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.eye(repr_dim))

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='silu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @gin.configurable
    def forward(self, x):
        with torch.no_grad():
            def apply_along_axis(function, x, axis: int = 0):
                return torch.stack([
                    function(x_i) for x_i in torch.unbind(x, dim=axis)
                ], dim=axis)
            
            def to_one_hot(arr):
                result = torch.eye(7, device=arr.device)[arr.to(int)]
                return result

            def transform(x):
                x = x.reshape((-1, int(x.shape[-1]**0.5), int(x.shape[-1]**0.5)))
                x = to_one_hot(x)
                x = x.transpose(2, 3).transpose(1, 2)

                return x

            if self.baseline:
                if len(x.split(12**2, dim=-1)) != 2:
                    import pdb; pdb.set_trace()
                    
                x, y = x.split(12**2, dim=-1)
                y = transform(y)
                x = transform(x)
                x = torch.cat([x, y], dim=1)
            
            else:
                x = transform(x)

        x = f.relu(self.first_ln(self.input_layer(x)))

        for module, res, ln, ln_res in zip(self.layers, self.res_layers, self.lns, self.lns_res):
            delta = f.relu(ln(module(x)))
            x = ln_res(res(delta)) + x

        psi = self.output_layer(x)
        phi = psi # @ self.A.T
        return phi, psi

    

@gin.configurable
class LNDenseNet(nn.Module):
    def __init__(self, input_size=54, hidden_size=128, repr_dim=64, depth=8, last_sd=None):
        super(LNDenseNet, self).__init__()
        modules = []
        res = []
        lns = []
        lns_res = []
        self.repr_dim = repr_dim
        assert depth % 2 == 0, "We expect the depth to be divisible by 2"
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.first_ln = nn.LayerNorm(hidden_size)
        for _ in range(depth // 2):
            modules.append(nn.Linear(hidden_size, hidden_size))
            lns.append(nn.LayerNorm(hidden_size))
        
        for _ in range(depth // 2):
            res.append(nn.Linear(hidden_size, hidden_size))
            lns_res.append(nn.LayerNorm(hidden_size))

        self.output_layer = nn.Linear(hidden_size, repr_dim)

        if last_sd is not None:
            small_init(self.output_layer, last_sd)

        self.layers = nn.ModuleList(modules)
        self.res_layers = nn.ModuleList(res)
        self.lns = nn.ModuleList(lns)
        self.lns_res = nn.ModuleList(lns_res)



        for m in self.parameters():
            self.initialize_weights(m)

        # self.initialize_weights
        self.log_lambda = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.eye(repr_dim))

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='silu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = f.relu(self.first_ln(self.input_layer(x)))

        for module, res, ln, ln_res in zip(self.layers, self.res_layers, self.lns, self.lns_res):
            delta = f.relu(ln(module(x)))
            x = ln_res(res(delta)) + x

        psi = self.output_layer(x)
        phi = psi
        return phi, psi
  