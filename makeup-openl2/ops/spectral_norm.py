import mindspore as ms
from mindspore import ops
from mindspore.common.initializer import Normal

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(object):
    def __init__(self):
        self.name = "weight"
        #print(self.name)
        self.power_iterations = 1

    def compute_weight(self, module):
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        w = getattr(module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(ops.mv(ops.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(ops.mv(w.view(height,-1).data, v.data))
        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        #TODO dot???
        # sigma = u.dot(w.view(height, -1).mv(v))
        # sigma = u.tensor_dot(w.view(height, -1).mv(v),axes=1)
        sigma = ops.tensor_dot(u,w.view(height, -1).mv(v),axes=1)
        return w / sigma.expand_as(w)

    @staticmethod
    def apply(module):
        name = "weight"
        fn = SpectralNorm()

        try:
            u = getattr(module, name + "_u")
            v = getattr(module, name + "_v")
            w = getattr(module, name + "_bar")
        except AttributeError:
            #TODO .data
            w = getattr(module, name)
            height = w.data.shape[0]
            width = w.view(height, -1).data.shape[1]
            #TODO new,normal_
            u = ms.Parameter(ms.Tensor(w.data.new(height),ms.float32,init=Normal()), name=name + "_u", requires_grad=False)
            # v = ms.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
            v = ms.Parameter(ms.Tensor(w.data.new(width),ms.float32,Normal()), name=name + "_v", requires_grad=False)
            w_bar = ms.Parameter(w.data, name=name + "_bar")

            #del module._parameters[name]

            #TODO register_parameter
            # module.register_parameter(name + "_u", u)
            # module.register_parameter(name + "_v", v)
            # module.register_parameter(name + "_bar", w_bar)

        # remove w from parameter list
        #TODO _params or _params_list ?
        del module._params[name]
        # del module._parameters[name]

        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        # del module._parameters[self.name + '_u']
        # del module._parameters[self.name + '_v']
        # del module._parameters[self.name + '_bar']
        #TODO 与上面一样
        del module._params[self.name + '_u']
        del module._params[self.name + '_v']
        del module._params[self.name + '_bar']
        ms.Parameter((weight.data), name=self.name)
        # module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

def spectral_norm(module):
    SpectralNorm.apply(module)
    return module

def remove_spectral_norm(module):
    name = 'weight'
    for k, hook in module._forward_pre_hook.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hook[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}"
                     .format(name, module))