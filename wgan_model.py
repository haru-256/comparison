import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from utils import Gamma_initializer


class Critic(chainer.Chain):
    """
    Critic

    build Critic.

    Parameters
    ----------------

    ksize: int
        kernel size. 4 or 5

    pad: int
        padding size. if ksize is 4, then pad has to be 1.
        if ksize is 5, then pad has to be 2.

    nobias: boolean
        whether don't apply bias to convolution layer with no BN layer.
    """

    def __init__(self, ksize=4, pad=1, nobias=True):
        super(Critic, self).__init__()
        print("Critic")
        with self.init_scope():
            # initializers
            conv_init = chainer.initializers.Normal(scale=0.02)
            gamma_init = Gamma_initializer(mean=1.0, scale=0.02)
            beta_init = chainer.initializers.Zero()

            # registar layers with variable
            self.c0 = L.Convolution2D(
                in_channels=3,
                out_channels=64,
                ksize=ksize,
                pad=pad,
                stride=2,
                initialW=conv_init,
                nobias=nobias
            )
            self.c1 = L.Convolution2D(
                in_channels=64,
                out_channels=128,
                ksize=ksize,
                pad=pad,
                stride=2,
                initialW=conv_init,
                nobias=True
            )
            self.c2 = L.Convolution2D(
                in_channels=128,
                out_channels=256,
                ksize=ksize,
                pad=pad,
                stride=2,
                initialW=conv_init,
                nobias=True
            )
            self.c3 = L.Convolution2D(
                in_channels=256,
                out_channels=1,
                ksize=ksize,
                pad=0,
                stride=1,
                initialW=conv_init,
                nobias=nobias
            )

            self.bn1 = L.BatchNormalization(
                size=128,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05
            )
            self.bn2 = L.BatchNormalization(
                size=256,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05
            )

    def __call__(self, x):
        """
        Function that computes forward

        Parametors
        ----------------
        x: Variable
           input image data. this shape is (N, C, H, W)
        """
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        y = self.c3(h)

        return y


class Generator(chainer.Chain):
    """Generator

    build Generator model

    Parametors
    ---------------------
    n_hidden: int
       dims of random vector z

    bottom_width: int
       Width when converting the output of the first layer
       to the 4-dimensional tensor

    in_ch: int
       Channel when converting the output of the first layer
       to the 4-dimensional tensor

    nobias: boolean
       whether don't apply bias to convolution layer with no BN layer.

    """

    def __init__(self, n_hidden=100, bottom_width=4, ch=256, ksize=4, pad=1, nobias=True):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width

        with self.init_scope():
            # initializers
            transposed_conv_init = chainer.initializers.Normal(scale=0.02)
            gamma_init = Gamma_initializer(mean=1.0, scale=0.02)
            beta_init = chainer.initializers.Zero()

            self.dc0 = L.Deconvolution2D(
                in_channels=self.n_hidden,
                out_channels=ch,
                ksize=4,
                pad=0,
                stride=1,
                initialW=transposed_conv_init,
                nobias=True)  # (, 256, 4, 4)
            self.dc1 = L.Deconvolution2D(
                in_channels=256,
                out_channels=ch // 2,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=transposed_conv_init,
                nobias=True)  # (, 128, 8, 8)
            self.dc2 = L.Deconvolution2D(
                in_channels=128,
                out_channels=ch // 4,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=transposed_conv_init,
                nobias=True)  # (, 64, 16, 16)
            self.dc3 = L.Deconvolution2D(
                in_channels=64,
                out_channels=3,
                ksize=ksize,
                stride=2,
                pad=pad,
                initialW=transposed_conv_init,
                nobias=nobias)  # (, 3, 64, 64)

            self.bn0 = L.BatchNormalization(
                ch,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05)
            self.bn1 = L.BatchNormalization(
                ch // 2,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05)
            self.bn2 = L.BatchNormalization(
                ch // 4,
                initial_gamma=gamma_init,
                initial_beta=beta_init,
                eps=1e-05)

    def make_hidden(self, batchsize):
        """
        Function that makes z random vector in accordance with the uniform(-1, 1)

        batchsize: int
           batchsize indicate len(z)

        """
        return self.xp.random.normal(0, 1, (batchsize, self.n_hidden, 1, 1))\
            .astype(self.xp.float32)

    def __call__(self, z):
        """
        Function that computs foward

        Parametors
        ----------------
        z: Variable
           random vector drown from a uniform distribution,
           this shape is (N, 100)

        """
        h = F.relu(self.bn0(self.dc0(z)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        x = F.tanh(self.dc3(h))
        return x


if __name__ == "__main__":
    import chainer.computational_graph as c
    from chainer import Variable

    model = Generator()
    img = model(Variable(model.make_hidden(10)))
    # print(img)
    g = c.build_computational_graph(img)
    with open('gen_graph.dot', 'w') as o:
        o.write(g.dump())
    model = Critic()
    logits = model(img)
    # print(img)
    g = c.build_computational_graph(img)
    with open('critic_graph.dot', 'w') as o:
        o.write(g.dump())
