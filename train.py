import chainer
from chainer import training
from chainer.training import extensions
# from utils import out_generated_image
from wgan_model import Critic, Generator
from updater import WGANUpdater
import argparse
import pathlib
# import matplotlib as mpl
# mpl.use("Agg")
# import matplotlib.pyplot as plt
# plt.style.use("ggplot")
import datetime

if __name__ == '__main__':
    since1 = datetime.datetime.now()
    # パーサーを作る
    parser = argparse.ArgumentParser(
        prog='train',  # プログラム名
        usage='python trtain.py',  # プログラムの利用方法
        description='description',  # 引数のヘルプの前に表示
        epilog='end',  # 引数のヘルプの後で表示
        add_help=True,  # -h/–help オプションの追加
    )

    # 引数の追加
    parser.add_argument('-s', '--seed', help='seed',
                        type=int, required=True)
    parser.add_argument('-n', '--number', help='the number of experiments.',
                        type=int, required=True)
    parser.add_argument('--hidden', help='the number of codes of Generator.',
                        type=int, default=100)
    parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 100',
                        type=int, default=100)
    parser.add_argument('-bs', '--batch_size', help='batch size. defalut value is 128',
                        type=int, default=128)
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0',
                        choices=[-1, 0, 1], type=int, default=0)
    parser.add_argument('-nc', '--n_critic',
                        help='specify number of iteretion of critic by this number.'
                        ' defalut value is 5',
                        type=int, default=5)
    parser.add_argument('-c_l', '--clip_lower',
                        help='specify lower of clip range by this number.',
                        type=float, default=-0.01)
    parser.add_argument('-c_u', '--clip_upper',
                        help='specify upper of clip range by this number.',
                        type=float, default=0.01)
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)

    # 引数を解析する
    args = parser.parse_args()

    gpu = args.gpu
    batch_size = args.batch_size
    n_hidden = args.hidden
    epoch = args.epoch
    seed = args.seed
    number = args.number  # number of experiments
    out = pathlib.Path("result_{0}".format(number))
    if not out.exists():
        out.mkdir()
    out /= pathlib.Path("result_{0}_{1}".format(number, seed))
    if not out.exists():
        out.mkdir()

    # 引数(ハイパーパラメータの設定)の書き出し
    with open(out / "args.txt", "w") as f:
        f.write(str(args))

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# n_hidden: {}'.format(n_hidden))
    print('# epoch: {}'.format(epoch))
    print('# out: {}'.format(out))

    # Set up a neural network to train
    gen = Generator(n_hidden=n_hidden, nobias=False)
    dis = Critic(nobias=False)

    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        gen.to_gpu()  # Copy the model to the GPU
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # Load the GMM dataset
    dataset, _ = chainer.datasets.get_svhn(withlabel=False, scale=255.)
    train_iter = chainer.iterators.SerialIterator(dataset, batch_size)
    print("# Data size: {}".format(len(dataset)), end="\n\n")

    # Set up a trainer
    updater = WGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen,
            'critic': opt_dis
        },
        n_critic=args.n_critic,
        clip_range=[args.clip_lower, args.clip_upper],
        device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    snapshot_interval = (5, 'epoch')
    display_interval = (1, 'epoch')
    # trainer.extend(extensions.dump_graph("gen/loss", out_name="gen.dot"))
    # trainer.extend(extensions.dump_graph("dis/loss", out_name="dis.dot"))
    # trainer.extend(
    #     extensions.snapshot(
    #         filename='snapshot_epoch_{.updater.epoch}.npz'),
    #     trigger=snapshot_interval)
    # trainer.extend(
    #     extensions.snapshot_object(gen, 'gen_epoch_{.updater.epoch}.npz'),
    #     trigger=snapshot_interval)
    # trainer.extend(
    #     extensions.snapshot_object(dis, 'dis_epoch_{.updater.epoch}.npz'),
    #     trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport([
            'epoch',
            'iteration',
            'gen/loss',
            'critic/loss',
            'elapsed_time'
        ]),
        trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=100))
    # trainer.extend(
    #     extensions.PlotReport(
    #         ['gen/loss', 'dis/loss'],
    #         x_key='epoch',
    #         file_name='loss_{0}_{1}.jpg'.format(number, seed),
    #         grid=False))
    # trainer.extend(out_generated_image(
    #     gen, 7, 7, seed, out), trigger=display_interval)

    since2 = datetime.datetime.now()
    # Run the training
    trainer.run()
    print("Elapsed Time:", datetime.datetime.now()-since2)
    print("Wall-Time:", datetime.datetime.now()-since1)
