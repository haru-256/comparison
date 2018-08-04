import chainer
from chainer import training
from chainer.training import extensions
import chainer.links as L
from chainer import initializers
from chainer import training
from chainer.training import updaters
from chainer import optimizers
from chainer.training import extensions
import argparse
from chainer import training
from chainer import optimizers

if __name__ == '__main__'
# make parser
    parser = argparse.ArgumentParser(
        prog='classify mnist',
        usage='python train.py',
        description='description',
        epilog='end',
        add_help=True
    )
    # add argument
    parser.add_argument('-s', '--seed', help='seed',
                        type=int, required=True)
    parser.add_argument('-n', '--number', help='the number of experiments.',
                        type=int, required=True)
    parser.add_argument('-e', '--epoch', help='the number of epoch, defalut value is 100',
                        type=int, default=100)
    parser.add_argument('-bs', '--batch_size', help='batch size. defalut value is 128',
                        type=int, default=128)
    parser.add_argument('-vs', '--val_size', help='validation dataset size. defalut value is 0.15',
                        type=float, default=0.15)
    parser.add_argument('-g', '--gpu', help='specify gpu by this number. defalut value is 0,'
                        ' -1 is means don\'t use gpu',
                        choices=[-1, 0, 1], type=int, default=0)
    parser.add_argument('-V', '--version', version='%(prog)s 1.0.0',
                        action='version',
                        default=False)
    # 引数を解析する
    args = parser.parse_args()

    gpu = args.gpu
    batch_size = args.batch_size
    epoch = args.epoch
    seed = args.seed
    number = args.number  # number of experiments
    out = pathlib.Path("result_{0}/result_{0}_{1}".format(number, seed))

    # 引数の書き出し
    with open(out / "args.text", "w") as f:
        f.write(str(args))

    # make directory
    pre = pathlib.Path(out.parts[0])
    for i, path in enumerate(out.parts):
        path = pathlib.Path(path)
        if i != 0:
            pre /= path
        if not pre.exists():
            pre.mkdir()
        pre = path

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batch_size))
    print('# epoch: {}'.format(epoch))
    print('# val size: {}'.format(args.val_size))
    print('# out: {}'.format(out))

    # path to a text file contains paths/labels pairs in distinct lines.
    train_file_path = pathlib.Path('train_path.txt').resolve()
    test_file_path = pathlib.Path('test_path.txt').resolve()

    try:
        # data immigration for validation data
        val_dir, _ = make_validation_dataset(
            data_dir, seed=seed, test_size=args.val_size)
        # load datasets
        train_datasets = chainer.datasets.LabeledImageDataset(train_file_path)
        test_datasets = chainer.datasets.LabeledImageDataset(test_file_path)
        # make iterator
        train_iter = chainer.iterators.SerialIterator(
            train_datasets, batch_size)
        test_iter = chainer.iterators.SerialIterator(
            test_datasets, batch_size, repeat=False, shuffle=False)

        # build model
        model = chainer.links.VGG16Layers()
        num_ftrs = model.fc8.out_size
        model.fc8 = L.Linear(in_size=None, out_size=101,
                             initialW=initializers.Normal(scale=0.02),
                             initial_bias=initializers.Normal(scale=0.02))
        model = L.Classifier(model)

        if gpu >= 0:
            # Make a specified GPU current
            chainer.backends.cuda.get_device_from_id(gpu).use()
            model.to_gpu()  # Copy the model to the GPU

        # make optimizer
        optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
        optimizer.setup(model)
        # make updater & set up trainer
        updater = updaters.StandardUpdater(
            train_iter, optimizer, device=device)
        trainer = training.Trainer(
            updater, stop_trigger=(epoch, 'epoch'), out=out)

        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.snapshot(
            filename='snapshot_epoch-{.updater.epoch}'))
        trainer.extend(extensions.snapshot_object(
            model.predictor, filename='model_epoch-{.updater.epoch}'))
        trainer.extend(extensions.Evaluator(test_iter, model, device=device))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
        trainer.extend(extensions.dump_graph('main/loss'))

        # run trainning
        trainer.run()

    except:
        Exception
        import traceback
        traceback.print_exc()
    finally:
        # undo data immigration
        for path in val_dir.glob("*/*.jpg"):
            shutil.move(path, str(path).replace("val", "train"))
