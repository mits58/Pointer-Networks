import argparse
import numpy
import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import util

class Seq2Seq(chainer.Chain):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2Seq, self).__init__()
        with self.init_scope():
            # NStepLSTM(n_layers, input_size, output_size, dropout)
            self.encoder = L.NStepLSTM(1, input_dim, hidden_dim, 0.1)
            self.decoder = L.NStepLSTM(1, output_dim, hidden_dim, 0.1)
            self.hidden_to_out = L.Linear(hidden_dim, output_dim + 1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def __call__(self, xs, ys):
        # set the array module based on using device
        xp = self.device.xp

        # make input data (batchsize, input_length, input_dim)
        xs = [chainer.Variable(x).reshape(10, 2) for x in xs]

        # make output data (batchsize, n + 1, output_dim)
        st_seq = xp.zeros((1, self.output_dim), dtype='float32')
        ys_in = [F.concat([st_seq, xp.identity(self.output_dim, dtype='float32')[y - 1]], axis=0) for y in ys]
        ys_out = [F.concat([y, xp.array([0])], axis=0) for y in ys]

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hx, cx, _ = self.encoder(None, None, xs)
        _, _, os = self.decoder(hx, None, ys_in)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        pred_dist = self.hidden_to_out(concat_os)
        loss = F.sum(F.softmax_cross_entropy(pred_dist, concat_ys_out, reduce='no')) / batch
        accuracy = F.accuracy(pred_dist, concat_ys_out)

        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)
        return loss

def main():
    parser = argparse.ArgumentParser(description='An implementation of seq2seq in chainer')
    parser.add_argument('--dataset', type=str, default="modeltest.txt",
                        help='dataset name')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=10)
    parser.add_argument('--batchsize', '-b', type=int, default=64)
    parser.add_argument('--device', '-d', type=str, default='-1')
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--epoch', type=int, default=350)
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # setting the using device
    device = chainer.get_device(args.device)

    # loading dataset
    print('loading dataset...')
    dataset = util.PlaneData(args.dataset, device)

    print('Device: {}'.format(device))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# dataset-size: {}'.format(len(dataset)))
    print('')

    # making seq2seq model
    model = Seq2Seq(args.input_dim, args.hidden_dim, args.output_dim)

    # Choose the using device
    model.to_device(device)
    device.use()

    # Setup an optimizer (default Adam)
    # Todo: checking add gradient clipping
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Split the dataset into traindata and testdata
    train, test = chainer.datasets.split_dataset_random(dataset, int(dataset.__len__() * 0.9))
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = chainer.training.updaters.StandardUpdater(train_iter, optimizer,
                                                device=device, converter=dataset.converter)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=device, converter=dataset.converter))

    trainer.extend(extensions.LogReport(filename='log.dat'))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                         'epoch', file_name='loss.png'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    # Print selected entries of the log to stdout
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                               'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()

    """
    Save the trained model and model attributes
    if args.save_model:
        chainer.serializers.save_npz('./result/{0}/{1}.model'.format(res_path, agg_name), model)
        save the model parameters (i.e. args)
        with open('./result/{0}/{1}.model_stat'.format(res_path, agg_name), 'wb') as f:
            pickle.dump(args, f)
    """

if __name__ == '__main__':
    main()
