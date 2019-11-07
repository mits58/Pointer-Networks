import argparse
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


class Encoder(chainer.Chain):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__(
            # transforming graph feature vector into (4 * hidden_dim) dim. vector
            feat_to_hidden = L.Linear(input_dim, 4 * hidden_dim),
            # transforming hidden rep into (4 * hidden_dim) dim. vector
            hidden_to_hidden = L.Linear(hidden_dim, 4 * hidden_dim),
        )

    # forward
    def __call__(self, graph_embed, c, h):
        """
        graph_embed : graph embedding
        c           : internal memory
        h           : hidden representation

        but i dont know clearly :(
        """
        return F.lstm(c, self.feat_to_hidden(graph_embed) + self.hidden_to_hidden(h))


class Decoder(chainer.Chain):
    def __init__(self, input_dim, hidden_dim, output_classes):
        super(Decoder, self).__init__(
            # transforming graph feature vector into (4 * hidden_dim) dim. vector
            feat_to_hidden = L.Linear(input_dim, 4 * hidden_dim),
            # transforming hidden rep into (4 * hidden_dim) dim. vector
            hidden_to_hidden = L.Linear(hidden_dim, 4 * hidden_dim),
            # transforming output vector into graph feat vector
            output_to_feat = L.Linear(hidden_dim, input_dim),
            # transforming graph feat vector into one-hot vector (output_classes dim.)
            feat_to_onehot = L.Linear(input_dim, output_classes)
        )
        # check whether hidden state is initialized ?
        """
                self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
                self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

                self.zerograds()
        """

    # forward
    def __call__(self, y, c, h):
        c, h = F.lstm(c, self.feat_to_hidden(y), self.hidden_to_hidden(h))
        t = self.output_to_feat(h)
        return self.feat_to_onehot(t), t, c, h


class Seq2Seq(chainer.Chain):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2Seq, self).__init__(
            encoder = Encoder(input_dim, hidden_dim),
            decoder = Decoder(input_dim, hidden_dim, output_dim),
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def __call__(self, batch, targets=None):
        xp = self.device.xp
        batchsize = len(batch)

        for points, perms in zip(batch, targets)
            # initialize internal representation and hidden representation
            points = points.reshape(10, 2)  # todo: change to adaptive
            inter_rep = chainer.Variable(xp.zeros(self.hidden_dim), dtype='float32'))
            hidden_rep = chainer.Variable(xp.zeros(self.hidden_dim), dtype='float32'))

            ''' encoder '''
            for p in points:
                inter_rep, hidden_rep = self.encoder(p, inter_rep, hidden_rep)

            self.inter_rep = inter_rep
            self.hidden_rep = chainer.Variable(xp.zeros(self.hidden_dim), dtype='float32')

            # initialize the loss
            loss = chainer.Variable(xp.zeros((), dtype='float32'))

            ''' decoder '''
            # input start character into the decoder
            t = chainer.Variable(xp.zeros((), dtype='float32'))

            for p in perms:
                y, self.c, self.h = self.decoder(t, self.c, self.h)
                t = chainer.Variable(xp.array(p, dtype='int32'))
                loss += F.softmax_cross_entropy(y, t)

        return loss


def main():
    parser = argparse.ArgumentParser(description='An implementation of seq2seq in chainer')
    parser.add_argument('--dataset', type=str, default="tsp_10_train_exact.txt",
                        help='dataset name')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=10)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
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
