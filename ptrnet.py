import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import matplotlib
matplotlib.use('Agg')

import util


class Attention(chainer.Chain):
    """
    A class for Attention in seq2seq.

    Attributes
    ----------
    W1, W2, v : chainer.links.Linear
        Learnable Matrices and vector (in soft attention)
    hidden_dim : int
        number of hidden units in each layer
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        with self.init_scope():
            self.W1 = L.Linear(hidden_dim, hidden_dim)
            self.W2 = L.Linear(hidden_dim, hidden_dim)
            self.v = L.Linear(hidden_dim, 1)

        self.hidden_dim = hidden_dim

    def __call__(self, es, ds):
        """
        Parameters
        ----------
        es : list
            hidden states of encoder
        ds : list
            hidden states of decoder
        """
        # calculation (W1 * ej),  (W2 * dj) and
        #    make tensor (n, m(P)) from each batchdata
        # i wanna accelerate this code :(
        probs = []
        indices = []
        for e_i, d_i in zip(es, ds):
            dim_n = e_i.shape[0]  # n
            dim_mp = d_i.shape[0]  # m(P)

            # expand e_i
            expanded_ei = F.repeat(F.expand_dims(self.W1(e_i), axis=0), dim_mp, axis=0)
            # expand d_i
            expanded_di = F.transpose(F.repeat(F.expand_dims(self.W2(d_i), axis=0), dim_n, axis=0), (1, 0, 2))
            # sum up two tensor and activate with tanh and lineared
            activated = F.squeeze(self.v(F.tanh(expanded_di + expanded_ei), n_batch_axes=len(expanded_ei.shape) - 1))

            probs.append(activated)
            indices.append(F.argmax(F.softmax(activated, axis=-1), axis=-1))

        return probs, indices


class Seq2Seq(chainer.Chain):
    def __init__(self, input_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        with self.init_scope():
            # NStepLSTM(n_layers, input_size, output_size, dropout)
            self.encoder = L.NStepLSTM(1, input_dim, hidden_dim, 0.1)
            self.decoder = L.NStepLSTM(1, input_dim, hidden_dim, 0.1)
            # For attention
            self.attention = Attention(hidden_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def __call__(self, xs, ys):
        # set the array module based on using device
        xp = self.device.xp
        batchsize = len(xs)

        # make input data (batchsize, input_length, input_dim)
        xs = [chainer.Variable(x).reshape(-1, self.input_dim) for x in xs]

        # make output data (batchsize, n + 1, output_dim)
        # input seq (all zero)
        st_seq = chainer.Variable(xp.zeros((1, self.input_dim), dtype='float32'))

        # make ys_input
        ys_in = []
        for x, y in zip(xs, ys):
            y_i = F.concat([x[ind - 1] for ind in y], axis=0).reshape(-1, self.input_dim)
            ys_in.append(F.concat([st_seq, y_i], axis=0).reshape(-1, self.input_dim))

        ys_out = [F.concat([y, xp.array([0])], axis=0) for y in ys]

        # None represents a zero vector in an encoder.
        hx, cx, yx = self.encoder(None, None, xs)
        hs, _, os = self.decoder(hx, None, ys_in)

        # calculation attention using yx, os
        prob, pointer = self.attention(yx, os)

        # calculating loss and accuracy
        loss = []
        acc = []
        for dist, t in zip(prob, ys_out):
            loss.append(F.softmax_cross_entropy(dist, t - 1))
            acc.append(F.accuracy(dist[:len(dist) - 1], t[:len(t) - 1] - 1))

        loss = F.sum(F.stack(loss)) / batchsize
        accuracy = F.sum(F.stack(acc)) / batchsize

        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)
        return loss


def main():
    parser = argparse.ArgumentParser(description='An implementation of pointer networks in chainer')
    parser.add_argument('--dataset', type=str, default="modeltest.txt",
                        help='dataset name')
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=64)
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
    model = Seq2Seq(args.input_dim, args.hidden_dim)

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
