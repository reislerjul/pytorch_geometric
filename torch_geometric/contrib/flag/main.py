# from utils.ckpt_util import save_ckpt
import logging
import sys
import time

import torch
import torch.nn.functional as F
from model import DeeperGCN
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

# from .args import ArgsInit
from torch_geometric.contrib.flag.args import ArgsInit
from torch_geometric.utils import add_self_loops, to_undirected

sys.path.insert(0, '..')
# from torch_geometric.contrib.flag.attacks import *


def flag(model_forward, perturb_shape, y, args, optimizer, device, criterion):
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(
        -args.step_size, args.step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= args.m

    for _ in range(args.m - 1):
        loss.backward()
        perturb_data = perturb.detach() + args.step_size * torch.sign(
            perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()

    return loss, out


@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator):
    model.eval()
    out = model(x, edge_index)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def train(model, x, edge_index, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()

    pred = model(x, edge_index)[train_idx]

    loss = F.nll_loss(pred, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def train_flag(model, x, edge_index, y_true, train_idx, optimizer, device,
               args):

    # forward = lambda perturb: model(x + perturb, edge_index)[train_idx]
    def forward(perturb):
        return model(x + perturb, edge_index)[train_idx]

    model_forward = (model, forward)
    target = y_true.squeeze(1)[train_idx]

    loss, out = flag(model_forward, x.shape, target, args, optimizer, device,
                     F.nll_loss)

    return loss.item()


def main():

    args = ArgsInit().save_exp()

    device = torch.device('cpu')
    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            raise ValueError(
                'GPU support requested but neither MPS nor CUDA is available')

    dataset = PygNodePropPredDataset(name=args.dataset)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(args.dataset)

    x = data.x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    edge_index = data.edge_index.to(device)
    edge_index = to_undirected(edge_index, data.num_nodes)

    if args.self_loop:
        edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)[0]

    # sub_dir = 'SL_{}'.format(args.self_loop)

    args.in_channels = data.x.size(-1)
    args.num_tasks = dataset.num_classes

    logging.info('%s' % args)

    model = DeeperGCN(args).to(device)

    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results = {
        'highest_valid': 0,
        'final_train': 0,
        'final_test': 0,
        'highest_train': 0
    }

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):

        # epoch_loss =
        # train(model, x, edge_index, y_true, train_idx, optimizer)
        epoch_loss = train_flag(model, x, edge_index, y_true, train_idx,
                                optimizer, device, args)

        logging.info('Epoch {}, training loss {:.4f}'.format(
            epoch, epoch_loss))
        model.print_params(epoch=epoch)

        result = test(model, x, edge_index, y_true, split_idx, evaluator)
        logging.info(result)
        train_accuracy, valid_accuracy, test_accuracy = result

        if train_accuracy > results['highest_train']:
            results['highest_train'] = train_accuracy

        if valid_accuracy > results['highest_valid']:
            results['highest_valid'] = valid_accuracy
            results['final_train'] = train_accuracy
            results['final_test'] = test_accuracy

            # save_ckpt(model, optimizer,
            #           round(epoch_loss, 4), epoch,
            #           args.model_save_path,
            #           sub_dir, name_post='valid_best')

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    main()
