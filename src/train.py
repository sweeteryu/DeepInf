#!/usr/bin/env python
# encoding: utf-8
# File Name: train.py
# Author: Jiezhong Qiu
# Create Time: 2017/11/08 07:43
# TODO:

from __future__ import absolute_import #忽略掉同目录的xxx.py而引入系统自带的标准xxx.py
from __future__ import unicode_literals #模块中显式出现的所有字符串转为unicode类型
from __future__ import division #"/"操作符执行的是截断除法3/4=0,当我们导入精确除法之后，"/"执行的是精确除法3/4=0.75
from __future__ import print_function #即使在python2.X，使用print就得像python3.X那样加括号使用。
#加上这些，如果你的python版本是python2.X，你也得按照python3.X那样使用这些函数。

import time
import argparse #用于解析命令行参数和选项的标准模块
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from gcn import BatchGCN
from gat import BatchGAT
from pscn import BatchPSCN
from sklearn.metrics import precision_recall_fscore_support #为每个类计算precision、recall、F-measure、support
from sklearn.metrics import roc_auc_score #直接根据真实值、预测值计算出AUC值（计算ROC的过程省略）
from sklearn.metrics import precision_recall_curve #根据预测值和真实值计算一条precision-recall曲线
from data_loader import ChunkSampler
from data_loader import InfluenceDataSet
from data_loader import PatchySanDataSet

import os #处理文件和目录的操作模块
import shutil #高级的文件、文件夹、压缩包处理模块
import logging #输出运行日志，可以设置输出日志的等级、保存路径、文件回滚等
from tensorboard_logger import tensorboard_logger #实现tensorboard可视化

logger = logging.getLogger(__name__) #返回一个名称为__name__的logger实例，一般name是各模块名。初始化
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp 打印日志时间，和日志信息

# Training settings
parser = argparse.ArgumentParser() #创建解析器对象ArgumentParser，可以添加参数
#add_argument()方法，用来指定程序需要接受的命令参数
parser.add_argument('--tensorboard-log', type=str, default='', help="name of this run")
parser.add_argument('--model', type=str, default='gcn', help="models used")
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="16,8",
                    help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="1,1,1",
                    help="Heads in each layer, splitted with comma")
parser.add_argument('--batch', type=int, default=2048, help="Batch size")
parser.add_argument('--dim', type=int, default=64, help="Embedding dimension")
parser.add_argument('--check-point', type=int, default=10, help="Eheck point")
parser.add_argument('--instance-normalization', action='store_true', default=False,
                    help="Enable instance normalization")
parser.add_argument('--shuffle', action='store_true', default=False, help="Shuffle dataset")
parser.add_argument('--file-dir', type=str, required=True, help="Input file directory")
parser.add_argument('--train-ratio', type=float, default=50, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=25, help="Validation ratio (0, 100)")
parser.add_argument('--class-weight-balanced', action='store_true', default=False,
                    help="Adjust weights inversely proportional"
                    " to class frequencies in the input data")
parser.add_argument('--use-vertex-feature', action='store_true', default=False,
                    help="Whether to use vertices' structural features")
parser.add_argument('--sequence-size', type=int, default=16,
                    help="Sequence size (only useful for pscn)")
parser.add_argument('--neighbor-size', type=int, default=5,
                    help="Neighborhood size (only useful for pscn)")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available() #（GPU是否可用）

np.random.seed(args.seed)
torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。

tensorboard_log_dir = 'tensorboard/%s_%s' % (args.model, args.tensorboard_log)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
tensorboard_logger.configure(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

# adj N*n*n
# feature N*n*f
# labels N*n*c
# Load data
# vertex: vertex id in global network N*n

if args.model == "pscn":
    influence_dataset = PatchySanDataSet(
            args.file_dir, args.dim, args.seed, args.shuffle, args.model,
            sequence_size=args.sequence_size, stride=1, neighbor_size=args.neighbor_size)
else:
    influence_dataset = InfluenceDataSet(
            args.file_dir, args.dim, args.seed, args.shuffle, args.model)

N = len(influence_dataset)
n_classes = 2
class_weight = influence_dataset.get_class_weight() \
        if args.class_weight_balanced else torch.ones(n_classes)
logger.info("class_weight=%.2f:%.2f", class_weight[0], class_weight[1])

feature_dim = influence_dataset.get_feature_dimension()
n_units = [feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")] + [n_classes]
logger.info("feature dimension=%d", feature_dim)
logger.info("number of classes=%d", n_classes)

train_start,  valid_start, test_start = \
        0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
train_loader = DataLoader(influence_dataset, batch_size=args.batch,
                        sampler=ChunkSampler(valid_start - train_start, 0))
valid_loader = DataLoader(influence_dataset, batch_size=args.batch,
                        sampler=ChunkSampler(test_start - valid_start, valid_start))
test_loader = DataLoader(influence_dataset, batch_size=args.batch,
                        sampler=ChunkSampler(N - test_start, test_start))

# Model and optimizer
if args.model == "gcn":
    model = BatchGCN(pretrained_emb=influence_dataset.get_embedding(),
                vertex_feature=influence_dataset.get_vertex_features(),
                use_vertex_feature=args.use_vertex_feature,
                n_units=n_units,
                dropout=args.dropout,
                instance_normalization=args.instance_normalization)
elif args.model == "gat":
    n_heads = [int(x) for x in args.heads.strip().split(",")]
    model = BatchGAT(pretrained_emb=influence_dataset.get_embedding(),
            vertex_feature=influence_dataset.get_vertex_features(),
            use_vertex_feature=args.use_vertex_feature,
            n_units=n_units, n_heads=n_heads,
            dropout=args.dropout, instance_normalization=args.instance_normalization)
elif args.model == "pscn":
    model = BatchPSCN(pretrained_emb=influence_dataset.get_embedding(),
                vertex_feature=influence_dataset.get_vertex_features(),
                use_vertex_feature=args.use_vertex_feature,
                n_units=n_units,
                dropout=args.dropout,
                instance_normalization=args.instance_normalization,
                sequence_size=args.sequence_size,
                neighbor_size=args.neighbor_size)
else:
    raise NotImplementedError

if args.cuda:
    model.cuda()
    class_weight = class_weight.cuda()

params = [{'params': filter(lambda p: p.requires_grad, model.parameters())
    if args.model == "pscn" else model.layer_stack.parameters()}]

optimizer = optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)


def evaluate(epoch, loader, thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    for i_batch, batch in enumerate(loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            features = features.cuda()
            graph = graph.cuda()
            labels = labels.cuda()
            vertices = vertices.cuda()

        output = model(features, vertices, graph)
        if args.model == "gcn" or args.model == "gat":
            output = output[:, -1, :]
        loss_batch = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
            log_desc, loss / total, auc, prec, rec, f1)

    tensorboard_logger.log_value(log_desc + 'loss', loss / total, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'auc', auc, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'prec', prec, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'rec', rec, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'f1', f1, epoch + 1)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return best_thr
    else:
        return None


def train(epoch, train_loader, valid_loader, test_loader, log_desc='train_'):
    model.train()

    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(train_loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            features = features.cuda()
            graph = graph.cuda()
            labels = labels.cuda()
            vertices = vertices.cuda()

        optimizer.zero_grad()
        output = model(features, vertices, graph)
        if args.model == "gcn" or args.model == "gat":
            output = output[:, -1, :]
        loss_train = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss in this epoch %f", loss / total)
    tensorboard_logger.log_value('train_loss', loss / total, epoch + 1)
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thr = evaluate(epoch, valid_loader, return_best_thr=True, log_desc='valid_')
        evaluate(epoch, test_loader, thr=best_thr, log_desc='test_')


# Train model
t_total = time.time()
logger.info("training...")
for epoch in range(args.epochs):
    train(epoch, train_loader, valid_loader, test_loader)
logger.info("optimization Finished!")
logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

logger.info("retrieve best threshold...")
best_thr = evaluate(args.epochs, valid_loader, return_best_thr=True, log_desc='valid_')

# Testing
logger.info("testing...")
evaluate(args.epochs, test_loader, thr=best_thr, log_desc='test_')
