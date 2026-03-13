import os
# os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
import argparse
import json
import time
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from torch.optim import lr_scheduler
# import pickle
from loss import SupConLoss
from models import MyDataset
from models.ccnet import ccnet
from utils import *


# Global device — set in __main__ after parsing args
device = torch.device('cpu')


def test(model, openset_file):
    """Compute EER on open-set persons (unseen during training).
    Uses self-matching within open-set. Returns EER (%).
    Saves threshold info to JSON for deployment.
    """
    print('--- EER Evaluation (open-set) ---')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    os.makedirs(path_rst + 'veriEER', exist_ok=True)

    testset = MyDataset(txt=openset_file, transforms=None, train=False)
    loader = DataLoader(dataset=testset, batch_size=512, num_workers=2)

    net = model
    net.to(device)
    net.eval()

    featDB, iddb = [], []
    for batch_id, (datas, target) in enumerate(loader):
        codes = net.getFeatureCode(datas[0].to(device)).cpu().detach().numpy()
        y = target.numpy()
        if batch_id == 0:
            featDB = codes
            iddb = y
        else:
            featDB = np.concatenate((featDB, codes), axis=0)
            iddb = np.concatenate((iddb, y))

    n = featDB.shape[0]
    n_persons = len(set(iddb.tolist()))
    print(f'  open-set: {n_persons} persons, {n} images')

    # intra/inter class distance stats
    cos_mat = np.dot(featDB, featDB.T)
    dis_mat = np.arccos(np.clip(cos_mat, -1, 1)) / np.pi
    same_mask = (iddb[:, None] == iddb[None, :])
    np.fill_diagonal(same_mask, False)
    diff_mask = ~same_mask
    np.fill_diagonal(diff_mask, False)
    intra = dis_mat[same_mask]
    inter = dis_mat[diff_mask]
    print('  inner (min, max, mean, std): [%f, %f, %f, %f]' % (intra.min(), intra.max(), intra.mean(), intra.std()))
    print('  outer (min, max, mean, std): [%f, %f, %f, %f]' % (inter.min(), inter.max(), inter.mean(), inter.std()))

    # all-pairs scores (upper triangle only, no diagonal)
    iu = np.triu_indices(n, k=1)
    s = dis_mat[iu].tolist()
    same_upper = same_mask[iu]
    l = np.where(same_upper, 1, -1).tolist()

    scores_path = path_rst + 'veriEER/scores_EER_openset.txt'
    with open(scores_path, 'w') as f:
        for score, label in zip(s, l):
            f.write(f'{score} {label}\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + scores_path + ' scores_EER_openset')
    os.system('python ./getEER.py' + '  ' + scores_path + ' scores_EER_openset')

    # Read EER result and save threshold JSON
    eer_result_file = path_rst + 'veriEER/scores_EER_openset/rst_eer_th_auc.txt'
    eer_val = float('inf')
    if os.path.exists(eer_result_file):
        with open(eer_result_file) as f:
            parts = f.readline().split()
        eer_val   = float(parts[0])
        thresh_val = float(parts[1])
        auc_val    = float(parts[2])

        threshold_info = {
            "eer_percent":      round(eer_val, 6),
            "threshold_at_eer": round(thresh_val, 6),
            "auc":              round(auc_val, 8),
            "n_train_persons":  num_classes,
            "n_test_persons":   n_persons,
            "n_test_images":    n,
        }
        json_path = des_path + 'threshold_info.json'
        with open(json_path, 'w') as f:
            json.dump(threshold_info, f, indent=2)
        print(f'  EER: {eer_val:.4f}%  threshold: {thresh_val:.4f}  → saved {json_path}')
    else:
        print('  [WARN] EER result file not found.')

    return eer_val

# perform one epoch
def fit(epoch, model, data_loader, phase='training'):
    if phase != 'training' and phase != 'testing':
        raise TypeError('input error!')

    if phase == 'training':
        model.train()

    if phase == 'testing':
        # print('test')
        model.eval()

    running_loss = 0
    running_correct = 0

    for batch_id, (datas, target) in enumerate(data_loader):

        data     = datas[0].to(device)
        data_con = datas[1].to(device)
        target   = target.to(device)
        if phase == 'training':
            optimizer.zero_grad()
            output, fe1 = model(data, target)
            output2, fe2 = model(data_con, target)
            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                output, fe1 = model(data, None)
                output2, fe2 = model(data_con, None)
                fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

        ce = criterion(output, target)
        ce2 = con_criterion(fe, target)

        loss = weight1*ce+weight2*ce2

        ## log
        running_loss += loss.data.cpu().numpy()

        preds = output.data.max(dim=1, keepdim=True)[1]  # max returns (value, index)
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()

        ## update
        if phase == 'training':
            loss.backward(retain_graph=None)  #
            optimizer.step()

    ## log info of this epoch
    total = len(data_loader.dataset)
    loss = running_loss / total
    accuracy = (100.0 * running_correct) / total

    if epoch % 10 == 0:
        print('epoch %d: \t%s loss is \t%7.5f ;\t%s accuracy is \t%d/%d \t%7.3f%%' % (
        epoch, phase, loss, phase, running_correct, total, accuracy))

    return loss, accuracy

if __name__== "__main__" :

    parser = argparse.ArgumentParser(
        description="CO3Net for Palmprint Recfognition"
    )

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epoch_num", type=int, default=3000)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--weight1", type=float, default=0.8)
    parser.add_argument("--weight2", type=float, default=0.2)
    parser.add_argument("--com_weight", type=float, default=0.8)
    parser.add_argument("--id_num", type=int, default=540,
                        help="number of training identities (our dataset: 540)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU id to use. Set to 'cpu' to force CPU mode.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--redstep", type=int, default=500)

    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=500)

    ##Training Path
    parser.add_argument("--train_set_file", type=str, default='./data/train_ours.txt')
    parser.add_argument("--test_set_file", type=str, default='./data/test_probe.txt')
    parser.add_argument("--openset_file", type=str, default='./data/test_openset.txt',
                        help="Open-set persons (unseen during training) for EER evaluation")

    ##Store Path
    parser.add_argument("--des_path", type=str, default='./results/checkpoint/')
    parser.add_argument("--path_rst", type=str, default='./results/rst_test/')

    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. results/checkpoint/net_params_best.pth)")

    args = parser.parse_args()

    # device setup
    if args.gpu_id == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
        print('Running on CPU')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = torch.device('cuda')
        print(f'Running on GPU {args.gpu_id}: {torch.cuda.get_device_name(0)}')

    batch_size = args.batch_size
    epoch_num = args.epoch_num
    num_classes = args.id_num
    weight1 = args.weight1
    weight2 = args.weight2
    comp_weight = args.com_weight

    print('weight of cross:', weight1)
    print('weight of contra:', weight2)
    print('weight of competition:',comp_weight)
    print('tempture:', args.temp)

    des_path = args.des_path
    path_rst = args.path_rst

    os.makedirs(des_path, exist_ok=True)
    os.makedirs(path_rst, exist_ok=True)

    # path
    train_set_file = args.train_set_file
    openset_file = args.openset_file

    # dataset
    trainset = MyDataset(txt=train_set_file, transforms=None, train=True, imside=128, outchannels=1)

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2, shuffle=True)

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    print('------Init Model------')
    net = ccnet(num_classes=num_classes,weight=comp_weight)
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(state)
        print(f'Resumed from: {args.resume}')
    net.to(device)

    #
    criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(temperature=args.temp, base_temperature=args.temp)  ######agfzgfda

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.redstep, gamma=0.8)

    train_losses, train_accuracy = [], []
    best_eer = float('inf')

    for epoch in range(epoch_num):

        epoch_loss, epoch_accuracy = fit(epoch, net, data_loader_train, phase='training')

        scheduler.step()

        # ── logs ──
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)

        # save current model and loss curve every 10 epochs
        if epoch % 10 == 0 or epoch == (epoch_num - 1):
            torch.save(net.state_dict(), des_path + 'net_params.pth')
            saveLossACC(train_losses, [], train_accuracy, [], best_eer, path_rst)

        # periodic checkpoint (keeps every N epochs)
        if epoch % args.save_interval == 0 and epoch != 0:
            torch.save(net.state_dict(), des_path + 'epoch_' + str(epoch) + '_net_params.pth')

        # periodic EER evaluation — best model saved by EER (lower is better)
        if epoch % args.test_interval == 0 and epoch != 0:
            print('------------')
            eer = test(net, openset_file)
            if eer < best_eer:
                best_eer = eer
                torch.save(net.state_dict(), des_path + 'net_params_best.pth')
                print(f'  → new best EER: {best_eer:.4f}%  model saved.')

    print('------------')
    print('Last')
    test(net, openset_file)

    print('------------')
    print('Best')
    best_net = ccnet(num_classes=num_classes, weight=comp_weight)
    best_net.load_state_dict(torch.load(des_path + 'net_params_best.pth', map_location='cpu'))
    best_net.to(device)
    test(best_net, openset_file)
