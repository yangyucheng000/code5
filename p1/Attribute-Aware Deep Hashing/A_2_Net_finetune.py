import mindspore as ms
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import numpy as np
from mindspore.experimental import optim
import os
import time
import utils.evaluate as evaluate

from tqdm import tqdm
from loguru import logger
from models.A_2_net_loss import A_2_net_Loss
from data.data_loader import sample_dataloader
from utils import AverageMeter
import models.A_2_net as A_2_net


def train(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args):

    num_classes, att_size, feat_size = args.num_classes, code_length, 2048
    model = A_2_net.a_2_net(code_length=code_length, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
                            device=args.device, pretrained=True)

    model.to(args.device)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)
    criterion = A_2_net_Loss(code_length, args.gamma, args.batch_size, args.margin, finetune=True)

    model.load_state_dict(torch.load('./checkpoints/' + args.info[:-2] + 'pre/' + str(code_length) + '/model.pkl', map_location='cpu'), strict=False)
    query_code = torch.load('./checkpoints/' + args.info[:-2] + 'pre/' + str(code_length) + '/query_code.t')
    query_code = query_code.to(args.device)
    query_dataloader.dataset.get_onehot_targets = torch.load('./checkpoints/' + args.info[:-2] + 'pre/' + str(code_length) + '/query_targets.t')
    B = torch.load('./checkpoints/' + args.info[:-2] + 'pre/' + str(code_length) + '/database_code.t')
    B = B.to(args.device)
    retrieval_targets = torch.load('./checkpoints/' + args.info[:-2] + 'pre/' + str(code_length) + '/database_targets.t')
    retrieval_targets = retrieval_targets.to(args.device)

    '''mAP = evaluate.mean_average_precision(
        query_code.to(args.device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(args.device),
        retrieval_targets,
        args.device,
        args.topk,
    )
    print("Code_Length: " + str(code_length), end="; ")
    print('[Start mAP:{:.5f}]'.format(mAP))'''
    mAP = 0.8097

    num_retrieval = len(retrieval_dataloader.dataset)
    U = torch.zeros(args.num_samples, code_length).to(args.device)
    # B = torch.randn(num_retrieval, code_length).to(args.device)
    # retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)
    cnn_losses, hash_losses, quan_losses, reconstruction_losses, decorrelation_losses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    gan_losses = AverageMeter()
    start = time.time()
    best_mAP = mAP
    for it in range(args.max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size, args.root, args.dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(args.device)
        S = (train_targets @ retrieval_targets.t() > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r

        # Training CNN model
        model.train()
        for epoch in range(args.max_epoch):
            cnn_losses.reset()
            hash_losses.reset()
            quan_losses.reset()
            reconstruction_losses.reset()
            decorrelation_losses.reset()
            gan_losses.reset()
            pbar = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
            # print((len(train_dataloader)))
            for batch, (data, targets, index) in pbar:
                data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
                optimizer.zero_grad()

                F, dret, all_f, deep_S, inputs = model(data, targets)
                inputs['img'] = data
                U[index, :] = F.data
                cnn_loss, hash_loss, quan_loss, reconstruction_loss, decorrelation_loss, gan_loss = criterion(F, B, S[index, :], sample_index[index], dret, all_f, deep_S, inputs)
                cnn_losses.update(cnn_loss.item())
                hash_losses.update(hash_loss.item())
                quan_losses.update(quan_loss.item())
                reconstruction_losses.update(reconstruction_loss.item())
                decorrelation_losses.update(decorrelation_loss.item())
                gan_losses.update(gan_loss.item())
                cnn_loss.backward()
                optimizer.step()
                # print(optimizer.param_groups[0]['lr'])
            logger.info('[epoch:{}/{}][cnn_loss:{:.4f}][hash_loss:{:.4f}][quan_loss:{:.4f}][reconstruction_loss:{:.4f}][decorrelation_loss:{:.4f}][gan_loss:{:.4f}]'.format(epoch+1, args.max_epoch,
                        cnn_losses.avg, hash_losses.avg, quan_losses.avg, reconstruction_losses.avg, decorrelation_losses.avg, gan_losses.avg))
        scheduler.step()
        # Update B
        expand_U = torch.zeros(B.shape).to(args.device)
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, code_length, args.gamma)

        logger.info('[iter:{}/{}][iter_time:{:.2f}]'.format(it+1, args.max_iter, time.time()-iter_start))
        # Total loss
        # iter_loss = calc_loss(U, B, S, code_length, sample_index, args.gamma)
        # logger.info('[iter:{}/{}][loss:{:.6f}][iter_time:{:.2f}]'.format(it+1, args.max_iter, iter_loss, time.time()-iter_start))

    # Evaluate
        # if (it < 20 and ((it+1)%20==0)) or (it>=20 and it<65 and ((it + 1) % 10 == 0)) or (it>=65 and ((it+1)%1==0)):
        if (it<30 and (it+1)%1==0) or (it>=30 and (it+1)%1==0):
            query_code = generate_code(model, query_dataloader, code_length, args.device)
            # print(len(query_dataloader))
            mAP = evaluate.mean_average_precision(
                query_code.to(args.device),
                B,
                query_dataloader.dataset.get_onehot_targets().to(args.device),
                retrieval_targets,
                args.device,
                args.topk,
            )
            logger.info(
                '[iter:{}/{}][code_length:{}][mAP:{:.5f}]'.format(it + 1, args.max_iter, code_length,
                                                                                   mAP))
            if mAP > best_mAP:
                best_mAP = mAP
                ret_path = os.path.join('checkpoints', args.info, str(code_length))
                if not os.path.exists(ret_path):
                    os.makedirs(ret_path)
                torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.t'))
                torch.save(B.cpu(), os.path.join(ret_path, 'database_code.t'))
                torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join(ret_path, 'query_targets.t'))
                torch.save(retrieval_targets.cpu(), os.path.join(ret_path, 'database_targets.t'))
                torch.save(model.state_dict(), os.path.join(ret_path, 'model.pkl'))
                # torch.save(model.cpu(), os.path.join(ret_path, 'model.t'))
                # model = model.to(args.device)
            logger.info('[iter:{}/{}][code_length:{}][mAP:{:.5f}][best_mAP:{:.5f}]'.format(it+1, args.max_iter, code_length, mAP, best_mAP))
    logger.info('[Training time:{:.2f}]'.format(time.time()-start))

    return best_mAP


def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    imgs=[]
    codes=[]
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length]).to(device)
        for batch, (data, targets, index) in enumerate(dataloader):
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            hash_code = model(data, targets)
            imgs.append(data.cpu())
            codes.append(hash_code.cpu())
            code[index, :] = hash_code.sign()
    '''codes=torch.cat(codes,0)[:,4]
    index=torch.argsort(codes)[:20]
    imgs=torch.cat(imgs,0)[index]
    for i, img in enumerate(imgs):
        show_image(img, 'images/'+str(i)+'.png')
    exit(0)'''
    return code


import matplotlib.pyplot as plt
@torch.no_grad()
def show_image(img, name):
    mean, std = [0.466, 0.471, 0.380], [0.195, 0.194, 0.192]
    mean, std = torch.tensor(mean).view(1, 1, 3), torch.tensor(std).view(1, 1, 3)
    img = img.permute(1, 2, 0).cpu()
    img = (img * std + mean).numpy()#[:, :, ::-1]

    plt.axis('off')
    plt.imshow(img)

    plt.draw()
    plt.savefig(name)
    #plt.show()
    plt.close()