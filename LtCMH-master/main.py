from config import opt
from data_handler import *
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from models import ImgModule, TxtModule,AutoEncoder,ImgModuleNus
from utils import calc_map_k
import xlrd
from utils import calc_hammingDist
def excel_to_Matrix(path):
#读excel数据转为矩阵函数
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    #获取excel中第一个sheet表
    nrows = table.nrows
    #行数
    ncols = table.ncols
    #列数
    datamatrix = np.zeros((nrows, ncols))
    for x in range(ncols):
        cols = table.col_values(x)
        cols1 = np.matrix(cols)
        #把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols1
        #把数据进行存储
    return datamatrix

def train(**kwargs):
    opt.parse(kwargs)

    # Xq,Xt,Xr,Yq,Yt,Yr, Lq,Lt,Lr = load_data()
    Xtest, Xtrain, Xretrieval, Ltest, Ltrain, Lretrieval,Ytest, Ytrain,Yretrieval= load_data(opt.data_path)
    pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    y_dim = 1386
    y2_dim= 4096

    # X,X1,Y,Y1,L,L1 = split_data(Xtest,Xtrain,Ytest,Ytrain,Ltest,Ltrain)

    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit,pretrain_model)
    img_model_nus= ImgModuleNus(y2_dim,opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)

    en_decoder = AutoEncoder()
    en_decoder.load_state_dict(torch.load('./data/encoderflicker64.pth'))
    if opt.use_gpu:
        img_model = img_model.to(opt.device)
        img_model_nus=img_model_nus.to(opt.device)
        txt_model = txt_model.to(opt.device)

        en_decoder=en_decoder.to(opt.device)


    train_L = torch.from_numpy(Ltrain)
    train_x = torch.from_numpy(Xtrain)
    train_y = torch.from_numpy(Ytrain)

    query_L = torch.from_numpy(Ltest)
    query_x = torch.from_numpy(Xtest)
    query_y = torch.from_numpy(Ytest)

    retrieval_L = torch.from_numpy(Lretrieval)
    retrieval_x = torch.from_numpy(Xretrieval)
    retrieval_y = torch.from_numpy(Yretrieval)

    num_train = 8072

    F_buffer = torch.randn(num_train, opt.bit)
    G_buffer = torch.randn(num_train, opt.bit)

    if opt.use_gpu:
        train_L = train_L.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()


    B = torch.sign(F_buffer + G_buffer)

    batch_size = opt.batch_size

    lr = opt.lr
    lr_hash = opt.hash_lr
    # optimizer_img = SGD(img_model.parameters(), lr=lr)
    optimizer_img = SGD(img_model_nus.parameters(), lr=lr)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr)


    learning_rate = np.linspace(opt.lr, np.power(10, -6.), opt.max_epoch + 1)
    hashnet_learning_rate = np.linspace(opt.hash_lr, np.power(10, -6.), opt.max_epoch + 1)
    result = {
        'loss': []
    }

    ones = torch.ones(batch_size, 1)
    ones_ = torch.ones(num_train - batch_size, 1)
    unupdated_size = num_train - batch_size

    max_mapi2t = max_mapt2i = 0.

    for epoch in range(opt.max_epoch):
        # train image net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            image = Variable(train_x[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float))
            if opt.use_gpu:
                image = image.cuda()
                sample_L = sample_L.cuda()
                ones = ones.cuda()
                ones_ = ones_.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)

            cur_f = img_model_nus(image)  # cur_f: (batch_size, bit)
            imgindi, txtindi, common, imgout, txtout = en_decoder(cur_f, G_buffer[ind,:])
            # hf, fhat = hashmodelf(cur_f+0.4*common+0.6*imgindi)
            e1 = img_model_nus.e1(cur_f)
            e2 = img_model_nus.e2(cur_f)
            cur_f=cur_f+e1*common+e2*imgindi
            F_buffer[ind, :] = cur_f.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)
            # hashloss=-torch.sum(sample_L*torch.log(fhat))
            theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
            logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
            quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))
            loss_x = logloss_x + opt.gamma * quantization_x + opt.eta * balance_x
            loss_x /= (batch_size * num_train)
            # optimizer_hashf.zero_grad()
            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()
            # optimizer_hashf.step()
        # train txt net
        for i in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            sample_L = Variable(train_L[ind, :])
            text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            if opt.use_gpu:
                text = text.cuda()
                sample_L = sample_L.cuda()

            # similar matrix size: (batch_size, num_train)
            S = calc_neighbor(sample_L, train_L)  # S: (batch_size, num_train)
            cur_g = txt_model(text)  # cur_f: (batch_size, bit)
            e1 = txt_model.e1(cur_g)
            e2 = txt_model.e2(cur_g)
            imgindi, txtindi, common, imgout, txtout = en_decoder(F_buffer[ind,:], cur_g)
            # hg, ghat = hashmodelg(cur_g+0.4*common+0.6*txtindi)
            cur_g=cur_g+e1*common+e2*txtindi
            G_buffer[ind, :] = cur_g.data
            F = Variable(F_buffer)
            G = Variable(G_buffer)

            # calculate loss
            # theta_y: (batch_size, num_train)
            # hashloss = -torch.sum(sample_L * torch.log(ghat))
            theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
            logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
            quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))
            loss_y = logloss_y + opt.gamma * quantization_y + opt.eta * balance_y
            loss_y /= (num_train * batch_size)

            optimizer_txt.zero_grad()
            # optimizer_hashg.zero_grad()
            loss_y.backward()
            optimizer_txt.step()
            # optimizer_hashg.step()

        # update B
        B = torch.sign(F_buffer + G_buffer)
        #
        # # calculate total loss
        # loss = calc_loss(B, F, G, Variable(Sim), opt.gamma, opt.eta)
        #
        # print('...epoch: %3d, loss: %3.3f, lr: %f' % (epoch + 1, loss.data, lr))
        # result['loss'].append(float(loss.data))

        if opt.valid:
            mapi2t, mapt2i = valid(img_model_nus, txt_model,query_x, retrieval_x, query_y, retrieval_y,
                                   query_L, retrieval_L,en_decoder)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
            if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                img_model_nus.save(img_model_nus.module_name + '.pth')
                txt_model.save(txt_model.module_name + '.pth')

        lr = learning_rate[epoch + 1]
        lr_hash = hashnet_learning_rate[epoch + 1]

        # set learning rate
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr


    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        result['mapi2t'] = max_mapi2t
        result['mapt2i'] = max_mapt2i
    else:
        mapi2t, mapt2i = valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y,
                               query_L, retrieval_L)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
        result['mapi2t'] = mapi2t
        result['mapt2i'] = mapt2i

    write_result(result)


def valid(img_model, txt_model, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L,encoder):
    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)

    rBX,rBY = generate_code(img_model,txt_model,retrieval_x,retrieval_y,opt.bit,encoder)


    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    return mapi2t, mapt2i
def transformCommas(line):
    out = ''
    insideQuote = False
    for c in line:
        if c == '"':
            insideQuote = not insideQuote
        if insideQuote == True and c == ',':
            out += '.'
        else:
            out += c
    return out

def test(**kwargs):
    opt.parse(kwargs)

    Xtest, Xtrain, Xretrieval, Ltest, Ltrain, Lretrieval,Ytest, Ytrain,Yretrieval= load_data(opt.data_path)
    pretrain_model = load_pretrain_model(opt.pretrain_model_path)
    y_dim = 1386
    y2_dim = 4096


    print('...loading and splitting data finish')

    img_model = ImgModule(opt.bit, pretrain_model)
    img_model_nus = ImgModuleNus(y2_dim, opt.bit)
    img_model_nus.load_state_dict(torch.load('./checkpoints/img_module_nus.pth'))
    txt_model = TxtModule(y_dim, opt.bit)
    txt_model.load_state_dict(torch.load('./checkpoints/text_module.pth'))

    en_decoder = AutoEncoder()
    en_decoder.load_state_dict(torch.load('./data/encoderflicker64.pth'))
    # labelid = transformCommas(np.loadtxt('data.txt'))
    if opt.use_gpu:
        img_model = img_model.to(opt.device)
        img_model_nus = img_model_nus.to(opt.device)
        txt_model = txt_model.to(opt.device)

        en_decoder = en_decoder.to(opt.device)

    train_L = torch.from_numpy(Ltrain)
    train_x = torch.from_numpy(Xtrain)
    train_y = torch.from_numpy(Ytrain)

    query_L = torch.from_numpy(Ltest)
    query_x = torch.from_numpy(Xtest)
    query_y = torch.from_numpy(Ytest)

    retrieval_L = torch.from_numpy(Lretrieval)
    retrieval_x = torch.from_numpy(Xretrieval)
    retrieval_y = torch.from_numpy(Yretrieval)

    num_train = 8072

    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)
    print('...test MAP: MAP(i->t): %3.3f, MAP(t->i): %3.3f' % (mapi2t, mapt2i))

    return mapi2t,mapt2i


def pr_curve(query_code, retrieval_code, query_targets, retrieval_targets):
    """
    P-R curve.
    Args
        query_code(torch.Tensor): Query hash code.
        retrieval_code(torch.Tensor): Retrieval hash code.
        query_targets(torch.Tensor): Query targets.
        retrieval_targets(torch.Tensor): Retrieval targets.
        device (torch.device): Using CPU or GPU.
    Returns
        P(torch.Tensor): Precision.
        R(torch.Tensor): Recall.
    """
    num_query = query_code.shape[0]
    num_bit = query_code.shape[1]
    P = torch.zeros(num_query, num_bit + 1).to(opt.device)
    R = torch.zeros(num_query, num_bit + 1).to(opt.device)
    for i in range(num_query):
        gnd = (query_targets[i].unsqueeze(0).mm(retrieval_targets.t()) > 0).float().squeeze().to(opt.device)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(opt.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    P=P.cpu()
    R=R.cpu()
    plt.xticks([0.2,0.4,0.6,0.8,1.0])
    plt.yticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9])

    plt.plot(R, P)
    plt.grid(True)
    # plt.xlim(0.3, 1)
    plt.ylim(0.3, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    # plt.legend()
    plt.show()




def split_data(Xtest,Xtrain,Ytest,Ytrain,Ltest,Ltrain):
    X = {}
    X1={}
    X['query'] = Xtest[0: opt.query_size]
    X1['train'] = Xtrain[0: opt.training_size]
    X1['retrieval'] = Xtrain[0: opt.database_size]

    Y = {}
    Y1={}
    Y['query'] = Ytest[0: opt.query_size]
    Y1['train'] = Ytrain[0: opt.training_size]
    Y1['retrieval'] = Ytrain[0: opt.database_size]

    L = {}
    L1={}
    L['query'] = Ltest[0: opt.query_size]
    L1['train'] = Ltrain[0: opt.training_size]
    L1['retrieval'] = Ltrain[0:opt.database_size]

    return X,X1, Y,Y1, L,L1

def generate_code(img_model,txt_model, X,Y,bit,encoder,queryL):
    batch_size = opt.batch_size
    # for i in queryL.shape[0]:
    #     if queryL[i]
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    Bi = torch.zeros(num_data, bit, dtype=torch.float)
    Bt =  torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        Bi = Bi.to(opt.device)
        Bt = Bt.to(opt.device)
        img_model = img_model.to(opt.device)
        txt_model = txt_model.to(opt.device)

        encoder = encoder.to(opt.device)
    img_model.eval()
    txt_model.eval()

    encoder.eval()


    with torch.no_grad():
        for i in tqdm(range(num_data // batch_size + 1)):
            ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
            image = X[ind,:].unsqueeze(1).unsqueeze(-1).type(torch.float)

            text= Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
            if opt.use_gpu:
                image = image.to(opt.device)
                text=text.to(opt.device)

            cur_f = img_model(image)
            ei1 = img_model.e1(cur_f)
            ei2 = imgout.e2(cur_f)

            cur_g = txt_model(text)
            et1 = txt_model.e1(cur_g)
            et2 = txt_model.e2(cur_g)
            imgindi, txtindi, common, imgout, txtout = encoder(cur_f, cur_g)

            # hf, _ = hashmodelf(cur_f+0.4*common+0.6*imgindi)
            cur_f=cur_f+ei1*common+ei2*imgindi

            # hg, _ = hashmodelg(cur_g+0.4*common+0.6*txtindi)
            cur_g=cur_g+et1*common+et2*txtindi
            Bi[ind, :] = cur_f.data
            Bt[ind,:] = cur_g.data
    Bi = torch.sign(Bi)
    Bt = torch.sign(Bt)
    return Bi,Bt
def calc_neighbor(label1, label2):
    # calculate the similar matrix
    if opt.use_gpu:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    else:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return Sim


def calc_loss(B, F, G, Sim, gamma, eta):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    loss = term1 + gamma * term2 + eta * term3
    return loss


def generate_image_code(img_model, X, bit):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        if opt.use_gpu:
            image = image.cuda()
        cur_f = img_model(image)
        B[ind, :] = cur_f.data
    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, bit):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1)):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if opt.use_gpu:
            text = text.cuda()
        cur_g = txt_model(text)
        B[ind, :] = cur_g.data
    B = torch.sign(B)
    return B


def write_result(result):
    import os
    with open(os.path.join(opt.result_dir, 'result.txt'), 'w') as f:
        for k, v in result.items():
            f.write(k + ' ' + str(v) + '\n')


def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    # import fire
    # fire.Fire()
    # qBX,qBY,rBX,rBY,qL,rL=test()
    # pr_curve(qBX,rBY,qL,rL)
    # R1=R1.cpu().numpy
    # P1=P1.cpu().numpy
    # fig = plt.figure(figsize=(5, 5))
    # plt.plot(R1, P1)
    # plt.grid(True)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # # plt.legend()
    # plt.show()
    # R2,P2=pr_curve(qBY, rBX, qL, rL)
    # fig = plt.figure(figsize=(5, 5))
    # plt.plot(R2, P2)
    # plt.grid(True)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # # plt.legend()
    # plt.show()
    test()
