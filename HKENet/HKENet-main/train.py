import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from data_loader import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace
import numpy as np
from util import llprint
import dill
from outer_models import multi_label_metric
import random
from model_net import demo_net
from outer_models import FocalLoss
from outer_models import trd_loss
from case_study_code import case_study
# torch.cuda.manual_seed_all(1203)
torch.manual_seed(1203)
# torch.manual_seed(42)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def main(temp=None):
    device = "cuda"
    mimic_ver = 'iii'
    voc = dill.load(open(r'datas/voc_final_{}.pkl'.format(mimic_ver), 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word) + 1, len(pro_voc.idx2word) + 1, len(med_voc.idx2word) + 1)


    data = dill.load(open(r'datas/records_final_{}.pkl'.format(mimic_ver), 'rb'))#[::100]

    for patient in range(len(data)):
        for vst in range(len(data[patient])):
            # print(data[patient][vst][0])
            data[patient][vst][0]=[i+1 for i in data[patient][vst][0]]
            data[patient][vst][1]=[i+1 for i in data[patient][vst][1]]
            data[patient][vst][2]=[i+1 for i in data[patient][vst][2]]
    # print(data)
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]#[:10]
    data_train = remove_elements_by_percentage(data_train,temp)
    print('当前随机移除了{}%的训练集数据'.format(temp))
    # data_train = data[:split_point][::5]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    matrics = dill.load(open(r'three_graph\matrics.pkl', 'rb'))
    ddi_matrix = dill.load(open(r'datas/ddi_A_final_{}.pkl'.format(mimic_ver),'rb'))
    ehr_matrix = dill.load(open('datas/ehr_adj_final_{}.pkl'.format(mimic_ver),'rb'))
    ehr_matrix = np.where(ehr_matrix>0,1,0)
    ehr_matrix = mm_attn_cov[1:,1:]
    #=========================================
    ddi_matrix = torch.tensor(ddi_matrix,device=device)
    
    print(voc_size)

    train_loader = DataLoader(data_train, batch_size=1, collate_fn=pad_batch_v2_train, shuffle=True, pin_memory=False)
    eval_loader = DataLoader(data_eval, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True, pin_memory=False)
    model = demo_net(emb_dim=64, voc_size=voc_size, device=device, ehr_adj=ehr_matrix,).to(device)
    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=0.0001)
    EPOCH = 50
    demo_loss_1 = nn.BCELoss()
    # demo_loss_1 = FocalLoss()
    demo_loss_2 = nn.MultiLabelMarginLoss()
    # demo_loss_3 = InfoNCE()
    # demo_loss_4 = InfoNCE()
    demo_loss_3 = trd_loss()
    for epoch in range(EPOCH):
        ddi_rate = 0
        avg_precise = 0
        avg_recall = 0
        avg_f1 = 0
        count = 1e-6
        model.to(device)
        model.train()
        model_train = True
        all_his = True
        # case_study(model, list(train_loader)[::100], voc)
        # return None
        if model_train:
            for index,datas in enumerate(train_loader):
                # [diag,proc,drug,age,gender]
                datas = [i.to(device) for i in datas]
                output = list(model(datas))
                # output = model(datas)[0]

                gt_container = torch.zeros_like(output[0], device=device).reshape(-1,voc_size[2])
                loss3_target = np.full((output[0].size()), -1).reshape([-1,voc_size[2]])

                #gamenet
                temp_drug_label = []
                if all_his:
                    for batch_idx, batch in enumerate(temp_drug_label):
                        for idx, seq in enumerate(batch.reshape(-1,batch.size()[-1])):
                            for seq_idx,item in enumerate(seq):
                                try:
                                    # print(temp_drug_label.size())
                                    loss3_target[batch_idx][seq_idx] = item
                                except:
                                    print(temp_drug_label.size())
                                    loss3_target[batch_idx][seq_idx] = item
                else:
                    for batch_idx, batch in enumerate(temp_drug_label):
                        for idx, seq in enumerate(batch.reshape(-1,batch.size()[-1])):
                            for seq_idx,item in enumerate(seq):
                                try:
                                    # print(temp_drug_label.size())
                                    loss3_target[batch_idx][seq_idx] = item
                                except:
                                    print(temp_drug_label.size())
                                    loss3_target[batch_idx][seq_idx] = item


                loss3_target = loss3_target.reshape(output[0].size())
                if all_his:
                    for batch_idx, batch in enumerate(datas[2][0]):
                        for idx, seq in enumerate(batch):
                            gt_container[batch_idx][seq] = 1.0
                else:
                    gt_container[0][datas[2][0][-1]] = 1


                gt_container = gt_container.reshape(output[0].size())
                if all_his:
                    gt_container[:,0] = 0
                else:
                    gt_container[0] = 0

                loss_1 = demo_loss_1(output[0],gt_container)

                # print(output.size())
                loss_2 = F.multilabel_margin_loss(output[0], torch.LongTensor(loss3_target).to(device))

                co = 0.02

                loss = loss_1 + 0.02*loss_2# + co*loss_5 + co*loss_6 + co*loss_7
                # ddi_gt_adjust = (1-gt_container)*(1-gt_container).unsqueeze(dim=-1)
                # ddi_loss = (ddi_gt_adjust*ddi_matrix*output[1]).sum(dim=-1)
                # if ddi_rate >= 0.7:
                #     loss += 0.01*ddi_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                llprint('\r|'+\
                         '#'*int(50*(index/len(train_loader)))+\
                         '-'*int(50-50*(index/len(train_loader)))+\
                         '|{:.2f}%|train_step:{}/{}'.format(100*(index/len(train_loader)),index,len(train_loader))
                        )

        print()
        model.eval()
        prob_container = []
        gt_container = []
        labels_container = []
        ddi_cnt = 0
        ddi_all_cnt = 0
        avg_med = 0
        for index, datas in enumerate(eval_loader):
            datas = [i.to(device) for i in datas]
            output = model(datas)[0]
            # output = model(datas)[0]

            for idx,vst in enumerate(output.reshape(-1,voc_size[2])):
                gt_temp = torch.zeros_like(vst, device=device)
                if all_his:
                    gt_temp[datas[2][0][idx]] = 1
                else:
                    gt_temp[datas[2][0][-1]] = 1
                avg_med += gt_temp[datas[2][0][idx]].sum()
                gt_temp[0] = 0
                out_labels = torch.where(vst > 0.35, 1.0, 0.0)
                out_numbers = torch.nonzero(out_labels.squeeze())
                ddi_temp_container = out_labels*out_labels.T.unsqueeze(dim=-1)
                labels_container.append(out_labels)
                prob_container.append(vst)
                gt_container.append(gt_temp)
                # print((out_labels*gt_temp).size())

                if gt_temp.sum()!=0:
                    precise = (out_labels * gt_temp).sum() / (out_labels.sum() + 1e-9)
                    recall = (out_labels * gt_temp).sum() / (gt_temp.sum() + 1e-9)
                else:
                    continue
                avg_precise += precise
                avg_recall += recall

                if (precise + recall) == 0:
                    continue
                else:
                    f1 = (2.0 * precise * recall) / (precise + recall)
                avg_f1 += f1

                # ddi_cnt += ddi_rate_score(ddi_temp_container,ddi_matrix)[0]
                # ddi_all_cnt += ddi_rate_score(ddi_temp_container,ddi_matrix)[1]

                count += 1

            llprint('\r|' + \
                    '@' * int(50 * (index / len(eval_loader))) + \
                    '-' * int(50 - 50 * (index / len(eval_loader))) + \
                    '|{:.2f}%|eval_step:{}/{}'.format(100 * (index / len(eval_loader)), index, len(eval_loader))
                    )
        avg_precise=avg_precise/count
        avg_recall=avg_recall/count
        avg_f1=avg_f1/count
        # print('\n\n\n')
        # print(len(gt_container))
        # print('\n\n\n')
        jac,prauc,F_1 = multi_label_metric(gt_container,labels_container,prob_container)
        try:
            ddi_rate = ddi_cnt/ddi_all_cnt
        except:
            print('没有药物相互作用对')
            ddi_rate = 0

        print('\navg_prc = {}\n'.format(avg_precise),
              'avg_rec = {}\n'.format(avg_recall),
              'jac = {}\n'.format(jac),
              'prauc = {}\n'.format(prauc),
              'avg_f1 = {}\n'.format(avg_f1),
              'ddi_rate = {}\n'.format(ddi_rate),
              'avg_med = {}\n'.format(avg_med/count)
               )
        # print('\navg_prc = {}\n'.format(avg_precise),
        #       'avg_rec = {}\n'.format(avg_recall),
        #       'jac = {}\n'.format(jac),
        #       'prauc = {}\n'.format(prauc),
        #       'avg_f1 = {}\n'.format(avg_f1),
        #       'ddi_rate = {}\n'.format(ddi_rate),
        #       'avg_med = {}\n'.format(avg_med / count),
        #       '============================================='
        #       ,file=f)

        print(f'epoch{epoch}\n')

        # torch.save(model.state_dict(),'D:\PyCharm\projects\isbra\state_dict\iii_f1_{}.pt'.format(avg_f1))
            # pass
    # f.close()
# main()

for i in [0]:
    main(i)

# mole_encoder = MolecularGraphNeuralNetwork(N_fingerprint, mole_dim, layer_hidden=2, device=device)
# mole_emb = mole_encoder(MPNN_molecule_Set)
# print(mole_emb.size())
