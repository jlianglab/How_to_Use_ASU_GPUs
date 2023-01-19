import torch.nn as nn
import torch
import argparse
import os
from config import RSNAPneumonia as config
from datasets import RSNAPneumonia,build_transform
from networks import resnet50, ConvMixer
import time
from utils import AverageMeter,save_model, computeAUROC
import numpy as np
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--batch_size', type=int, default=512,  help='batch_size')
parser.add_argument('--num_workers', type=int, default=10, help='num of workers to use')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight', dest='weight', default=None)
parser.add_argument('--gpu', dest='gpu', default="0,1,2,3", type=str, help="gpu index")
parser.add_argument('--run', type=int, default=1, help='multiple runs')
parser.add_argument('--backbone', dest='backbone', default="convmixer12", type=str, help="backbone network")
parser.add_argument('--test', action='store_true', default=False)

args = parser.parse_args()

assert args.backbone in ['resnet50', 'convmixer12','convmixer36','resnet50_imgnet','resnet50_nih14',
                         'resnet50_dira_barlowtwins','resnet50_dira_mocov2','resnet50_dira_simsiam',
                         'resnet50_caid_barlowtwins','resnet50_caid_mocov2','resnet50_caid_simsiam',
                         'resnet50_barlowtwins','resnet50_simsiam','resnet50_mocov2']

def build_model(conf):

    if conf.backbone =="resnet50":
        model = resnet50(num_classes= 3)
    if conf.backbone =="resnet50_imgnet":
        model = resnet50(num_classes= 1000, pretrained=True)
        kernelCount = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernelCount, 3), nn.Sigmoid())
        # init the fc layer
        model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
        model.fc[0].bias.data.zero_()


    if conf.backbone =="resnet50_nih14":
        model = resnet50(num_classes= 3)
        if conf.server == "lab":
            checkpoint = torch.load("/mnt/dfs/jpang12/downstream_2d/pretrained_weights/resnet50_random_nih14.pth", map_location='cpu')
        elif conf.server == "agave":
            checkpoint = torch.load("/data/jliang12/jpang12/resnet50_random_nih14.pth", map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)
        print(msg, file=conf.log_writter)

    if conf.backbone =="resnet50_dira_barlowtwins":
        model = resnet50(num_classes= 3)
        checkpoint = torch.load("/data/jliang12/jpang12/DiRA/DiRA_barlowtwins.pth", map_location='cpu')
        state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)
        print(msg, file=conf.log_writter)

    if conf.backbone =="resnet50_dira_mocov2":
        model = resnet50(num_classes= 3)
        checkpoint = torch.load("/data/jliang12/jpang12/DiRA/DiRA_moco-v2.pth", map_location='cpu')
        state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)
        print(msg, file=conf.log_writter)

    if conf.backbone =="resnet50_dira_simsiam":
        model = resnet50(num_classes= 3)
        checkpoint = torch.load("/data/jliang12/jpang12/DiRA/DiRA_simsiam.pth", map_location='cpu')
        state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)
        print(msg, file=conf.log_writter)


    if conf.backbone =="resnet50_caid_barlowtwins":
        model = resnet50(num_classes= 3)
        checkpoint = torch.load("/data/jliang12/jpang12/CAiD/CAiD_barlowtwins.pth", map_location='cpu')
        state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)
        print(msg, file=conf.log_writter)
    if conf.backbone =="resnet50_caid_mocov2":
        model = resnet50(num_classes= 3)
        checkpoint = torch.load("/data/jliang12/jpang12/CAiD/CAiD_moco-v2.pth", map_location='cpu')
        state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)
        print(msg, file=conf.log_writter)
    if conf.backbone =="resnet50_caid_simsiam":
        model = resnet50(num_classes= 3)
        checkpoint = torch.load("/data/jliang12/jpang12/CAiD/CAiD_simsiam.pth", map_location='cpu')
        state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)
        print(msg, file=conf.log_writter)

    if conf.backbone =="resnet50_barlowtwins":
        model = resnet50(num_classes= 3)
        checkpoint = torch.load("/data/jliang12/jpang12/CAiD/barlowtwins.pth", map_location='cpu')
        state_dict = {k.replace("encoder.", ""): v for k, v in checkpoint.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)
        print(msg, file=conf.log_writter)


    if conf.backbone =="resnet50_simsiam":
        model = resnet50(num_classes= 3)
        checkpoint = torch.load("/data/jliang12/jpang12/CAiD/simsiam.pth.tar", map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)
        print(msg, file=conf.log_writter)

    if conf.backbone =="resnet50_mocov2":
        model = resnet50(num_classes= 3)
        checkpoint = torch.load("/data/jliang12/jpang12/CAiD/moco-v2.pth.tar", map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
        state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)
        print(msg, file=conf.log_writter)


    if conf.weight is not None:
        checkpoint = torch.load(conf.weight, map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(state_dict)
        print("Loading pretrianed weight from: {}".format(conf.weight), file=conf.log_writter)



    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = torch.optim.Adam(parameters, lr=conf.lr)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        model = model.cuda()
        cudnn.benchmark = True

    return model, optimizer

def train(train_loader, model, optimizer , epoch, conf):

    model.train(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    criteria = nn.CrossEntropyLoss()

    for idx, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        bsz = input.shape[0]

        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        pred = model(input)
        loss = criteria(pred, target)

        losses.update(loss.item(),bsz)
        optimizer.zero_grad()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lr {lr}\t'
                  'loss {ttloss.val:.3f} ({ttloss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, ttloss=losses, lr = optimizer.param_groups[0]['lr'] ), file=conf.log_writter)
            conf.log_writter.flush()
            if conf.debug_mode:
                break
    return losses.avg




def valid(valid_loader, model, epoch, conf):

    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    criteria = nn.CrossEntropyLoss()
    with torch.no_grad():
        for idx, (input, target) in enumerate(valid_loader):

            data_time.update(time.time() - end)
            bsz = input.shape[0]

            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            pred = model(input)
            loss = criteria(pred, target)

            losses.update(loss.item(),bsz)



            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % 10 == 0:
                print('Valid: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {ttloss.val:.3f} ({ttloss.avg:.3f})'.format(
                       epoch, idx + 1, len(valid_loader), batch_time=batch_time,
                       data_time=data_time, ttloss=losses ), file=conf.log_writter)
                conf.log_writter.flush()
                if conf.debug_mode:
                    break
        return losses.avg


def test(test_loader, conf, best_file_path,model):

    if best_file_path is not None:
        conf.weight = best_file_path
        model,_ = build_model(conf)

    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)


            # compute output
            with torch.cuda.amp.autocast():
                if len(images.size()) == 4:
                    bs, c, h, w = images.size()
                    n_crops = 1
                elif len(images.size()) == 5:
                    bs, n_crops, c, h, w = images.size()
                with torch.no_grad():
                    varInput = torch.autograd.Variable(images.view(-1, c, h, w).cuda())
                    out = model(varInput)
                    out = torch.softmax(out,dim=1)
                    outMean = out.view(bs, n_crops, -1).mean(1)
                    _, predicted = torch.max(outMean, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %',file=conf.log_writter)
    conf.log_writter.flush()


def main(conf):

    model, optimizer = build_model(conf)
    train_dataset = RSNAPneumonia(conf.data_root, conf.train_list, build_transform())
    valid_dataset = RSNAPneumonia(conf.data_root, conf.valid_list, build_transform(mode="validation"))
    test_dataset = RSNAPneumonia(conf.data_root, conf.test_list, build_transform(mode="test"))
    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_valid = torch.utils.data.RandomSampler(valid_dataset)
    sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,pin_memory=True,sampler=sampler_train)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,pin_memory=True,sampler=sampler_valid)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size//2, num_workers=conf.num_workers, pin_memory=True,sampler=sampler_test)

    print(model, file=conf.log_writter)
    max_val_loss = 100000
    no_improvement = 0
    best_file_path = None

    for epoch in range(1, conf.epochs + 1):
        time1 = time.time()

        loss = train(train_loader, model, optimizer , epoch, conf)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1),file = conf.log_writter)

        # tensorboard logger
        print('loss: {}@Epoch: {}'.format(loss,epoch),file = conf.log_writter)
        print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'],epoch),file = conf.log_writter)
        conf.log_writter.flush()

        valid_loss = valid(valid_loader, model, epoch,conf)
        if max_val_loss > valid_loss:
            print("Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model"
                  .format(epoch, max_val_loss, valid_loss),file = conf.log_writter)
            max_val_loss = valid_loss
            save_file = os.path.join(conf.model_path, 'ckpt.pth')
            best_file_path = save_file
            save_model(model, optimizer, conf,epoch,save_file=save_file)
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement> conf.patience:
            print("Model does not imporove from {:.5f} for {} epochs. Early Stop".format(max_val_loss,conf.patience),file = conf.log_writter)
            break

        if conf.debug_mode:
            break

    test(test_loader,conf,best_file_path,model)
    conf.log_writter.flush()




if __name__ == '__main__':


    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    conf = config(args)

    if args.test and conf.weight is not None:
        test_dataset = RSNAPneumonia(conf.data_root, conf.test_list, build_transform(mode="test"))
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size//2, num_workers=conf.num_workers, pin_memory=True, sampler=sampler_test)
        test(test_loader,conf,conf.weight,None)
        exit(0)
    elif args.test and conf.weight is None:
        print("Test mode, but no weight path given, exit!",file = conf.log_writter)
        exit(0)



    conf.display()
    main(conf)

#>> AUC = [0.7808,0.8823,0.8321,0.7049,0.8247,0.7703,0.7302,0.8697,0.7503,0.8494, 0.9124,0.8272,0.7862,0.8637]   >>Mean AUC = 0.8132   swav tencrop

#>> AUC = [0.7808,0.8823,0.8321,0.7049,0.8247,0.7703,0.7302,0.8697,0.7503,0.8494, 0.9124,0.8272,0.7862,0.8637]   >>Mean AUC = 0.8132   swav NO - tencrop
