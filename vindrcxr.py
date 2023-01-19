import torch.nn as nn
import torch
import argparse
import os
from config import VinDRCXR as config
from datasets import VinDrCXR,build_transform
from networks import resnet50, ConvMixer
import time
from utils import AverageMeter,save_model, computeAUROC
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--batch_size', type=int, default=512,  help='batch_size')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--weight', dest='weight', default=None)
parser.add_argument('--gpu', dest='gpu', default="0,1,2,3", type=str, help="gpu index")
parser.add_argument('--run', type=int, default=1, help='multiple runs')
parser.add_argument('--backbone', dest='backbone', default="resnet50_imgnet", type=str, help="backbone network")
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--gt', type=str, default='global_vote', choices=['global_vote','global_one','global_whole',
                                                                      'local_vote','local_vote', 'local_whole'])

args = parser.parse_args()

assert args.backbone in ['resnet50', 'convmixer12','convmixer36','resnet50_imgnet']

def build_model(conf):
    if conf.backbone =="resnet50":
        model = resnet50(num_classes= conf.num_classes)
    if conf.backbone =="resnet50_imgnet":
        model = resnet50(num_classes= 1000, pretrained=True)
        kernelCount = model.fc.in_features
        #model.fc = nn.Sequential(nn.Linear(kernelCount, conf.num_classes), nn.Sigmoid())
        model.fc = nn.Sequential(nn.Linear(kernelCount, conf.num_classes))
        # init the fc layer
        model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
        model.fc[0].bias.data.zero_()

    elif conf.backbone =="convmixer12":
        model = ConvMixer(768, 12, patch_size=16, n_classes=conf.num_classes)
    elif conf.backbone =="convmixer36":
        model = ConvMixer(768, 36, patch_size=16, n_classes=conf.num_classes)


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
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=conf.patience // 2, mode='min',
                                     threshold=0.0001, min_lr=0, verbose=True)
    return model, optimizer,lr_scheduler

def train(train_loader, model, optimizer , epoch, conf):

    model.train(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    criteria = nn.BCELoss()

    for idx, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        bsz = input.shape[0]

        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        pred = torch.sigmoid(model(input))
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

    criteria = nn.BCELoss()
    with torch.no_grad():
        for idx, (input, target) in enumerate(valid_loader):

            data_time.update(time.time() - end)
            bsz = input.shape[0]

            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            pred = torch.sigmoid(model(input))
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
        model,_,_ = build_model(conf)

    y_test = torch.FloatTensor().cuda()
    p_test = torch.FloatTensor().cuda()

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True).float()
                target = target.cuda(non_blocking=True).float()


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
                    out = torch.sigmoid(out)
                    outMean = out.view(bs, n_crops, -1).mean(1)
                    p_test = torch.cat((p_test, outMean.data), 0)

                    target = target.type_as(out)
                    y_test = torch.cat((y_test, target), 0)
                # p_test = torch.cat((p_test, output), 0)


    aurocIndividual = computeAUROC(y_test, p_test, conf.num_classes)
    print("Individual Diseases:", file=conf.log_writter)
    print(">> AUC = {}".format(np.array2string(np.array(aurocIndividual), precision=4, separator=',')), file=conf.log_writter)

    aurocMean = np.array(aurocIndividual).mean()
    print(">>Mean AUC = {:.4f}".format(aurocMean), file=conf.log_writter)


def main(conf):

    model, optimizer,lr_scheduler = build_model(conf)



    train_dataset = VinDrCXR(conf.data_root, conf.train_list, build_transform())
    valid_dataset = VinDrCXR(conf.data_root, conf.valid_list, build_transform(mode="validation"))
    test_dataset = VinDrCXR(conf.data_root, conf.test_list, build_transform(mode="test"))
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

        lr_scheduler.step(valid_loss)

        if max_val_loss > valid_loss:
            print("Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model"
                  .format(epoch, max_val_loss, valid_loss),file = conf.log_writter)
            max_val_loss = valid_loss
            save_file = os.path.join(conf.model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
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
        test_dataset = VinDrCXR(conf.data_root, conf.test_list, build_transform(mode="test"))
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size, num_workers=conf.num_workers, pin_memory=True, sampler=sampler_test)
        test(test_loader,conf,conf.weight,None)
        exit(0)
    elif args.test and conf.weight is  None:
        print("Test mode, but no weight path given, exit!",file = conf.log_writter)
        exit(0)

    conf.display()
    main(conf)



#>> AUC = [0.7808,0.8823,0.8321,0.7049,0.8247,0.7703,0.7302,0.8697,0.7503,0.8494, 0.9124,0.8272,0.7862,0.8637]   >>Mean AUC = 0.8132   swav tencrop

#>> AUC = [0.7808,0.8823,0.8321,0.7049,0.8247,0.7703,0.7302,0.8697,0.7503,0.8494, 0.9124,0.8272,0.7862,0.8637]   >>Mean AUC = 0.8132   swav NO - tencrop
