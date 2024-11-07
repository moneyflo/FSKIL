import numpy as np
import torch
from torch.utils.data import Dataset
def set_up_datasets(args):
    if args.dataset == 'gsc2':
        import dataloader.gsc2.gsc2 as Dataset
        args.base_class = 20
        args.num_classes = 35
        args.way = 3
        args.shot = 5
        args.sessions = 6

    args.Dataset=Dataset
    return args

def get_dataloader(args, session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    
    if args.dataset == 'gsc2':
        trainset = args.Dataset.GSC2(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.GSC2(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader


def get_new_dataloader(args, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'

    if args.dataset == 'gsc2':
        trainset = args.Dataset.GSC2(root=args.dataroot, train=True,
                                    index_path=txt_path)
        
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'gsc2':
        testset = args.Dataset.GSC2(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list

