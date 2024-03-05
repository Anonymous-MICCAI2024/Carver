import sys
sys.path.append('..')
import torch
from model.encoder import resnet,swin_transformer,simplenet,trans_plus_conv


moco_weight_path = {
    'resnet18':None,
    'resnet50':'/staff/shijun/torch_projects/Med_Seg/cls/ckpt/HaN_GTV/v3.0-half/GTV/fold1/epoch=21-train_loss=0.16699-train_acc=0.92985-train_f1=0.92222-val_loss=0.21024-val_acc=0.92394-val_f1=0.91263.pth'
}


def build_encoder(arch='resnet18', weights=None, **kwargs):
        
    arch = arch.lower()
    
    if arch.startswith('resnet'):
        backbone = resnet.__dict__[arch](classification=False,**kwargs)
    elif arch.startswith('swin_transformer'):
        backbone = swin_transformer.__dict__[arch](classification=False,**kwargs)
    elif arch.startswith('simplenet'):
        backbone = simplenet.__dict__[arch](**kwargs)
    elif arch.startswith('swinplus'):
        backbone = trans_plus_conv.__dict__[arch](classification=False,**kwargs)
    else:
        raise Exception('Architecture undefined!')

    if weights is not None and isinstance(moco_weight_path[arch], str):
        print('Loading weights for backbone')
        state_dict=torch.load(moco_weight_path[arch], map_location=lambda storage, loc: storage)['state_dict']
        for key in state_dict.keys():
            if '_tmp_' in key:
                # print(key)
                state_dict[key.replace('_tmp_','')] = state_dict[key]
                del state_dict[key]
        msg = backbone.load_state_dict(state_dict, strict=False)
        print(msg)
    
    return backbone



if __name__ == '__main__':

    import os 
    from torchsummary import summary
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # net = build_encoder('swin_transformer',n_channels=1)
    net = build_encoder('resnet50',n_channels=1,weights='moco')
    summary(net.cuda(),input_size=(1,256,256),batch_size=1,device='cuda')
    # net = build_encoder('simplenet',n_channels=1)
    net = net.cuda()
    net.train()
    input = torch.randn((1,1,256,256)).cuda()
    output = net(input)
    for item in output:
        print(item.size())

    # import sys
    # sys.path.append('..')
    # from utils import count_params_and_macs
    # count_params_and_macs(net.cuda(),(1,3,256,256))