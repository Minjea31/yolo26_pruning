from compress import GM
import torch.nn.utils.prune as prune
import time
import torch.nn as nn
import torch
from ultralytics.nn.modules import *
import yaml
from ultralytics import YOLO


class PruneHandler():
    def __init__(self, model, compression_ratio, method, cfg_output_path, prune_type='ALL'):
        self.model = model
        self.ckpt = model.ckpt['model']
        self.model.cpu()
        self.cr = compression_ratio
        self.method = method
        self.cfg_output_path = cfg_output_path
        self.model.to('cpu')  # cuda cannot convert to numpy
        self.remain_index_out = {}
        self.prune_type = prune_type

    def prune(self):
        backbone_layers = [str(i) for i in range(11)]
        detect_layers = ['23']

        def in_layers(name, layers):
            return any(name == n or name.startswith(n + '.') for n in layers)

        if self.method == 'GM':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, detect_layers):
                        if isinstance(module, nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self.cr, dim=0)
                            mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.l1_unstructured(module, name='weight', amount=self.cr, importance_scores=mask)
                            prune.l1_unstructured(module, name='bias', amount=self.cr, importance_scores=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, backbone_layers + detect_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self.cr, dim=0)
                            mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.l1_unstructured(module, name='weight', amount=self.cr, importance_scores=mask)
                            prune.l1_unstructured(module, name='bias', amount=self.cr, importance_scores=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if in_layers(name, backbone_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            GM.gm_structured(module, name='weight', amount=self.cr, dim=0)
                            mask = torch.norm(module.weight_mask, 1, dim=(1, 2, 3))
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.l1_unstructured(module, name='weight', amount=self.cr, importance_scores=mask)
                            prune.l1_unstructured(module, name='bias', amount=self.cr, importance_scores=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

        elif self.method == 'L1':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, detect_layers):
                        if isinstance(module, nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, backbone_layers + detect_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if in_layers(name, backbone_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=1, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

        elif self.method == 'L2':
            if self.prune_type == 'ALL':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, detect_layers):
                        if isinstance(module, nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        if isinstance(module, nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'H':
                for name, module in self.model.model.model.named_modules():
                    if not in_layers(name, backbone_layers + detect_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')
            elif self.prune_type == 'B':
                for name, module in self.model.model.model.named_modules():
                    if in_layers(name, backbone_layers):
                        if isinstance(module, torch.nn.Conv2d):
                            prune.ln_structured(module, name='weight', amount=self.cr, n=2, dim=0)
                            mask = torch.where(torch.norm(module.weight_mask, p=2, dim=(1, 2, 3)) != 0, 1, 0)
                            prune.remove(module, 'weight')
                        elif isinstance(module, torch.nn.BatchNorm2d):
                            prune.custom_from_mask(module, name='weight', mask=mask)
                            prune.custom_from_mask(module, name='bias', mask=mask)
                            prune.remove(module, 'weight')
                            prune.remove(module, 'bias')

    def reconstruct(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.training = False

        detect_in_channels = []
        concat = {}
        remain_out_channels = [0, 1, 2]
        for name, module in self.model.model.model.named_modules():
            if isinstance(module, Conv):
                if name in ['0', '1', '3', '5', '7', '17', '20']:
                    num = int(name)
                    offset = self.model.model.model[num].conv.weight.shape[0]
                    remain_out_channels = module.recon(remain_out_channels)
                    if name in ['17', '20']:
                        concat[name] = [remain_out_channels, offset]

            elif isinstance(module, C3k2):
                first_m = module.m[0]

                if hasattr(first_m, "cv1") and hasattr(first_m, "cv3"):
                    c3k = True
                else:
                    c3k = False
                num = int(name)
                offset = self.model.model.model[num].cv2.conv.weight.shape[0]
                remain_out_channels = module.recon(remain_out_channels, c3k)
                if name in ['16', '19', '22']:
                    detect_in_channels.append(remain_out_channels)
                elif name in ['4', '6', '13']:
                    concat[name] = [remain_out_channels, offset]

            elif isinstance(module, SPPF):
                num = int(name)
                offset = self.model.model.model[num].cv2.conv.weight.shape[0]
                remain_out_channels = module.recon(remain_out_channels)
                concat[name] = [remain_out_channels, offset]

            elif isinstance(module, C2PSA):
                num = int(name)
                offset = self.model.model.model[num].cv2.conv.weight.shape[0]
                remain_out_channels = module.recon(remain_out_channels)
                concat[name] = [remain_out_channels, offset]

            elif isinstance(module, Detect):
                remain_out_channels = module.recon(detect_in_channels)

            elif isinstance(module, Concat):
                if name == '12':
                    concat['6'][0] = [x + concat['10'][1] for x in concat['6'][0]]
                    remain_out_channels = concat['10'][0] + concat['6'][0]
                elif name == '15':
                    concat['4'][0] = [x + concat['13'][1] for x in concat['4'][0]]
                    remain_out_channels = concat['13'][0] + concat['4'][0]
                elif name == '18':
                    concat['13'][0] = [x + concat['17'][1] for x in concat['13'][0]]
                    remain_out_channels = concat['17'][0] + concat['13'][0]
                elif name == '21':
                    concat['10'][0] = [x + concat['20'][1] for x in concat['10'][0]]
                    remain_out_channels = concat['20'][0] + concat['10'][0]

    def model_to_yaml(self):
        from_ = -1
        repeats = 1
        yaml_dict = {}
        detect = self.model.model.model[-1]
        yaml_dict["nc"] = 8
        yaml_dict["end2end"] = detect.end2end
        yaml_dict["reg_max"] = detect.reg_max
        yaml_dict["scales"] = {'prune': [1, 1, 1024]}
        yaml_dict["backbone"] = []
        yaml_dict["head"] = []

        for name, module in self.model.ckpt['model'].model.named_modules():
            if isinstance(module, Conv):
                if name in ['0', '1', '3', '5', '7', '17', '20']:
                    args = [module.conv.out_channels, module.conv.kernel_size[0], module.conv.stride[0]]
                    layer = [from_, repeats, type(module).__name__, args]
                    if name in ["17", "20"]:
                        yaml_dict["head"].append(layer)
                    else:
                        yaml_dict["backbone"].append(layer)

            elif isinstance(module, C3k2):
                # 직접 전달 하게 함.
                attn = (
                    len(module.m) > 0
                    and isinstance(module.m[0], nn.Sequential)
                    and len(module.m[0]) > 1
                    and module.m[0][1].__class__.__name__ == "PSABlock"
                )
                if name in ['2', '4']:
                    c3k = False
                    e = 0.25
                else:
                    c3k = True
                    e = 0.5               
                args = [module.cv2.conv.out_channels, c3k, e]
                if attn:
                    args.append(True)
                layer = [from_, len(module.m), type(module).__name__, args]
                if name in ['13', '16', '19', '22']:
                    yaml_dict["head"].append(layer)
                else:
                    yaml_dict["backbone"].append(layer)

            elif isinstance(module, C2PSA):
                # import pdb; pdb.set_trace()

                args = [module.cv2.conv.out_channels]
                layer = [from_, len(module.m), type(module).__name__, args]
                yaml_dict["backbone"].append(layer)

            elif isinstance(module, SPPF):
                k = 5
                n = 3
                shortcut = True

                args = [module.cv2.conv.out_channels, k, n, shortcut]
                layer = [from_, repeats, type(module).__name__, args]

                yaml_dict["backbone"].append(layer)

            elif isinstance(module, Detect):
                args = [yaml_dict["nc"]]
                layer = [[16, 19, 22], repeats, type(module).__name__, args]
                yaml_dict["head"].append(layer)

            elif isinstance(module, Concat):
                args = [1]
                if name == '12':
                    # import pdb; pdb.set_trace()
                    layer = [[from_, 6], repeats, type(module).__name__, args]
                elif name == '15':
                    layer = [[from_, 4], repeats, type(module).__name__, args]
                elif name == '18':
                    layer = [[from_, 13], repeats, type(module).__name__, args]
                elif name == '21':
                    layer = [[from_, 10], repeats, type(module).__name__, args]
                yaml_dict["head"].append(layer)

            elif isinstance(module, nn.Upsample):
                args = ['None', module.scale_factor, module.mode]
                layer = [from_, repeats, "nn." + type(module).__name__, args]
                yaml_dict["head"].append(layer)
        yaml_str = yaml.dump(yaml_dict)
        with open(f'./{self.cfg_output_path}/best_model_prune.yaml', "w") as file:
            file.write(yaml_str)

    def compress_yolo26(self):
        print('Pruning...')
        start = time.time()
        self.prune()
        self.reconstruct()
        torch.save(self.model, f'./{self.cfg_output_path}/best_model_prune.pt')
        self.model_to_yaml()
        print('Done')
        # import pdb; pdb.set_trace()
        print(f'time : {time.time() - start}')
        return YOLO(f'./{self.cfg_output_path}/best_model_prune.yaml').load(f'./{self.cfg_output_path}/best_model_prune.pt')
