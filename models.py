import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
import torchvision.models as tvmodels

class clip_a2v(nn.Module):
    def __init__(self, args):
        super(clip_a2v, self).__init__()

        if args.single_part:
            self.module_trained = [args.single_part]
        else:
            self.module_trained = ['verb', 'foot', 'leg', 'hip', 'hand', 'arm', 'head']

        self.num_fc= 512
        self.num_clip = 512
        self.num_bert = 1536

        self.dropout_rate = 0.5

        self.pasta_names = ['foot', 'leg', 'hip', 'hand', 'arm', 'head']
        self.num_pastas  = [16, 15, 6, 34, 8, 14]
        self.pasta_range_starts = [ 0, 16, 31, 37, 71, 79]
        self.pasta_range_ends   = [16, 31, 37, 71, 79, 93]
        self.characteristics = {
		    'foot'	: {'standing on something', 'treading on something', 'walking with something', 'walking to something', 'running with something', 'running to something', 'dribbling', 'kicking something', 'jumping down', 'jumping with something', 'walking away', 'crawling', 'dancing', 'falling down', 'playing martial art', 'doing nothing'},
		    'leg'	: {'walking with something', 'walking to something', 'running with something', 'running to something', 'jumping with something', 'close with something', 'straddling something', 'jumping down', 'walking away', 'bending', 'kneeling', 'crawling', 'dancing', 'playing martial art', 'doing nothing'},
		    'hip'	: {'sitting on something', 'sitting in something', 'sitting beside something', 'close with something', 'bending', 'doing nothing'}, 
		    'hand'	: {'holding something', 'carrying something', 'reaching for something', 'touching', 'putting on something', 'twisting', 'wearing something', 'throwing something', 'throwing out something', 'writting on something', 'pointing with something', 'pointing to something', 'using something to point to something', 'pressing something', 'squeezing something', 'scratching something', 'pinching something', 'gesturing to something', 'pushing something', 'pulling something', 'pulling with something', 'washing something', 'washing with something', 'holding something in both hands', 'lifting something', 'raising something', 'feeding', 'cutting with something', 'catching with something', 'pouring something into something', 'crawling ', 'dancing', 'playing martial art', 'doing nothing'},
		    'arm'	: {'carrying something', 'close to something', 'hugging', 'swinging', 'crawling', 'dancing', 'playing martial art', 'doing nothing'},
		    'head'	: {'eating', 'inspecting', 'talking with something', 'talking to something', 'closing with something', 'kissing', 'put somthing over', 'licking', 'blowing', 'drinking with something', 'smelling', 'wearing', 'listening', 'doing nothing'}
        }
        self.num_verbs = 157

        self.focal_alpha = 3
        self.focal_gamma = 1
        self.loss_weight = 1.0

        # Load the model
        self.device = args.device
        # self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        # self.pasta_language_matrix  = torch.from_numpy(np.load("util/pasta_language_matrix.npy")).cuda()
        self.model = tvmodels.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, self.num_clip)
        torch.nn.init.eye(self.model.fc.weight)

        # CLIP feature to PaSta feature.
        self.clip2pasta = nn.ModuleList(
                                            [
                                                nn.Sequential(
                                                    nn.Linear(self.num_clip, self.num_fc),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(self.num_fc, self.num_fc),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(self.num_fc, self.num_pastas[pasta_idx])
                                                )
                                                for pasta_idx in range(len(self.pasta_names))
                                            ]
                                        )
        
        # Verb classifier.
        self.verb_cls_scores = nn.Sequential(
                        nn.Linear(self.num_fc * (len(self.num_pastas) + 1), self.num_fc),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.num_fc, self.num_fc),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.num_fc, self.num_verbs)
                    ) 

        # Freeze the useless params.			
        for pasta_idx in range(len(self.pasta_names)):
            for p in self.clip2pasta[pasta_idx].parameters():
                p.requires_grad = self.pasta_names[pasta_idx] in self.module_trained

        for p in self.verb_cls_scores.parameters():
            p.requires_grad = 'verb' in self.module_trained

    def cal_loss(self, logit, label, name):
        loss = F.binary_cross_entropy_with_logits(logit, label, reduction='none')
        pt = torch.exp(-loss)
        loss = self.focal_alpha * (1 - pt)**self.focal_gamma * loss
        loss = loss.mean()
        loss = loss * self.loss_weight

        pred = (logit.detach().sigmoid() > 0.5)
        correct = (pred == label).sum()
        acc = 100. * correct / pred.numel()

        out = {
            f'loss_{name}': loss,
            f'accuray_{name}': acc
        }
        return out
    
    def get_loss(self, s_parts, s_verb, gt):
        losses = dict()
        for i, name in enumerate(self.pasta_names):
            if name not in self.module_trained:
                continue
            gt_label = gt[name].to(self.device)
            loss = self.cal_loss(s_parts[i], gt_label, name)
            losses.update(loss)
        
        if 'verb' in self.module_trained:
            gt_label = gt['verb'].to(self.device)
            loss = self.cal_loss(s_verb, gt_label, 'verb')
            losses.update(loss)
        return losses

    def cal_map(self, gt, pred):
        assert gt.shape[0] == pred.shape[0]

        hit = []
        idx = np.argsort(pred)[::-1]

        for i in idx:
            if gt[i] == 1:
                hit.append(1)
            else:
                hit.append(0)

        npos = gt.sum()

        bottom  = np.array(range(len(hit))) + 1
        hit     = np.cumsum(hit)
        rec     = hit / npos
        prec    = hit / bottom
        ap = 0.0
        for i in range(11):
            mask = rec >= (i / 10.0)
            if np.sum(mask) > 0:
                ap += np.max(prec[mask]) / 11.0
        return ap 
    
    def get_map(self, pred_pasta, pred_verb, gt):
        output = {}

        for i, name in enumerate(self.pasta_names):
            aps = []
            for pasta_idx in range(self.pasta_range_starts[i], self.pasta_range_ends[i]):
                pasta_score = pred_pasta[:, pasta_idx]
                gt_pasta = gt[name][:, pasta_idx - self.pasta_range_starts[i]]
                if gt_pasta.sum() < 1:
                    continue
                ap = self.cal_map(gt_pasta, pasta_score)
                aps.append(ap)
            aps = np.array(aps)
            map = aps.mean()
            out = {
                f'mAP_{name}': map
            }
            output.update(out)
    
        vaps = []
        for verb_idx in range(self.num_verbs):
            verb_score = pred_verb[:, verb_idx]
            gt_verb = gt['verb'][:, verb_idx]
            if gt_verb.sum() < 1:
                continue
            vap = self.cal_map(gt_verb, verb_score)
            vaps.append(vap)
        vaps = np.array(vaps)
        vmap = vaps.mean()
        vout = {
            f'mAP_verb': vmap
        }
        output.update(vout)

        return output

    def forward(self, image):
            
        f_parts = []
        s_parts = []
        p_parts = []

        with torch.no_grad():
            # image_features = self.model.encode_image(image).float().to(self.device)
            image_features = self.model(image).float().to(self.device)
        f_parts.append(image_features)

        for class_id, classes in enumerate(self.pasta_names):
            '''
            # Image and Text Feature
            text_inputs = []
            for characteristics in self.characteristics[classes]:
                text_inputs.append(clip.tokenize(f"the person's {classes} is {characteristics}"))
            text_inputs = torch.cat(text_inputs,dim=0).to(self.device)

            with torch.no_grad():
                text_features_i = self.model.encode_text(text_inputs).float()

            image_features_i = self.clip2pasta[class_id](image_features)  # Note: need to modify the output dim of the mlp

            image_features_i = image_features_i / image_features_i.norm(dim=-1, keepdim=True)
            text_features_i = text_features_i / text_features_i.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features_i @ text_features_i.T)

            s_part = similarity - torch.mean(similarity)
            p_part = torch.sigmoid(s_part)

            f_parts.append(image_features_i)
            s_parts.append(s_part)
            p_parts.append(p_part)
            '''
            
            # Only Image Feature
            x = image_features
            for i in range(len(self.clip2pasta[class_id])):
                x = self.clip2pasta[class_id][i](x)
                if i == len(self.clip2pasta[class_id]) - 2:
                    f_parts.append(x)
                
            s_part = x
            p_part = torch.sigmoid(s_part)

            s_parts.append(s_part)
            p_parts.append(p_part)
            
        f_pasta = torch.cat(f_parts, 1)
        p_pasta = torch.cat(p_parts, 1)

        # f_pasta_language = torch.matmul(p_pasta, self.pasta_language_matrix)

        # f_pasta = torch.cat([f_pasta_visual, f_pasta_language], 1)  # Note: need to modify the input dim of the mlp 
		
        s_verb = self.verb_cls_scores(f_pasta)
        p_verb = torch.sigmoid(s_verb)

        if self.training:
            return s_parts, s_verb
        else:
            return p_pasta, p_verb
        