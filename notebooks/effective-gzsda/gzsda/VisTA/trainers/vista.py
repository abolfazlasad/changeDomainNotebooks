import os
import datetime
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from sklearn.cluster import KMeans, MiniBatchKMeans

from utils.clip_custom import *
from utils.utils import *
from utils.data_manager import DataManager

class PromptLearner(nn.Module):
    def __init__(self, cfg, class_names, clip_model, text_attr, n_ctx):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        dtype = clip_model.dtype
        self.clip_model = clip_model
        self.class_names = [name.replace('_', ' ') for name in class_names]
        n_cls = len(self.class_names)
        self.n_cls = n_cls 
        self.dtype = dtype
        self.attr_L = cfg.TRAINER.VISTA.L # number of selected attribute per image
        self.attr_N = cfg.TRAINER.VISTA.N # number of attributes in the dictionary
        self.attr_M = n_ctx # prompt length
        self.text_attr = text_attr  # learnable text attributes

        prompt_prefix =' '.join(['x'] * n_ctx * self.attr_L)
        prompts = [prompt_prefix + ' ' + name + '.' for name in self.class_names]

        # naive prompts used to create fixed text embeddings
        naive_prompt_prefix = "a " + cfg.DATASET.TARGET_DOMAINS[0].replace("_", " ") + " photo of a"
        naive_prompts = [naive_prompt_prefix + " " + name + "." for name in self.class_names]

        self.name_lens = [len(_tokenizer.encode(name)) for name in self.class_names]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        naive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in naive_prompts])

        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)
            # encoded naive prompts (used for regularization)
            self.fixed_embeddings = clip_model.encode_text(naive_tokenized_prompts).type(self.dtype)
            
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + (n_ctx * self.attr_L):,:])
        self.register_buffer('cls_token_suffix', embedding[:, 1 + n_ctx:,:])

        # tokenized "no-class" prompt (only prefix + '.'), used in only_prefix()
        nc_prompts = [prompt_prefix + '.']
        nc_tokenized_prompts = torch.cat([clip.tokenize(p) for p in nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(nc_tokenized_prompts).type(self.dtype)
        self.register_buffer('nc_token_prefix', embedding[:, :1,:])
        self.register_buffer('nc_token_suffix', embedding[:, 1 + n_ctx:,:])

        for param in self.clip_model.parameters():
            param.requires_grad = False

    @autocast()
    def forward(self, indices_s, indices_t):
        # construct prompt embeddings for selected attribute indices of source and target domains.
        if indices_s != None:
            batch_s = indices_s.shape[0]
            selected_prompts_s = self.text_attr[indices_s]
            ctx_s = selected_prompts_s.view(batch_s, self.attr_M * self.attr_L, self.ctx_dim)
        batch_t = indices_t.shape[0]
        selected_prompts_t = self.text_attr[indices_t + self.attr_N]
        ctx_t = selected_prompts_t.view(batch_t, self.attr_M * self.attr_L, self.ctx_dim)

        tokenized_prompts = self.tokenized_prompts.view(self.n_cls, -1)

        # prefix + ctx + suffix
        if indices_s != None:
            prefix_s = self.token_prefix.unsqueeze(0).repeat(batch_s,1,1,1)
            suffix_s = self.token_suffix.unsqueeze(0).repeat(batch_s,1,1,1)
            ctx_s = ctx_s.unsqueeze(1).repeat(1, self.n_cls, 1, 1)
            prompts_s = torch.cat([prefix_s, ctx_s, suffix_s],dim=2)
        prefix_t = self.token_prefix.unsqueeze(0).repeat(batch_t,1,1,1)
        suffix_t = self.token_suffix.unsqueeze(0).repeat(batch_t,1,1,1)
        ctx_t = ctx_t.unsqueeze(1).repeat(1, self.n_cls, 1, 1)
        prompts_t = torch.cat([prefix_t, ctx_t, suffix_t],dim=2)

        if indices_s != None:
            prompts_s = prompts_s.squeeze(2).view(batch_s*self.n_cls, -1, self.ctx_dim)
            tokenized_prompts_s = tokenized_prompts.unsqueeze(0).repeat(batch_s,1,1).view(batch_s*self.n_cls, -1)
        prompts_t = prompts_t.squeeze(2).view(batch_t*self.n_cls, -1, self.ctx_dim)
        tokenized_prompts_t = tokenized_prompts.unsqueeze(0).repeat(batch_t,1,1).view(batch_t*self.n_cls, -1)

        if indices_s == None:
            return prompts_t, tokenized_prompts_t
        else:
            return prompts_s, prompts_t, tokenized_prompts_s, tokenized_prompts_t

    def only_prefix(self, target):
        # construct prompts that no [CLS].
        ctx = self.text_attr[self.attr_N:] if target else self.text_attr[:self.attr_N]
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        nc_prompts = torch.cat([prefix, ctx, suffix],dim=1)
        return nc_prompts, nc_tokenized_prompts
    

class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model, vis_attr, text_attr, n_ctx):
        super().__init__()
        self.cfg = cfg
        self.n_class = len(class_names)
        self.attr_L = cfg.TRAINER.VISTA.L
        self.num_prompt = cfg.TRAINER.VISTA.N

        # text enoder
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_learner = PromptLearner(cfg, class_names, clip_model, text_attr, n_ctx=n_ctx)
        self.vis_attr = vis_attr

        # image encoder
        self.image_encoder = ImageEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale

        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder)
            self.image_encoder = nn.DataParallel(self.image_encoder)

    @autocast()
    def forward(self, image_s, image_t, test=False, cluster=False):
        if cluster:
            # return only image features for clustering
            with torch.no_grad():
                image_features_s = self.image_encoder(image_s)
                image_features_s = image_features_s / image_features_s.norm(dim=-1, keepdim=True)
                image_features_s = image_features_s.detach()
                image_features_t = self.image_encoder(image_t)
                image_features_t = image_features_t / image_features_t.norm(dim=-1, keepdim=True)
                image_features_t = image_features_t.detach()
            return image_features_s, image_features_t
        else:
            # standard forward
            if not test:
                image_features_s, last_feat_s = self.image_encoder(image_s, return_feat=True)
                image_features_s = image_features_s / image_features_s.norm(dim=-1, keepdim=True)
            image_features_t, last_feat_t = self.image_encoder(image_t, return_feat=True)
            image_features_t = image_features_t / image_features_t.norm(dim=-1, keepdim=True)
        
        if test:
            # target images pick top-L target visual attributes for inference
            image_features_t = image_features_t.detach()
            probability = image_features_t @ self.vis_attr[self.num_prompt:].t()
            _, indices = probability.topk(k=min(self.attr_L, probability.shape[1]), dim=1, largest=True)
            logit_scale = self.logit_scale.exp()

            text_prompt, tokenized_prompts = self.prompt_learner(None, indices)

            text_features = self.text_encoder(text_prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(image_features_t.shape[0], self.n_class, -1)

            image_features_t = image_features_t.unsqueeze(1)

            logits = logit_scale * (image_features_t * text_features).sum(-1) 
            return logits
            
        else:
            nc_prompts_s, nc_tokenized_prompts = self.prompt_learner.only_prefix(target = False)
            nc_prompts_t, nc_tokenized_prompts = self.prompt_learner.only_prefix(target = True)
            
            nc_text_features_s = self.text_encoder(nc_prompts_s, nc_tokenized_prompts)
            nc_text_features_s = nc_text_features_s / nc_text_features_s.norm(dim=-1, keepdim=True)
            nc_text_features_t = self.text_encoder(nc_prompts_t, nc_tokenized_prompts)
            nc_text_features_t = nc_text_features_t / nc_text_features_t.norm(dim=-1, keepdim=True)
 
            cosine_ss = (image_features_s @ nc_text_features_s.detach().T)
            cosine_st = (image_features_s @ nc_text_features_t.detach().T)
            cosine_ts = (image_features_t @ nc_text_features_s.detach().T)
            cosine_tt = (image_features_t @ nc_text_features_t.detach().T)
            
            image_features_s = image_features_s.detach()
            image_features_t = image_features_t.detach()

            probability_ss = (image_features_s @ self.vis_attr[:self.num_prompt].t())
            probability_tt = (image_features_t @ self.vis_attr[self.num_prompt:].t())

            _, indices_ss = probability_ss.topk(k=min(self.attr_L, probability_ss.shape[1]), dim=1, largest=True)
            _, indices_tt = probability_tt.topk(k=min(self.attr_L, probability_tt.shape[1]), dim=1, largest=True)

            indices_st = torch.zeros(image_features_s.shape[0], self.attr_L, dtype = torch.int)
            indices_ts = torch.zeros(image_features_t.shape[0], self.attr_L, dtype = torch.int)

            # compute grad-cam used to select cross-domain attributes
            grad_emaps_ss = grad_cam(cosine_ss[torch.arange(cosine_ss.shape[0]).unsqueeze(1), indices_ss], last_feat_s)
            grad_emaps_tt = grad_cam(cosine_tt[torch.arange(cosine_tt.shape[0]).unsqueeze(1), indices_tt], last_feat_t)
            grad_emaps_st = grad_cam(cosine_st, last_feat_s)
            grad_emaps_ts = grad_cam(cosine_ts, last_feat_t)

            # match heatmaps from different attribute dictionary
            for i in range(image_features_s.shape[0]):
                indices_st[i,:] = ppmcc(grad_emaps_ss[i,...], grad_emaps_st[i,...])
                indices_ts[i,:] = ppmcc(grad_emaps_tt[i,...], grad_emaps_ts[i,...])
            
            unique_indices_s, inverse_indices_s = torch.unique(torch.cat([indices_ss.to('cpu'), 
                                                                        indices_ts], dim=0), dim=0, return_inverse=True)
            unique_indices_t, inverse_indices_t = torch.unique(torch.cat([indices_tt.to('cpu'), 
                                                                        indices_st,], dim=0), dim=0, return_inverse=True)

            mapping_ss = inverse_indices_s[:torch.sort(indices_ss, dim=1)[0].size(0)]
            mapping_ts = inverse_indices_s[torch.sort(indices_ss, dim=1)[0].size(0):]
            mapping_tt = inverse_indices_t[:torch.sort(indices_tt, dim=1)[0].size(0)]
            mapping_st = inverse_indices_t[torch.sort(indices_tt, dim=1)[0].size(0):]

            text_prompt_s, text_prompt_t, tokenized_prompts_s, tokenized_prompts_t = self.prompt_learner(unique_indices_s, unique_indices_t)
            text_features_s = self.text_encoder(text_prompt_s, tokenized_prompts_s)
            text_features_t = self.text_encoder(text_prompt_t, tokenized_prompts_t)

            text_features_s = text_features_s / text_features_s.norm(dim=-1, keepdim=True)
            text_features_s = text_features_s.view(unique_indices_s.shape[0], self.n_class, -1)
            text_features_t = text_features_t / text_features_t.norm(dim=-1, keepdim=True)
            text_features_t = text_features_t.view(unique_indices_t.shape[0], self.n_class, -1)

            fixed_embeddings = self.prompt_learner.fixed_embeddings.unsqueeze(0).expand(mapping_ss.shape[0], -1, -1)
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)

            # loss to enhance the generalization capacity of source textual attributes
            loss_hp = F.l1_loss(text_features_s[mapping_ss], fixed_embeddings.to('cuda:0'), reduction='mean') 

            image_features_s = image_features_s.unsqueeze(1)
            image_features_t = image_features_t.unsqueeze(1)

            logit_scale = self.logit_scale.exp()

            # logits correspond to different image/text combinations
            logits_ss = logit_scale * (image_features_s * text_features_s[mapping_ss]).sum(-1)
            logits_st = logit_scale * (image_features_s * text_features_t[mapping_st]).sum(-1)
            logits_ts = logit_scale * (image_features_t * text_features_s[mapping_ts]).sum(-1)
            logits_tt = logit_scale * (image_features_t * text_features_t[mapping_tt]).sum(-1)

            dis_s = nc_text_features_s @ nc_text_features_s.permute(1, 0)
            dis_t = nc_text_features_t @ nc_text_features_t.permute(1, 0)

            # diversity loss among textual attributes
            loss_div = dis_s[~torch.eye(self.num_prompt, dtype=torch.bool)].abs().mean() \
                     + dis_t[~torch.eye(self.num_prompt, dtype=torch.bool)].abs().mean()

            return logits_ss, logits_st, logits_ts, logits_tt, loss_div, loss_hp

    @property
    def dtype(self):
        return self.image_encoder.module.conv1.weight.dtype if isinstance(self.image_encoder, nn.DataParallel) else self.image_encoder.conv1.weight.dtype


@TRAINER_REGISTRY.register()
class VISTA(TrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.VISTA.PREC in ["fp16", "fp32", "amp"]
    
    def build_data_loader(self):
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname
        self.dm = dm

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        device_ids = [int(gpu_id.strip()) for gpu_id in cuda_visible.split(',')]
        self.device = device_ids[0]
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.VISTA.PREC == "fp32" or cfg.TRAINER.VISTA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.attr_M = cfg.TRAINER.VISTA.M
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.text_projection.shape[1]
        vis_attr = torch.empty(2 * cfg.TRAINER.VISTA.N, vis_dim, dtype=clip_model.dtype)
        nn.init.normal_(vis_attr, std=0.02)
        text_attr = torch.empty(2 * cfg.TRAINER.VISTA.N, self.attr_M, ctx_dim, dtype=clip_model.dtype)
        nn.init.normal_(text_attr, std=0.02)
        vis_attr = nn.Parameter(vis_attr)
        text_attr = nn.Parameter(text_attr)
        self.model = CustomCLIP(cfg, classnames, clip_model, vis_attr, text_attr, self.attr_M)
        try:
            self.model.text_encoder.transformer.use_gradient_checkpoint = True 
        except:
            self.model.text_encoder.module.transformer.use_gradient_checkpoint = True

        self.kmeans = MiniBatchKMeans(n_clusters=self.cfg.TRAINER.VISTA.N, init='k-means++', n_init='auto', random_state=0)

        print("Turning off gradients in both the image and the text encoder")
        
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)
            
        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)#param_groups=param_groups
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        '''
        register model could be updated. When new module needs to be updated
        register the module before use
        '''
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.VISTA.PREC == "amp" else None

    def save_model(self, task, directory):
        save_dir = os.path.join(directory, f"Task_{task+1}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.model.prompt_learner.state_dict()['text_attr'], 
                os.path.join(save_dir, 'text_attr.pth.tar'))
        torch.save(self.model.state_dict()['vis_attr'], 
                    os.path.join(save_dir, 'vis_attr.pth.tar'))

    def load_model(self, directory, task=None):
        save_dir = os.path.join(directory, f"Task_{task+1}")
        path_key = os.path.join(save_dir, 'vis_attr.pth.tar')
        path_value = os.path.join(save_dir, 'text_attr.pth.tar')
        if os.path.exists(path_key) and os.path.exists(path_value):
            vis_attr_data = torch.load(path_key)
            text_attr_data = torch.load(path_value)
            self.model.vis_attr.data.copy_(vis_attr_data)
            self.model.prompt_learner.text_attr.data.copy_(text_attr_data)

    def train(self):
        """Class Incremental Unsupervised Domain Adaptation Training"""

        self.before_train()
        for self.task in range(0, self.num_classes // self.cfg.TRAIN.PER):
            self.increments(self.task)
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch() 
                self.after_epoch()
            self.after_task()
        self.after_train()

    def increments(self, task = None):
        self.train_loader_u, self.step_loader, self.s1_loader, self.final_loader = self.dm.new_task(task)
        if self.cfg.TRAINER.VISTA.DEBIASPL:
            self.qhat = (torch.ones([1, len(self.dm.dataset.classnames)], dtype=torch.float)/len(self.dm.dataset.classnames)).to('cuda:0')

    def run_epoch(self):
        """
        Run one epoch of training:
            if epoch == 0: gather features for kmeans++ initialization/updates
        """
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        elif self.cfg.TRAIN.COUNT_ITER == "bigger_one":
            self.num_batches = max(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        if self.epoch == 0:
            feat_s, feat_t = [], [] 
            for self.batch_idx in range(max(len_train_loader_x, len_train_loader_u)):   
                try:
                    batch_x = next(train_loader_x_iter)
                except StopIteration:
                    train_loader_x_iter = iter(self.train_loader_x)
                    batch_x = next(train_loader_x_iter)
                try:
                    batch_u = next(train_loader_u_iter)
                except StopIteration:
                    train_loader_u_iter = iter(self.train_loader_u)
                    batch_u = next(train_loader_u_iter)

                image_x, _, image_u, _ = self.parse_batch_train(batch_x, batch_u)
                output_s, output_t = self.model(image_x, image_u, cluster=True)
                if self.batch_idx < min(len(self.train_loader_x), len(self.train_loader_u)):
                    feat_s.extend(torch.chunk(output_s, output_s.shape[0], dim=0))
                    feat_t.extend(torch.chunk(output_t, output_t.shape[0], dim=0))
                else:
                    if len(self.train_loader_x) < len(self.train_loader_u):
                        feat_t.extend(torch.chunk(output_t, output_t.shape[0], dim=0))
                    else:
                        feat_s.extend(torch.chunk(output_s, output_s.shape[0], dim=0))
            
            data_s = torch.cat(feat_s, dim=0)
            data_t = torch.cat(feat_t, dim=0)

            if self.task == 0:
                kmeans_s = KMeans(n_clusters=self.cfg.TRAINER.VISTA.N, init='k-means++', n_init='auto', random_state=0).fit(data_s.cpu().numpy())
                center_s = kmeans_s.cluster_centers_
                self.kmeans = self.kmeans.partial_fit(data_t.cpu().numpy())
                center_t = self.kmeans.cluster_centers_
                self.model.vis_attr.data[:self.cfg.TRAINER.VISTA.N].copy_(torch.from_numpy(center_s).to('cuda:0'))   
                self.model.vis_attr.data[self.cfg.TRAINER.VISTA.N:].copy_(torch.from_numpy(center_t).to('cuda:0'))
        
            else:
                self.kmeans = self.kmeans.partial_fit(data_t.cpu().numpy())
                center_t = self.kmeans.cluster_centers_
                self.model.vis_attr.data[self.cfg.TRAINER.VISTA.N:].copy_(torch.from_numpy(center_t).to('cuda:0'))

        end = time.time()
        
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                    self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                batch_remain = 0
                batch_remain += self.num_batches - self.batch_idx - 1
                batch_remain += (self.max_epoch - self.epoch -
                              1) * self.num_batches
                task_remain = ((self.num_classes // self.cfg.TRAIN.PER - self.task - 1) * self.max_epoch * self.num_batches)
                eta_seconds = batch_time.avg * (batch_remain + task_remain)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                          self.epoch + 1,
                          self.max_epoch,
                          self.batch_idx + 1,
                          self.num_batches,
                          batch_time=batch_time,
                          data_time=data_time,
                          eta=eta,
                          losses=losses,
                          lr=self.get_current_lr(),
                      ))

            n_iter = self.task * self.max_epoch * self.num_batches + self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch_x, batch_u):
        # label_u only used for matric
        image_x, label, image_u, _ = self.parse_batch_train(batch_x, batch_u)
        prec = self.cfg.TRAINER.VISTA.PREC
        lam_1 = self.cfg.TRAINER.VISTA.LAM_1
        lam_2 = self.cfg.TRAINER.VISTA.LAM_2
        lam_3 = self.cfg.TRAINER.VISTA.LAM_3
        if prec == "amp":
            with autocast():
                output_ss, output_st, output_ts, output_tt, loss_div, loss_hp = self.model(image_x, image_u)

                # supervised loss on source labeled data
                loss_s = F.cross_entropy(output_ss, label)
                
                if self.cfg.TRAINER.VISTA.DEBIASPL:
                    # debiased pseudo-labeling
                    probs_tt = debias(output_tt, self.qhat, tau=self.cfg.TRAINER.VISTA.TAU)
                else:
                    probs_tt = torch.softmax(output_tt, dim=-1)
                
                gamma = self.cfg.TRAINER.VISTA.GAMMA
                max_probs, label_p = torch.max(probs_tt, dim=-1)
                mask_ge = max_probs.ge(gamma).float()

                if self.cfg.TRAINER.VISTA.DEBIASPL:
                    self.qhat = update_qhat(torch.softmax(output_tt.detach(), dim=-1), self.qhat, momentum=0.99, qhat_mask=mask_ge)

                # target loss: only include pseudo-labeled examples with high confidence
                loss_t = torch.tensor(0.0, device=self.device) if mask_ge.sum() == 0 \
                else (F.cross_entropy(output_tt, label_p, reduction='none') * mask_ge).sum() / mask_ge.sum()

                # prediction consistency loss (Jensen-Shannon divergence)
                loss_con = JSD(output_tt, output_ts) + JSD(output_ss, output_st)
                
                loss = loss_s + loss_t + lam_1 * loss_con + lam_2 * loss_hp + lam_3 * loss_div

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        loss_summary = {
            "loss": loss.item(),
            "loss_s": loss_s.item(),
            "loss_t": loss_t.item(),
            "loss_con": loss_con.item(),
            "loss_hp": loss_hp.item(),
            "loss_div": loss_div.item(),
            "acc_clip": compute_accuracy(output_ss, label)[0].item(),
        }

        self.update_lr()
        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)
        
        if do_test:
            curr_result = self.step(split='Step')
            self.set_model_mode("train")

    def after_task(self):
        last_task = (self.task + 1) == (self.num_classes // self.cfg.TRAIN.PER)
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.task + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)
        
        self.save_model(task=self.task, directory=self.output_dir,)

        if do_test:
            print(f"Task:{self.task}")
            s1_result = self.s1(split='S-1')
            curr_result = self.final(split='Final')
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result

            self.set_model_mode("train")

    def after_train(self):
        print("Finish training")
        print("Deploy the last-epoch model")
        self.final()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        input_u = batch_u["img"]
        label_u = batch_u['label']
        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        label_u = label_u.to(self.device)
        return input, label, input_u, label_u

    
          
    @torch.no_grad()
    def step(self, split=None):
        """Step-level Accuracy"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.step_loader
        print("Step-level Accuracy")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model(image_s=None, image_t=input, test=True)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.task * self.max_epoch + self.epoch)

        results_all = results["accuracy"]

        return results_all

    @torch.no_grad()
    def s1(self, split=None):
        """S-1 Accuracy"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.s1_loader
        print("S-1 Accuracy")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model(image_s=None, image_t=input, test=True)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.task)

        results_all = results["accuracy"]

        return results_all

    @torch.no_grad()
    def final(self, split=None):
        """Final Accuracy"""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.final_loader
        print("Final Accuracy")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model(image_s=None, image_t=input, test=True)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.task)

        results_all = results["accuracy"]

        return results_all