import sys
import torch
import scipy
import argparse
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models import VAE2, Classifier

def get_args(trialIndex, sourceDomainIndex, targetDomainIndex, input_dim=2048):
    sys.argv = [
        '',  # argv[0] is the script name; keep it as ''
        '--sourceDomainIndex', str(sourceDomainIndex),
        '--targetDomainIndex', str(targetDomainIndex),
        '--trialIndex', str(trialIndex), # 0, 1, 2, 3, 4
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[input_dim, 512])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[512, input_dim])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--sourceDomainIndex", type=int, default=0)
    parser.add_argument("--targetDomainIndex", type=int, default=1)
    parser.add_argument("--trialIndex", type=int, default=0)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    return parser.parse_args()

def set_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

def test_model(model,dataset,dataloader,device):
    num_class = dataset.num_class
    running_corrects = np.zeros((num_class,))
    num_sample_per_class = np.zeros((num_class,))
    # Iterate over data.
    for index, (features,labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            model.eval()
            preds = model(features)
            preds = preds.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            if index == 0:
                outputs_test = preds
                labels_test = labels
            else:
                outputs_test = np.concatenate((outputs_test, preds), 0)
                labels_test = np.concatenate((labels_test, labels), 0)
        preds = np.argmax(outputs_test,1)
   
    for i in range(len(labels_test)):
        num_sample_per_class[labels_test[i]] += 1
        if preds[i]==labels_test[i]:
            running_corrects[labels_test[i]] += 1

    acc_per_class = running_corrects / num_sample_per_class
    acc = np.mean(acc_per_class)
    acc_seen = np.mean(acc_per_class[dataset.unseenClass_B==0])
    acc_unseen = np.mean(acc_per_class[dataset.unseenClass_B==1])
    h = acc_seen * acc_unseen * 2 / (acc_seen + acc_unseen)

    print('per-class acc:{:2.4f}, seen acc:{:2.4f}, unseen acc:{:2.4f}, H:{:2.4f}'.format(acc,acc_seen,acc_unseen,h))
    return acc_per_class, acc, acc_seen, acc_unseen

def loss_fn2(
        recon_xS, recon_xS2, xS,
        recon_xT, recon_xT2, xT,
        meanS, log_varS,
        meanT, log_varT,
        yT,
        epoch):
    # criterion = torch.nn.MSELoss(size_average=False)
    criterion = torch.nn.MSELoss(reduction='sum')

    mask = yT!=-1
    reconstruction_loss = criterion(recon_xS, xS) + criterion(recon_xT[mask,:], xT[mask,:])
    cross_reconstruction_loss = criterion(recon_xS2[mask,:], xT[mask,:]) + criterion(recon_xT2[mask,:], xS[mask,:])
    KLD = -0.5 * torch.sum(1 + log_varS - meanS.pow(2) - log_varS.exp())  -0.5 * torch.sum(1 + log_varT[mask,:] - meanT[mask,:].pow(2) - log_varT[mask,:].exp())
    weight = epoch*5e-4
    return (reconstruction_loss + 1*cross_reconstruction_loss + weight*KLD) / xS.size(0)

def generate_z(xS, yS, xT, yT, vae, device):
    vae.eval()
    recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, yS, d=torch.zeros_like(yS).to(device))
    recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, yT, d=torch.ones_like(yT).to(device))
    return recon_xS2, recon_xT2

def get_datesets_and_loaders(args, DOMAIN_SET, DATA_DIR, DATASET_DETAILS):
    datasets = {}
    datasets['train'] = BaseTwoModalDataset(domain_set=DOMAIN_SET, data_dir=DATA_DIR, phase='train',sourceDomainIndex=args.sourceDomainIndex, targetDomainIndex=args.targetDomainIndex,trialIndex=args.trialIndex, dataset_details=DATASET_DETAILS)
    datasets['test'] = BaseTwoModalDataset(domain_set=DOMAIN_SET, data_dir=DATA_DIR, phase='test',sourceDomainIndex=args.sourceDomainIndex, targetDomainIndex=args.targetDomainIndex,trialIndex=args.trialIndex, dataset_details=DATASET_DETAILS)
    data_loaders={}
    data_loaders['train'] = DataLoader(dataset=datasets['train'], batch_size=args.batch_size, shuffle=True)
    data_loaders['test'] = DataLoader(dataset=datasets['test'], batch_size=len(datasets['test']), shuffle=False)

    return datasets, data_loaders


#TODO merget these functions
def get_trained_VAE(data_loaders, args, device):
    vae = VAE2(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_domains = 2).to(device)    
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    ############################################################
    # train CVAE
    ############################################################
    for epoch in range(args.epochs):
        for iteration, (xS,xT,yS,yT) in enumerate(data_loaders['train']):

            xS,xT,yS,yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)

            recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, yS, d=torch.zeros_like(yS).to(device))
            recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, yT, d=torch.ones_like(yT).to(device))
            loss = loss_fn2(recon_xS, recon_xS2, xS, recon_xT, recon_xT2, xT, meanS, log_varS, meanT, log_varT, yT, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return vae
def get_trained_VAE_with_domain_classifier(data_loaders, args, device):
    vae = VAE2(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_domains = 2).to(device)    

    domain_classifier = DomainClassifier(latent_size=args.latent_size).to(device)
    optimizer = torch.optim.Adam(list(vae.parameters()) + list(domain_classifier.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    ############################################################
    # train CVAE
    ############################################################
    for epoch in range(args.epochs):
        for iteration, (xS,xT,yS,yT) in enumerate(data_loaders['train']):

            xS,xT,yS,yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)

            recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, yS, d=torch.zeros_like(yS).to(device))
            recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, yT, d=torch.ones_like(yT).to(device))
            loss = loss_fn2(recon_xS, recon_xS2, xS, recon_xT, recon_xT2, xT, meanS, log_varS, meanT, log_varT, yT, epoch)

            # Domain predictions
            domain_pred_S = domain_classifier(zS)
            domain_pred_T = domain_classifier(zT[yT != -1])
            
            # Domain labels
            label_S = torch.zeros(xS.size(0), dtype=torch.long).to(device)
            label_T = torch.ones(xT[yT != -1].size(0), dtype=torch.long).to(device)

            domain_loss = F.cross_entropy(domain_pred_S, label_S) + \
                     F.cross_entropy(domain_pred_T, label_T)

            total_loss = loss + .2 * domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        scheduler.step()
    return vae


#TODO merge these functions
def get_trained_classifier(data_loaders, vae, NUM_LABELS, device, num_epochs = 50, change_policy_epoch = 50, input_dim=2048):
    classifier = Classifier(input_dim=input_dim,num_labels=NUM_LABELS).to(device) # train and test a classifier
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=0.01)
    scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=25, gamma=0.1)
    for epoch in range(num_epochs):
        for iteration, (xS,xT,yS,yT) in enumerate(data_loaders['train']):
            xS, xT, yS, yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)
            recon_xS, recon_xT = generate_z(xS, yS, xT, yT, vae, device)
            mask = yT!=-1
            xT = xT[mask,:]
            yT = yT[mask]
            recon_xT = recon_xT[mask,:]

            if epoch < change_policy_epoch:
                xtrain = torch.cat((xS, xT, recon_xS, recon_xT), dim=0)
                ytrain = torch.cat((yS, yT, yS, yT), dim=0)
            else:
                xtrain = recon_xS
                ytrain = yS

            output = classifier(xtrain)
            loss_cls = classifier.lossfunction(output, ytrain)
            optimizer_cls.zero_grad()
            loss_cls.backward()
            optimizer_cls.step()
        scheduler_cls.step()
    return classifier

def get_trained_classifier_Base(data_loaders, NUM_LABELS, device, input_dim=2048):
    classifier = Classifier(input_dim=input_dim,num_labels=NUM_LABELS).to(device) # train and test a classifier
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=0.01)
    scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=25, gamma=0.1)
    num_epochs = 50 # TODO remove
    for epoch in range(num_epochs):
        for iteration, (xS,xT,yS,yT) in enumerate(data_loaders['train']):
            xS, xT, yS, yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)

            mask = yT!=-1
            xT = xT[mask,:]
            yT = yT[mask]

            xtrain = torch.cat((xS, xT), dim=0)
            ytrain = torch.cat((yS, yT), dim=0)

            output = classifier(xtrain)
            loss_cls = classifier.lossfunction(output, ytrain)
            optimizer_cls.zero_grad()
            loss_cls.backward()
            optimizer_cls.step()
        scheduler_cls.step()
    return classifier

def prepare_report(results):
    seen = np.array([r[2] for r in results]) * 100
    unseen = np.array([r[3] for r in results]) * 100
    h = seen * unseen * 2 / (seen + unseen)

    mean_seen = np.mean(seen)
    sem_seen = np.std(seen, ddof=1) / np.sqrt(len(seen))

    mean_unseen = np.mean(unseen)
    sem_unseen = np.std(unseen, ddof=1) / np.sqrt(len(unseen))

    mean_h = np.mean(h)
    sem_h = np.std(h, ddof=1) / np.sqrt(len(h))

    return (
        f"Seen:     {mean_seen:.2f} ± {sem_seen:.2f}"     + "\n"
        f"Unseen:   {mean_unseen:.2f} ± {sem_unseen:.2f}" + "\n"
        f"H-mean:   {mean_h:.2f} ± {sem_h:.2f}")

def run_all_senario(main, DOMAIN_SET, input_dim=2048, num_trial=5, pairs=None):
    senario_report_map = dict()
    if pairs is None:
        pairs = [
            (s, t)
            for s in range(len(DOMAIN_SET))
            for t in range(len(DOMAIN_SET))
            if s != t
        ]
    else:
        pairs = list(pairs)
    pbar = tqdm(total=len(pairs) * num_trial, desc="episodes", unit="run")
    for s, t in pairs:
        senario = "%s -> %s" % (DOMAIN_SET[s], DOMAIN_SET[t])
        results = []
        for i in range(num_trial):
            pbar.set_postfix_str("%s trial %d" % (senario, i))
            args = get_args(trialIndex=i, sourceDomainIndex=s, targetDomainIndex=t, input_dim=input_dim)
            res = main(args)
            results.append(res)
            acc_s = float(res[2])
            acc_u = float(res[3])
            h = 0.0 if (acc_s + acc_u) == 0 else 2 * acc_s * acc_u / (acc_s + acc_u)
            tqdm.write(
                "%s trial %d  seen:%6.2f  unseen:%6.2f  H:%6.2f"
                % (senario, i, acc_s * 100, acc_u * 100, h * 100)
            )
            pbar.update(1)
        report = prepare_report(results)
        senario_report_map[senario] = report
        print(senario)
        print(report)
    pbar.close()
    return senario_report_map

class BaseTwoModalDataset(Dataset):
    def __init__(
            self,
            domain_set,
            data_dir,
            phase,
            sourceDomainIndex,
            targetDomainIndex,
            trialIndex,
            dataset_details
    ):
        super().__init__()
        self.domain_set = domain_set
        self.data_dir = data_dir
        self.dataset_details = dataset_details

        self._load_mat(sourceDomainIndex, targetDomainIndex, trialIndex)
        self.phase = phase
        if self.phase == 'train':
            flag = 1
        if self.phase == 'test':
            flag = 2
        self.feature_B = self.feature_B[self.splitFlag_B == flag,]
        self.label_B = self.label_B[self.splitFlag_B == flag]
    
    def _load_mat(self, sourceDomainIndex, targetDomainIndex, trialIndex):
        data_A = scipy.io.loadmat(
            self.data_dir +
            self.dataset_details["prefix"] +
            self.domain_set[sourceDomainIndex] + 
            self.dataset_details["suffix"])
        feature_A = data_A[self.dataset_details["resnet_feature"]].squeeze()
        self.feature_A = normalize(feature_A, norm='l2')
        self.label_A = data_A['labels'][0,]
        self.num_class = len(np.unique(self.label_A))
        
        data_B = scipy.io.loadmat(
            self.data_dir +
            self.dataset_details["prefix"] +
            self.domain_set[targetDomainIndex] +
            self.dataset_details["suffix"])
        feature_B = data_B[self.dataset_details["resnet_feature"]].squeeze()
        self.feature_B = normalize(feature_B,norm='l2')
        self.label_B = data_B['labels'][0,]

        dataSplit = scipy.io.loadmat(self.data_dir + self.dataset_details["split_file_name"])
        self.splitFlag_B = dataSplit['targetDomain_splitFlag'][0,trialIndex][0,targetDomainIndex][0,] # [0, index of trial][0, index of domain], 1--train, 2--test, 0--not used
        self.unseenClass_B = dataSplit['targetDomain_unseenClass'][0,trialIndex][0,targetDomainIndex][0,]

    def __len__(self):
        if self.phase == 'train':
            return self.feature_A.shape[0]
        if self.phase == 'test':
            return self.feature_B.shape[0]

    def __getitem__(self,idx):
        if self.phase == 'test':
            idx_B = idx
            return self.feature_B[idx_B,:], self.label_B[idx_B]

        label = self.label_A[idx]
        indicesB_this_label = np.argwhere(self.label_B == label)
        if len(indicesB_this_label) > 0:
            idx_B = np.random.choice(indicesB_this_label[:,0])
            return self.feature_A[idx,:], self.feature_B[idx_B,:]             , self.label_A[idx], self.label_B[idx_B]
        else:
            return self.feature_A[idx,:], np.zeros_like(self.feature_A[idx,:]), self.label_A[idx], np.ones_like(self.label_A[idx]) * -1



class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DomainClassifier(nn.Module):
    def __init__(self, latent_size, hidden_size=128):
        super().__init__()
        
        self.grl = GradientReversalLayer(alpha=0.1)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2)  # source=0, target=1
        )
    
    def forward(self, z):
        z_grl = self.grl(z)
        return self.classifier(z_grl)

