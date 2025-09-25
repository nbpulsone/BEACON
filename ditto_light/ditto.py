import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
from .process_budget import get_nn_samples, get_distribution_aware_samples, get_k_center_greedy_samples
from .dataset import DittoDataset, EmbedDittoDataset
from .generate_embeddings import get_singletons
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from apex import amp
import copy
#TODO: Update to new pytorch version

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8, use_pairwise_encoder=True, use_singleton_encoder=False, shared_model=None):
        super().__init__()
        print(f"SINGLETON: {use_singleton_encoder}, PAIRWISE: {use_pairwise_encoder}")
        # init the PLM(s): pairwise and maybe singleton
        if shared_model is not None: # copy from saved model (optimization)
            if use_pairwise_encoder:
                self.bert_pairwise = copy.deepcopy(shared_model)
            if use_singleton_encoder:
                self.bert_singleton = copy.deepcopy(shared_model)
        elif lm in lm_mp: # provided lm is in the map
            if use_pairwise_encoder:
                self.bert_pairwise = AutoModel.from_pretrained(lm_mp[lm])
            self.bert_singleton = None
            if use_singleton_encoder:
                self.bert_singleton = AutoModel.from_pretrained(lm_mp[lm])
        else: # provided lm use directly
            if use_pairwise_encoder:
                self.bert_pairwise = AutoModel.from_pretrained(lm)
            self.bert_singleton = None
            if use_singleton_encoder:
                self.bert_singleton = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        if use_pairwise_encoder:
            hidden_size = self.bert_pairwise.config.hidden_size
            self.fc = torch.nn.Linear(hidden_size, 2)


    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert_pairwise(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert_pairwise(x1)[0][:, 0, :]

        return self.fc(enc) # .squeeze() # .sigmoid()


    def forward_with_singletons(self, x_pair, x_left, x_right):
        # send tokenized input to gpu
        x_pair = x_pair.to(self.device)
        x_left = x_left.to(self.device)
        x_right = x_right.to(self.device)

        # pass the encoded pairwise tokens through PLM
        enc_pair = self.bert_pairwise(x_pair)[0][:, 0, :]

        # pass each singleton token through the singleton PLM
        enc_left = self.bert_singleton(x_left)[0][:, 0, :]
        enc_right = self.bert_singleton(x_right)[0][:, 0, :] 

        # pass the pairwise embeddings through main classification head
        logits = self.fc(enc_pair)
        
        # return classification head output and left and right embeddings
        return logits, enc_left, enc_right

    def forward_only_singletons(self, x_left, x_right):
        # send tokenized input to gpu
        x_left = x_left.to(self.device)
        x_right = x_right.to(self.device)

        # pass each singleton token through the singleton PLM
        enc_left = self.bert_singleton(x_left)[0][:, 0, :]
        enc_right = self.bert_singleton(x_right)[0][:, 0, :] 
        
        # return classification head output and left and right embeddings
        return enc_left, enc_right

    def get_cls_embeddings(self, input_ids):
        # batch of singleton embeddings
        input_ids = input_ids.to(self.device)

        # extract CLS embeddings from a batch of data without updating the model
        with torch.no_grad():
            outputs = self.bert_singleton(input_ids)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        return cls_embeddings
    
    # potential optimization: freeze after 2 epochs of raw training
    def toggle_singleton_encoder(self, unfreeze=True):
        for param in self.bert_singleton.parameters():
            param.requires_grad = unfreeze


def evaluate(model, iterator, threshold=None, dynamic_sampling=False, late_fusion_only=False):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class
        dynamic_sampling (bool, optional): if the training set is being resampled dynamically dynamically
            --> architecture = pairwise PLM + singleton PLM
        late_fusion_only (bool, optional): if the model fuses the singletons late
            --> architecture = singleton PLM only

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_y = []
    all_probs = []
    with torch.no_grad():
        # use only cosine similarity to evaluate
        if late_fusion_only:
            for batch in iterator:
                _, x_left, x_right, y = batch
                enc_left, enc_right = model.forward_only_singletons(x_left, x_right)

                enc_left = F.normalize(enc_left, dim=1)
                enc_right = F.normalize(enc_right, dim=1)

                cosine_sim = F.cosine_similarity(enc_left, enc_right)
                all_y += y.cpu().numpy().tolist()
                all_probs += cosine_sim.detach().cpu().numpy().tolist()
        # use only pairwise PLM to evaluate 
        else:
            for batch in iterator:
                #x, y = batch
                x = batch[0]
                y = batch [-1]
                logits = model(x)
                probs = logits.softmax(dim=1)[:, 1]
                all_probs += probs.cpu().numpy().tolist()
                all_y += y.cpu().numpy().tolist()

    # if threshold given, calculate F1 directly
    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        # calculate threshold that maximizes F1
        f1 = 0.0
        if late_fusion_only:
            # cosine sim
            best_th = 0.5
            thresh_range = np.linspace(0.0, 1.0, 101)
        else:
            # crossentropy loss
            best_th = 0.5
            thresh_range = np.arange(0.0, 1.0, 0.05)

        for th in thresh_range:
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th

def train_step(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        # initialize crossentropy and cosine losses
        ce_loss = None
        cosine_loss = None

        # late fusion: just use singleton PLM output
        if hp.late_fusion_only:
            x_pair, x_left, x_right, y = batch
            enc_left, enc_right = model.forward_only_singletons(x_left, x_right)
        # no late fusion and dynamic sampling: need singleton PLM and pairwise PLM output
        elif hp.dynamic_sampling:
            x_pair, x_left, x_right, y = batch
            prediction, enc_left, enc_right = model.forward_with_singletons(x_pair, x_left, x_right)

            ce_loss = criterion(prediction, y.to(model.device))
        else:
            # use just pairwise PLM output
            if len(batch) == 2:
                x, y = batch
                prediction = model(x)
            else:
                x1, x2, y = batch
                prediction = model(x1, x2)
  
            ce_loss = criterion(prediction, y.to(model.device))

        # cosine loss for singletons
        if hp.late_fusion_only or hp.dynamic_sampling:
            # auxiliary task: cosine similarity loss
            enc_left = F.normalize(enc_left, dim=1)
            enc_right = F.normalize(enc_right, dim=1)
            cosine_sim = F.cosine_similarity(enc_left, enc_right)
            match_mask = y.to(model.device) == 1
            nonmatch_mask = ~match_mask

            # match loss: want cosine similarity to approach 1
            match_loss = (1 - cosine_sim[match_mask]).mean() if match_mask.any() else 0.0
            nonmatch_loss = torch.clamp(cosine_sim[nonmatch_mask], min=0).mean() if nonmatch_mask.any() else 0.0

            cosine_loss = match_loss + nonmatch_loss

        # calculate final loss 
        if hp.late_fusion_only:
            loss = cosine_loss
        elif hp.dynamic_sampling:
            # loss with auxiliary task
            loss = ce_loss + hp.alpha_cosine * cosine_loss
        else:
            loss = ce_loss

        # fp-16 optimization
        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % 10 == 0: # monitoring
            if hp.late_fusion_only:
                print(f"step: {i}, cosine loss: {loss.item():.4f}")
            elif hp.dynamic_sampling:
                print(f"step: {i}, loss: {loss.item():.4f}, cosine loss: {cosine_loss.item():.4f}, ce loss: {ce_loss.item():.4f}")
            else:
                print(f"step: {i}, loss: {loss.item()}")
        del loss

def get_embeddings_from_lines(model, lines, tokenizer, hp, max_len=256, batch_size=32):
    """Use the given model to embed the given lines representing records

    Args:
        model (DMModel): the model
        lines (list): list of N strings representing singleton records
        tokenizer (class): the tokenizer for the given model
        hp (Namespace): other hyper-parameters (e.g., fp16)
        max_len (int, optional): max length for dataloader
        batch_size (int, optional): batch size for dataloader

    Returns:
        result (tensor): embeddings corresponding to lines 
                         (N x D, where D is dimension of model)
    """
    # set model in eval mode and create a data loader from the given data
    model.eval()
    embeddings = []
    dataset = EmbedDittoDataset(lines, max_len=max_len, lm=hp.lm, tokenizer=tokenizer)
    pad = dataset.pad
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad)

    # embed the data using current model
    with torch.no_grad():
        for batch in loader:
            x, _ = batch
            emb = model.get_cls_embeddings(x)
            embeddings.append(emb.cpu())
    
    result = torch.cat(embeddings, dim=0).numpy()
    return result

def regenerate_train_dataset(model, domain_data, domains, tokenizer, hp):
    """regenerate the train set using embeddings from the current model
       for nearest neighbors and train/val distribution methods

    Args:
        model (DMModel): the model
        domain_data (dict): train, val, and test data for each domain
        domains (list): list of strings representing the domains
        tokenizer (class): the tokenizer for the given model
        hp (Namespace): other hyper-parameters (e.g., fp16)
        
    Returns:
        result (DittoDataset): Ditto-formatted dataset of the regenerated data
    """
    print(f"~~~REGENERATING {hp.task}~~~")
    # get target embeddings (if needed)
    if (hp.method).startswith('tv') or (hp.method).startswith('tt'):
        # get singletons (for just this domain), pass that instead
        ds_index = 1 if (hp.method).startswith('tv') else 2 # 1 for val, 2 for test
        target_singeltons = get_singletons(domain_data, [hp.task], ds_index) 
        target_embed_data = get_embeddings_from_lines(model, target_singeltons[hp.task], tokenizer, hp)
        target_embed_data = [target_embed_data]

    # regenerate the training set embeddings (singletons)
    train_embed_data = []
    train_singletons = get_singletons(domain_data, domains, 0) # 0 for train
    for dom in domain_data:
        dom_embeds = get_embeddings_from_lines(model, train_singletons[dom], tokenizer, hp)
        train_embed_data.append(dom_embeds)
        # target data from train minimization method
        if (hp.method).startswith('tm') and dom == hp.task:
            target_embed_data = [dom_embeds]

    # get new samples to fill budget (if needed) from newly generated embeddings
    if (hp.method).startswith('nn'):
        # nearest neighbors resample
        new_train_data = get_nn_samples(domain_data, domains, hp.budget, train_embed_data, EMBED_DIM=768, single_domain=hp.task)
    elif (hp.method).startswith('kcg'):
        new_train_data = get_k_center_greedy_samples(domain_data, domains, hp.budget, train_embed_data, single_domain=hp.task)
    elif (hp.method).startswith('tv') or (hp.method).startswith('tt') or (hp.method).startswith('tm'):
        # parse metric, distribution-aware resample
        method_parts = (hp.method).split("_")
        metric = method_parts[-1]
        if len(method_parts) == 1 or metric == "tv" or metric == "tt" or metric == "tm":
            metric = "cs"
        new_train_data = get_distribution_aware_samples(domain_data, domains, hp.budget, train_embed_data, target_embed_data, single_domain=hp.task, metric=metric)
    else:
        new_train_data = domain_data
    
    # return new dataset in DITTO format
    print(f"Completed embedding-based resampling for {hp.task}")
    return DittoDataset(new_train_data[hp.task],
                        lm=hp.lm,
                        max_len=hp.max_len,
                        size=hp.size,
                        da=hp.da,
                        singleton=hp.dynamic_sampling,
                        tokenizer=tokenizer)


def train(trainset, validset, testset, run_tag, hp, domain_data, tokenizer=None, shared_model=None, init_dev_f1=0.0, init_test_f1=0.0, ensemble_method=None):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)
        domain_data (dict): Domain-seperated training data
        shared_model (class): instance of default PLM to copy
        init_dev_f1 (float): best validation f1 seen so far for this ensemble
        init_test_f1 (float): test f1 corresponding to init_dev_f1
        ensemble_method (bool): the ensemble this run of training is a part of

    Returns:
        best_dev_f1 (float): new best validation F1 after running this method
        best_test_f1 (float): new test F1 corrsponding to best_dev_f1
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug,
                       use_pairwise_encoder=not hp.late_fusion_only,
                       use_singleton_encoder=(hp.dynamic_sampling or hp.late_fusion_only),
                       shared_model=shared_model)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    # initialize LR scheduler
    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, # lr goes from 0 to inital LR, then decreases linearly across training steps
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)

    # start with current best val f1 being the best seen so far in the ensemble
    best_dev_f1 = init_dev_f1
    best_test_f1 = init_test_f1
    for epoch in range(1, hp.n_epochs+1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter, dynamic_sampling=hp.dynamic_sampling, late_fusion_only=hp.late_fusion_only)
        test_f1 = evaluate(model, test_iter, threshold=th, dynamic_sampling=hp.dynamic_sampling, late_fusion_only=hp.late_fusion_only)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                if hp.late_fusion_only:
                    arch = 'singleton-encoder'
                elif hp.dynamic_sampling:
                    arch = 'dual-encoder'
                else:
                    arch = 'pairwise-encoder'
                save_method_name = hp.method if ensemble_method is None else ensemble_method
                ckpt_path = os.path.join(hp.logdir, hp.task, f'{save_method_name}_model.pt')
                print(f"CP PATH: {ckpt_path}")

                # load previous and see if we should modify
                if hp.keep_lower_budgets: 
                    saved_state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
                    saved_f1 = float(saved_state['val_f1'])
                    print(f"LOWER BUDGET MODEL FOUND: VAL F1 = {saved_f1}, vs. CURRENT BEST = {best_dev_f1}")
                    if saved_f1 <= best_dev_f1 or int(saved_state['budget']) > hp.budget:
                        # save model and achitecture type
                        ckpt = {'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'epoch': epoch,
                                'arch': arch,
                                'budget': hp.budget,
                                'val_f1': best_dev_f1
                        }
                        torch.save(ckpt, ckpt_path)
                        print(f"\nMODEL SAVED!!!!!\n")
                    else:
                        # update best f1 from the one of the lower budget model
                        best_dev_f1 = saved_f1
                        print("\nSMALLER BUDGET MODEL WAS BETTER!!!\n")
                else:
                    # save model anc architecture type
                    ckpt = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                            'arch': arch,
                            'budget': hp.budget,
                            'val_f1': best_dev_f1
                    }
                    torch.save(ckpt, ckpt_path)
                    print(f"\nMODEL SAVED!!!!!\n")

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # dynamic resampling of data for nearest neighbors and train val distribution fitting methods, only for those domains that need to fill a budget (and thus sample)
        if hp.dynamic_sampling and epoch % hp.resample_freq == 0 and ((hp.method).startswith('nn') or (hp.method).startswith('tv') or (hp.method).startswith('tt') or (hp.method).startswith('tm') or (hp.method).startswith('kcg')) and hp.budget > len(domain_data[hp.task][0]):
            print(f"Resampling training set for epoch {epoch}...")
            trainset = regenerate_train_dataset(model, domain_data, list(domain_data.keys()), tokenizer, hp)
            train_iter = data.DataLoader(dataset=trainset,
                                        batch_size=hp.batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        collate_fn=padder)

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()

    return best_dev_f1, best_test_f1
