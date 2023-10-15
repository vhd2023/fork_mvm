import copy

import torch
import torch.nn as nn
import math

def log_it(*args, **kwargs):
    with open("colog/logit==Loss===LOG=====instance-shapes.out", "a") as f:
        print(*args, **kwargs, file=f)
        info = torch.cuda.mem_get_info(device="cuda:0")
        info = [bytes // (10 ** 6) for bytes in info]
        print("####\t\t#### free gpu:", info[0], "Mb\t", "occupied:", info[1], "Mb", file=f)

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, confidence_bs, class_num, temperature_ins, temperature_clu, device):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.confidence_bs = confidence_bs
        self.class_num = class_num
        self.temperature_ins = temperature_ins
        self.temperature_clu = temperature_clu
        self.device = device
        # self.stats = open("c_neg_logit_min_mean_stats.out", "a")
        

        self.mask_ins     = None #self.mask_correlated(batch_size)
        self.mask_clu     = self.mask_correlated(class_num)
        self.criterion    = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated(self, size):
        N = 2 * size
        mask = torch.ones((N, N)).to(self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(size):
            mask[i, size + i] = 0
            mask[size + i, i] = 0
        mask = mask.bool()
        return mask


    def my_diagonal_extractor(self, sim, class_num, contrast_count):
        #N = class_num * 2
        #length = sim.shape[0]
        # n_levels = contrast_count - 1
        pairs = []
        plevs = [[] for _ in range(contrast_count)]
        nlevs = [[] for _ in range(contrast_count)]
        # k = 1

        for level in range(1, contrast_count):
            # print("Level:", level)
            pdiag   = torch.diag(sim, level * class_num)
            ndiag   = torch.diag(sim, -level * class_num)
            #print(pdiag)
            pchunks = torch.split(pdiag, class_num)
            nchunks = torch.split(ndiag, class_num)
            # print(pchunks, end=f'\np diag level {level}\n')
            # print(ndiag)
            # print(nchunks, end=f'\nn diag level {level}\n')
            for i, (pos, neg) in enumerate(zip(pchunks, nchunks)):
                #print(i, levs)
                # if k != level:
                #     raise ValueError(f"kkkkkkkkkkkkkkkkkk: {k}, level: {level}")
                plevs[i    ].append(pos.unsqueeze(0))
                nlevs[i + level].append(neg.unsqueeze(0))
                #print(i, *plevs, sep='\n', end='\n------------------------------\n')
                #print(i, *nlevs, sep='\n', end='\n------------------------------\n')

            # k += 1
        # print("levvvvvvvvvvvvvvvvvvvs")
        for level, (p_list, n_list) in enumerate(zip(plevs, nlevs)):
            #pt_lev_diags = torch.cat(lev_diags, dim=0)
            n_list.reverse()
            #print(f"### {level}", p_list, n_list, sep='\n')

            pairs += n_list + p_list
        # print("\npair list:")
        # print(*pairs, sep='\n')
        # print("\nreturned tensor:")
        #pairs = [pair.unsqueeze() for pair in pairs]
        return torch.cat(pairs, dim=0)


    def my_multi_pos_loss(
                            self,
                            contrast_count,
                            # batch_size,
                            # class_num,
                            # device,
                            # temperature_clu,
                            # temperature_ins,
                            z_reps=None, c_reps=None,
                            z_i=None, z_j=None, c_i=None, c_j=None,
                        ):
        # if contrast_count == 2:
        #     z_reps, c_reps = None, None
            
        #rep = lambda s, t: f"### {s}:\t{t.shape}\n{str(t)}\n"

        def my_mask_correlated(size):
            N = 2 * size
            mask = torch.ones((N, N)).to(self.device)
            mask = mask.fill_diagonal_(0)
            for i in range(size):
                mask[i, size + i] = 0
                mask[size + i, i] = 0
            
            mask = mask.bool()
            if z_reps is not None:
                #print("repeating ........")
                mask = mask.repeat(contrast_count // 2, contrast_count // 2) ###

            return mask
            
        # mask_ins = my_mask_correlated(batch_size)
        mask_clu = my_mask_correlated(self.class_num)
        #print(rep("mask_ins (batch_size)", mask_ins.float()))
        #del mask_ins
        #print(rep("mask_clu (class_num)", mask_clu.float()))

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        similarity_f = torch.nn.CosineSimilarity(dim=2)

        # Entropy Loss
        if c_reps is None:
            p_i = c_i.sum(0).view(-1)
            p_i /= p_i.sum()
            ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
            
            p_j = c_j.sum(0).view(-1)
            p_j /= p_j.sum()

            ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
            ne_loss = ne_i + ne_j

            # Cluster Loss
            c_i = c_i.t() # Size([20, 7])
            c_j = c_j.t() # Size([20, 7])

        else:# contrast_count == 4:
            # p_k = c_k.sum(0).view(-1)
            # p_k /= p_k.sum()
            # ne_k = math.log(p_k.size(0)) + (p_k * torch.log(p_k)).sum()

            # p_l = c_l.sum(0).view(-1)
            # p_l /= p_l.sum()
            # ne_l = math.log(p_l.size(0)) + (p_l * torch.log(p_l)).sum()
            # ne_loss += ne_k + ne_l

            # c_k = c_k.t()
            # c_l = c_l.t()
            ne_loss = 0.
            for c_ii in c_reps:
                p_ii     = c_ii.sum(0).view(-1)
                '''
                sum  =p_ii.sum().tolist()
                mean = p_ii.mean().tolist()
                print("=====be===== sum:\n", sum, "======be===== mean:\n", mean)
                check_plus = (p_ii > 0).sum().item()
                check_zero = (p_ii == 0).sum().item()
                print("### before ### check plus:", check_plus, "### before ### check zero:", check_zero)
                print("p_ii:", p_ii.shape, p_ii.data[:20], end='\n' + "#" * 20 + '\n')
'''
                p_ii    /= p_ii.sum()

                min = p_ii.min().item()
                mean = p_ii.mean().item()
                # p_ii = p_ii + 5
                # logsoftmax = nn.LogSoftmax(dim=0)
                # logsoft_out = logsoftmax(p_ii)
                # added =nn.Softmax(dim=0)(p_ii)
                log_it("p_cluster MIN::::: ", min, "\tp_cluster Mean:::::",mean)
                
                '''
                sum  = p_ii.sum().tolist()
                mean = p_ii.mean().tolist()
                print("=====aft===== sum:\n", sum, "======aft===== mean:\n", mean)
                check_plus = (p_ii > 0).sum().item()
                check_zero = (p_ii == 0).sum().item()
                print("### after ### check plus:", check_plus, "### after ### check zero:", check_zero)
                print("p_ii:", p_ii.shape, p_ii.data[:20], end='\n' + "#" * 20 + '\n')'''
                
                # ne_ii    = math.log(p_ii.size(0)) + (p_ii * torch.log(p_ii)).sum()
                ne_ii    = math.log(p_ii.size(0)) + (p_ii * torch.log(p_ii)).sum()

                ne_loss += ne_ii

            c_reps = [c_ii.t() for c_ii in c_reps]
            
        # N = 2 * class_num
        if c_reps is not None:
            c = torch.cat((c_reps), dim=0) # Size([80, 7])
        else:
            c = torch.cat((c_i, c_j), dim=0) # Size([40, 7])
        #TODO: ?????????????????????????????  ?????
        N = self.class_num * contrast_count # 2 * 20 /  4 * 20
        #print("N:", N)
        #print("c_i:", c_i.shape, sep='\n')
        #print("c_j:", c_i.shape, c_j.data, sep='\n') 

        #print("c:", c.shape, c, sep='\n')

        sim = similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature_clu # Size([40, 40]) # sim: torch.Size([80, 80])

        #print("sim:", sim.shape, sim.data, sep='\n')
        if c_reps is None:
            sim_i_j = torch.diag(sim, class_num) # Size([20]) 
            #print("sim_i_j:", sim_i_j.shape, sim_i_j.data, sep='\n') 
        
            sim_j_i = torch.diag(sim, -class_num) # Size([20]) 
            #print("sim_j_i:", sim_j_i.shape, sim_j_i.data, sep='\n')
        
            positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # Size([40, 1]) 
        else:
            diag_output = self.my_diagonal_extractor(
                                sim=sim,
                                contrast_count=contrast_count, 
                                class_num=self.class_num
                                )
            #print(rep("diag_output", diag_output))
            positive_clusters = diag_output.reshape(N, -1) # Size [80,3])
            #print(rep("(positive_clusters)", positive_clusters))
            #pass .reshape(N, 1) ??????
            
        #print("positive_clusters:", positive_clusters.shape, positive_clusters, sep='\n')

        negative_clusters = sim[mask_clu].reshape(N, -1) # Size([40, 38])   # Size([80, 76])
        #print("negative_clusters:", negative_clusters.shape, negative_clusters.data, sep='\n')

        labels = torch.zeros(N).to(positive_clusters.device).long() # # Size([40])   # Size([80])
        #print("labels:", labels.shape, labels.data, sep='\n') 

        logits = torch.cat((positive_clusters, negative_clusters), dim=1) # Size([40, 39]) torch.Size([80, 79])
        #print("logits:", logits.shape, logits.data, sep='\n')

        cluster_loss = criterion(logits, labels)
        #print("cluster_loss:", cluster_loss.item(), end='\n####################')

        cluster_loss /= N

        mask = torch.eye(self.batch_size).bool().to(self.device)
        mask = mask.float()

        #contrast_count = contrast_count# 2
        # contrast_feature = torch.cat((z_i, z_j), dim=0)
        if z_reps is not None:
            #c = torch.cat((c_i, c_j, c_k, c_l), dim=0)
            contrast_feature = torch.cat((z_reps), dim=0)

        else:
            contrast_feature = torch.cat((z_i, z_j), dim=0)
            #c = torch.cat((c_i, c_j), dim=0)
            

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature_ins)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # print("logits_max, _", type(logits_max), logits_max.shape, end="\t")

        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        #TODO: TODO:TODO:TODO WHAT ABOUT POWERS OF MORE THAN 4?? 
        #                                   SHOULD IT BE REPEATED TWICE??? TODO:TODO:TODO:TODO:
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
                                    torch.ones_like(mask),
                                    1,
                                    torch.arange(self.batch_size * anchor_count).view(-1, 1).to(self.device),
                                    0
                                )
        mask = mask * logits_mask
        #print("## " * 10, "## " * 10 ,sep='\n')
        #print(rep("in mask", mask))
        #res1 = torch.nonzero(mask == 1, as_tuple=False)
        #res0 = torch.nonzero(mask == 0, as_tuple=False)
        #print(rep("perform instance contrastive learning on:", res1))
        # compute log_prob
        log_it("logit:", logits.shape,"\n", logits)
        log_it("min: ", logits.min().item(), "mean:", logits.mean().item(), "max:", logits.max().item(), sep=' / ')
        epsilon = 1e-5
        exp_logits = torch.exp(logits + epsilon) * logits_mask
        summed_exp_logits = exp_logits.sum(1, keepdim=True)
        log_it("summed_exp_logits:", summed_exp_logits.shape,"\n",  summed_exp_logits)

        log_it("min: ", summed_exp_logits.min().item(), "mean:",
                        summed_exp_logits.mean().item(), "max:",
                        summed_exp_logits.max().item(), sep=' / '
                        )

        log_prob = logits - torch.log(summed_exp_logits + epsilon)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss

        instance_loss = - mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, self.batch_size).mean()

        clusterwise_loss = cluster_loss + ne_loss
        batch_loss = instance_loss + clusterwise_loss
        log_it(f"batch loss: {batch_loss.item():.4f} / instance loss:{instance_loss.item():.4f} / cluster loss:{(cluster_loss + ne_loss).item():.4f} / ")
        return instance_loss, cluster_loss + ne_loss        #, mask
    

     
    def forward1(self, z_i, z_j, c_i, c_j):
        # Entropy Loss
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        # Cluster Loss
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature_clu
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask_clu].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        cluster_loss = self.criterion(logits, labels)
        cluster_loss /= N
        del logits, labels, positive_clusters, negative_clusters, sim, sim_i_j, sim_j_i, c, c_i, c_j, p_j, p_i
        

        contrast_count = 2
        device = z_i.device
        batch_size = 20000

        contrast_feature = torch.cat((z_i, z_j), dim=0) #.cpu() 5859 8310 4942 1604 
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        torch.cuda.empty_cache()
        log_it("anchor_feature:", anchor_feature.shape)

        product_out = chunked_matmul(anchor_feature, contrast_feature.T, chunk_size=1000, device=device)
        del anchor_feature, contrast_feature
        log_it("product_out:", product_out.shape)
        torch.cuda.empty_cache()
        anchor_dot_contrast = torch.div(product_out, self.temperature_ins)
        del product_out

        # anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature_ins)
        log_it("anchor_dot_contrast:", anchor_dot_contrast.shape)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        torch.cuda.empty_cache()

        logits = (anchor_dot_contrast - logits_max.detach())#.to('cuda')
        del anchor_dot_contrast, logits_max
        log_it("logits:", logits.shape)

        torch.cuda.empty_cache()

        mask = torch.eye(batch_size).bool().to(device)
        mask = mask.float()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        log_it("logits_mask:", logits_mask.shape)

        mask = mask * logits_mask
        log_it("mask:", mask.shape)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        del logits_mask

        log_it("exp_logits:", exp_logits.shape)

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_it("log_prob:", log_prob.shape)

        del logits, exp_logits
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        log_it("mean_log_prob_pos:", mean_log_prob_pos.shape)

        del mask, log_prob
        # loss
        instance_loss = - mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, batch_size).mean()
        return instance_loss, cluster_loss + ne_loss


    def generate_pseudo_labels(self, c, class_num):
        pseudo_label = -torch.ones(self.confidence_bs, dtype=torch.long).to(self.device)
        tmp = torch.arange(0, self.confidence_bs).to(self.device)
        with torch.no_grad():
            prediction = c.argmax(dim=1)
            confidence = c.max(dim=1).values
            pseudo_per_class = math.ceil(self.confidence_bs / class_num * 0.5)
            for i in range(class_num):
                class_idx = (prediction == i)
                confidence_class = confidence[class_idx]
                num = min(confidence_class.shape[0], pseudo_per_class)
                confident_idx = torch.argsort(-confidence_class)
                for j in range(num):
                    idx = tmp[class_idx][confident_idx[j]]
                    pseudo_label[idx] = i
        return pseudo_label

    def forward_instance_elim(self, z_i, z_j, pseudo_labels):
        batch_size =z_i.shape[0]
        # instance loss
        invalid_index = (pseudo_labels == -1)
        mask = torch.eq(pseudo_labels.view(-1, 1),
                        pseudo_labels.view(1, -1)).to(z_i.device)
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(batch_size).float().to(z_i.device)
        mask &= (~(mask_eye.bool()).to(z_i.device))
        mask = mask.float()

        contrast_count = 2
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature_ins)
        # for numeriecal stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask_with_eye = mask | mask_eye.bool()
        # mask = torch.cat(mask)
        mask = mask.repeat(anchor_count, contrast_count)
        mask_eye = mask_eye.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(self.batch_size * anchor_count).view(-1, 1).to(
                z_i.device), 0)
        logits_mask *= (1 - mask)
        mask_eye = mask_eye * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_eye * log_prob).sum(1) / mask_eye.sum(1)

        # loss
        instance_loss = -mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count,
                                           batch_size).mean()

        return instance_loss

    def forward_weighted_ce(self, c_, pseudo_label, class_num):
        idx, counts = torch.unique(pseudo_label, return_counts=True)
        freq = pseudo_label.shape[0] / counts.float()
        weight = torch.ones(class_num).to(pseudo_label.device)
        weight[idx] = freq
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss_ce = criterion(c_, pseudo_label)
        return loss_ce


    def forward(self, z_i, z_j, c_i, c_j):
        batch_size = z_i.shape[0]
        device = z_i.device
        # Entropy Loss
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        # Cluster Loss
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)
        # print(c.unsqueeze(1).shape, c.unsqueeze(0).shape)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature_clu
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask_clu].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        cluster_loss = self.criterion(logits, labels)
        cluster_loss /= N

        del logits, labels, positive_clusters, negative_clusters, sim, sim_i_j, sim_j_i, c, c_i, c_j, p_j, p_i


        mask = torch.eye(batch_size).bool().to(self.device)
        mask = mask.float()
        torch.cuda.empty_cache()

        contrast_count = 2
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature_ins)
        torch.cuda.empty_cache()

        # del anchor_feature, contrast_feature

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        torch.cuda.empty_cache()

        logits = anchor_dot_contrast - logits_max.detach()
        # del logits_max, anchor_dot_contrast
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        torch.cuda.empty_cache()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask
        
        torch.cuda.empty_cache()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # del logits_mask
        torch.cuda.empty_cache()

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # del logits, exp_logits

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        torch.cuda.empty_cache()

        # del mask, log_prob
        # loss
        instance_loss = - mean_log_prob_pos
        # del mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, batch_size).mean()

        return instance_loss, cluster_loss + ne_loss



'''


    def mask_correlated(self, size):
        N = 2 * size
        mask = torch.ones((N, N)).to(self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(size):
            mask[i, size + i] = 0
            mask[size + i, i] = 0
        mask = mask.bool()
        return mask

    def generate_pseudo_labels(self, c, class_num):
        pseudo_label = -torch.ones(self.confidence_bs, dtype=torch.long).to(self.device)
        tmp = torch.arange(0, self.confidence_bs).to(self.device)
        with torch.no_grad():
            prediction = c.argmax(dim=1)
            confidence = c.max(dim=1).values
            pseudo_per_class = math.ceil(self.confidence_bs / class_num * 0.5)
            for i in range(class_num):
                class_idx = (prediction == i)
                confidence_class = confidence[class_idx]
                num = min(confidence_class.shape[0], pseudo_per_class)
                confident_idx = torch.argsort(-confidence_class)
                for j in range(num):
                    idx = tmp[class_idx][confident_idx[j]]
                    pseudo_label[idx] = i
        return pseudo_label
        

    def forward_weighted_ce(self, c_, pseudo_label, class_num):
        idx, counts = torch.unique(pseudo_label, return_counts=True)
        freq = pseudo_label.shape[0] / counts.float()
        weight = torch.ones(class_num).to(pseudo_label.device)
        weight[idx] = freq
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss_ce = criterion(c_, pseudo_label)
        return loss_ce

    def forward(self, z_i, z_j, c_i, c_j, pseudo_labels, pseudo_labels_oc):
        # Entropy Loss
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        # Cluster Loss
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature_clu
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask_clu].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        cluster_loss = self.criterion(logits, labels)
        cluster_loss /= N

        mask = torch.eye(self.batch_size).bool().to(self.device)
        mask = mask.float()

        contrast_count = 2
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature_ins)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(self.batch_size * anchor_count).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        instance_loss = - mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, self.batch_size).mean()

        return instance_loss, cluster_loss + ne_loss

    
    def forward2(z_i, z_j, c_i, c_j, z_k=None, z_l=None, c_k=None, c_l=None, contrast_count=2, class_num=2):
        # Entropy Loss
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()

        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        if contrast_count == 4:
            p_k = c_k.sum(0).view(-1)
            p_k /= p_k.sum()
            ne_k = math.log(p_k.size(0)) + (p_k * torch.log(p_k)).sum()

            p_l = c_l.sum(0).view(-1)
            p_l /= p_l.sum()
            ne_l = math.log(p_l.size(0)) + (p_l * torch.log(p_l)).sum()
            ne_loss += ne_k + ne_l
            c_k = c_k.t()
            c_l = c_l.t()
        # Cluster Loss
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * class_num
        if contrast_count == 4:
            c = torch.cat((c_i, c_j, c_k, c_l), dim=0)
        else:
            c = torch.cat((c_i, c_j), dim=0)


        sim = similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / temperature_clu
        sim_i_j = torch.diag(sim, class_num)
        sim_j_i = torch.diag(sim, -class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[mask_clu].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        cluster_loss = criterion(logits, labels)
        cluster_loss /= N

        mask = torch.eye(batch_size).bool().to(device)
        mask = mask.float()

        #contrast_count = contrast_count# 2
        contrast_feature = torch.cat((z_i, z_j), dim=0)
        if contrast_count == 4:
            #c = torch.cat((c_i, c_j, c_k, c_l), dim=0)
            contrast_feature = torch.cat((z_i, z_j, z_k, z_l), dim=0)

        else:
            contrast_feature = torch.cat((z_i, z_j), dim=0)
            #c = torch.cat((c_i, c_j), dim=0)
            

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature_ins)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
                                    torch.ones_like(mask),
                                    1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                                    0
                                )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        instance_loss = - mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, batch_size).mean()

        return instance_loss, cluster_loss + ne_loss


def batched_matrix_multi2(tensor_data, chunk_size=2000):

    # Sample data sizes
    num_samples, embedding_size = tensor_data.shape #40000
    #embedding_size = 128
    # chunk_size = 2000  # Split the data into chunks of 2000 samples

    # Generate some random data for demonstration
    tensor_data = torch.rand(num_samples, embedding_size)#.cuda()

    # Initialize an empty tensor to store the result
    result = torch.empty(num_samples, num_samples)#.cuda()

    # Compute matrix multiplication in chunks
    for i in range(0, num_samples, chunk_size):
        start_idx = i
        end_idx = min(i + chunk_size, num_samples)

        # Select a chunk of data
        chunk = tensor_data[start_idx:end_idx]

        # Compute matrix multiplication for the chunk and store the result
        partial_result = torch.matmul(chunk, tensor_data.T)

        # Place the partial result in the correct position in the final result tensor
        result[start_idx:end_idx, :] = partial_result

    # Now, 'result' contains the matrix multiplication of tensor_data with its transpose
    return result

def batched_matrix_multi(A, batch_size = 1024):
    # Input tensors
    # A = torch.randn(40000, 128) 
    B = A.t()#transpose(0, 1)

    # Batch size
    
    # Number of batches 
    num_batches = A.shape[0] // batch_size

    # Output tensor 
    C = torch.empty(A.shape[0], A.shape[0], dtype=A.dtype, device=A.device)

    for i in range(num_batches):

        # Batch slices
        A_batch = A[i*batch_size:(i+1)*batch_size]  
        B_batch = B[:, i*batch_size:(i+1)*batch_size]

        # Batched multiplication
        C_batch = torch.matmul(A_batch, B_batch)

        # Write to output tensor
        C[i*batch_size:(i+1)*batch_size, i*batch_size:(i+1)*batch_size] = C_batch
    return C

'''
def chunked_matmul(A, B, chunk_size, device): 
    # Split matrices A and B into smaller chunks
    # B = A.t() 
    A_chunks = torch.split(A, chunk_size, dim=1) 
    B_chunks = torch.split(B, chunk_size, dim=0) 
    # Multiply each chunk of A with the corresponding chunk of B 
    C_chunks = [] 
    torch.cuda.empty_cache()
    for i in range(len(A_chunks)): 

        C_chunk = torch.mm(A_chunks[i], B_chunks[i]) 
        C_chunks.append(C_chunk) 
        # Concatenate the resulting chunks to form the final matrix 
    C = torch.cat(C_chunks, dim=1).to(device)
    torch.cuda.empty_cache()

    return C 
    # Example usage A = torch.randn(10000, 5000) B = torch.randn(5000, 10000) C = chunked_matmul(A, B, 1000)
    
def get_diag(sim, class_num, contrast_count):
    #N = class_num * 2
    #length = sim.shape[0]
    # n_levels = contrast_count - 1
    pairs = []
    plevs = [[] for _ in range(contrast_count)]
    nlevs = [[] for _ in range(contrast_count)]
    k = 1

    for level in range(1, contrast_count):
        print("Level:", level)
        pdiag   = torch.diag(sim, level * class_num)
        ndiag   = torch.diag(sim, -level * class_num)
        print(pdiag)
        pchunks = torch.split(pdiag, class_num)
        nchunks = torch.split(ndiag, class_num)
        # print(pchunks, end=f'\np diag level {level}\n')
        # print(ndiag)
        # print(nchunks, end=f'\nn diag level {level}\n')
        for i, (pos, neg) in enumerate(zip(pchunks, nchunks)):
            #print(i, levs)
            plevs[i    ].append(pos.unsqueeze(0))
            nlevs[i + k].append(neg.unsqueeze(0))
            print(i, *plevs, sep='\n', end='\n------------------------------\n')
            print(i, *nlevs, sep='\n', end='\n------------------------------\n')

        k += 1
    print("levvvvvvvvvvvvvvvvvvvs")
    for level, (p_list, n_list) in enumerate(zip(plevs, nlevs)):
        #pt_lev_diags = torch.cat(lev_diags, dim=0)
        n_list.reverse()
        print(f"### {level}", p_list, n_list, sep='\n')

        pairs += n_list + p_list
    print("doooooooooooooooonnnneeeeeeeeeeeeeeeeeeeeeeeeeeed")
    print(*pairs, sep='\n')
    print("adfjsfja;lfja;fajfl;adj")
    #pairs = [pair.unsqueeze() for pair in pairs]
    return torch.cat(pairs, dim=0)

if __name__ == "__main__":
    def are_tensors_approximately_equal(tensor1, tensor2, tolerance=1e-6):
        # Check if the shapes of the tensors are the same
        if tensor1.shape != tensor2.shape:
            return False

    # Check if the absolute difference between corresponding elements is within the tolerance
        elementwise_difference = torch.abs(tensor1 - tensor2)
        return torch.all(elementwise_difference < tolerance)
    
    A = torch.rand(1000, 50)
    # R1 = batched_matrix_multi(A)
    # R2 = batched_matrix_multi2(A)
    R4 = chunked_matmul(A, chunk_size=200)
    # r = are_tensors_approximately_equal(R1, R2)
    import numpy as np
    # try:
    #     print("1111111111111111111111111111111111", R1.shape, R2.shape)
    #     np.testing.assert_allclose(R1, R2, rtol=1e-5, atol=1e-8)
    # except Exception as e:
    #     pass
    #     #print(e)
    
    R3 = torch.matmul(A, A.t())

    # try:
    #     print("22222222222222222222222222222222222", R3.shape, R1.shape)
    #     np.testing.assert_allclose(R3, R1, rtol=1e-5, atol=1e-8)
    # except Exception as e:
    #     pass
    #     #print(e)


    # try:
    #     print("33333333333333333333333333333333333", R3.shape, R2.shape)
    #     np.testing.assert_allclose(R3, R2, rtol=1e-5, atol=1e-8)
    # except Exception as e:
    #     pass
    #     #print(e)
    
    try:
        print("4444444444444444444444444444444444444", R3.shape, R4.shape)
        np.testing.assert_allclose(R3, R4,  rtol=1e-5, atol=1e-8)
    except Exception as e:
        print(e)

    print(are_tensors_approximately_equal(R3, R4))

    # Calculate the MSE between the two tensors
    mse = torch.mean((R3 - R4) ** 2)
    # Check if the difference is insignificant or not
    if mse.item() < 1e-10:
        print("The two tensors are close and accurate enough for a deep learning project.")
    else:
        print("The two tensors are different.")
    # print(are_tensors_approximately_equal(R3, R1), are_tensors_approximately_equal(R3, R2))
    # contrast_count = 8
    # class_num =  20
    # N = class_num * contrast_count
    # ts = []
    # squares = contrast_count * contrast_count
    
    # for i in range(1, squares + 1):
    #     tens = torch.ones(class_num) * i
    #     a = torch.diag(tens)
    #     #print(i, a.shape, end="\n\n")
    #     ts.append(a)
    # ts = [torch.cat(ts[s:s + contrast_count], dim=1) for s in range(0, len(ts), contrast_count)]
    # # print(len(ts), ts[0].shape, end="\n\n")
    # # print(*ts, sep="\n\n", end="\n**##########################***\n\n")
    # sim = torch.cat(ts, dim=0)
    # print(sim.shape)
    # print(sim)

    # out = get_diag(sim, class_num, contrast_count)
    # print(sim)
    # print(sim.shape)
    # print(out.shape)
    # print(out)

    #  def get_diag(sim, class_num, contrast_count):
    #         N = class_num * 2
    #         #length = sim.shape[0]
    #         # n_levels = contrast_count - 1
    #         pairs = []
    #         levs = [[] for _ in range(contrast_count)]
    #         for level in range(1, contrast_count, 2):
    
    #             pdiag = torch.diag(sim, level * class_num)
    #             ndiag = torch.diag(sim, -level * class_num)
    #             pchunks = torch.split(pdiag, class_num)
    #             nchunks = torch.split(ndiag, class_num)
    #             k = 1
    #             for i, (pos, neg) in enumerate(zip(pchunks, nchunks), start=1):
    #                 levs[i].append(pos)
    #                 levs[i + k].append(neg)
    #                 k += 2
    #         for lev_diags in levs:
    #             pairs += lev_diags
    #         pairs