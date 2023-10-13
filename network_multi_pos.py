# import numpy as np
import torch.nn as nn
import torch
torch.set_printoptions(precision=3, sci_mode=False)

from torch.nn.functional import normalize
import time
import functools
import numpy as np


# from train import timer
# 14, 15, 18, 23, 24, and 27
def log_it(*args, **kwargs):
    with open("colog/logit==Loss===LOG=====instance-shapes.out", "a") as f:
        print(*args, **kwargs, file=f)
        print(*args, **kwargs)
        info = torch.cuda.mem_get_info(device="cuda:0")
        info = [bytes // (10 ** 6) for bytes in info]
        print("####\t\t#### free gpu:", info[0], "Mb\t", "occupied:", info[1], "Mb", file=f)


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        dif = time.time() - start
        dif_min = dif / 60
        name = func.__name__
        if True:  # global_step - start_itr < 20 or global_step % 50 == 0:
            # if dif_min > 0.1:
            with open("timer_for_mix_mlm.out", "a") as f:
                print(
                    f"## timer step NETWORK ## {name:10}: {dif_min:.3f} MIN, {dif:.1f} SEC",
                    file=f,
                )
        # if global_step - start_itr < 20 or global_step % 20:
        # random = np.random.choice(2, 1, p=[0.9, 0.1])
        # print(f"random: {random}")
        if False:  # random.item() == 5:
            print(
                f"## timer step NETWORK ## {name:10}: {dif_min:.3f} MIN, {dif:.1f} SEC",
            )
            # import train_t5_mix

            # from train_t5_mix import writer, batch_counter
            # wb.log({f"t(m)/{name}": dif_min, f"t(s)/{name}": dif})
            # train_t5_mix.tb_writer.add_scalar(
            #     f"t(m)/{name}", dif_min, train_t5_mix.batch_counter
            # )
            # train_t5_mix.tb_writer.add_scalar(
            #     f"t(s)/{name}", dif, train_t5_mix.batch_counter
            # )
        return results

    return wrapper


class Network(nn.Module):
    def __init__(
        self, backbone, feature_dim, class_num, tokenizer, config, args
    ):
        super(Network, self).__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.backbone = backbone
        from models_mix_mlm import MaskedMixCSEHead

        self.mlm_mix_head = None
        if args.do_mlm or args.do_mix:
            self.mlm_mix_head = MaskedMixCSEHead(
                config=config, do_mlm=args.do_mlm, do_mix=args.do_mix
            )

        self.mlm_mix_head = None
        #self.activation = 
        #self.activation()
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
                                                    # nn.BatchNorm1d(self.config.hidden_size),##########
            nn.BatchNorm1d(self.config.hidden_size, device="cuda"),
            nn.ReLU(), #nn.ReLU(),                            # reeeal nn.ReLU(),
            nn.Linear(
                self.config.hidden_size,
                self.config.hidden_size,
            ),
            nn.BatchNorm1d(self.config.hidden_size, device="cuda"),
            nn.ReLU(), #nn.ReLU(),                            # reeeal nn.ReLU(),
            nn.Linear(
                self.config.hidden_size,
                self.feature_dim,
            ),
            nn.BatchNorm1d(self.feature_dim, device="cuda"),
            nn.ReLU(), #nn.ReLU(),                            # reeeal nn.ReLU(),
        )
        self.cluster_projector = nn.Sequential(
            nn.BatchNorm1d(self.config.hidden_size, device="cuda"),###########                     
            nn.ReLU(),                      #nn.ReLU(),
            nn.Linear(
                self.config.hidden_size,
                self.config.hidden_size,
            ),   
            nn.BatchNorm1d(self.config.hidden_size, device="cuda"),###########
            nn.ReLU(), #nn.ReLU(),                            # reeeal nn.ReLU(),
            nn.Linear(
                self.config.hidden_size,
                self.cluster_num,
            ),
            nn.BatchNorm1d(self.cluster_num, device="cuda"),
            nn.ReLU(), #nn.ReLU(),                            # reeeal nn.ReLU(),
        )
        # self.instance_projector = nn.Sequential(
        #                                             # nn.BatchNorm1d(self.config.hidden_size),##########
        #                                             # nn.ReLU(),###############
        #     nn.BatchNorm1d(self.config.hidden_size),###########
        #     nn.ReLU(), #nn.ReLU(),
        #     nn.Linear(
        #         self.config.hidden_size,
        #         self.config.hidden_size,
        #     ),
        #     nn.BatchNorm1d(self.config.hidden_size),###########
        #     nn.ReLU(), # reeeal nn.ReLU(),
        #     nn.Linear(
        #         self.config.hidden_size,
        #         self.feature_dim,
        #     ),
        # )
        # self.cluster_projector = nn.Sequential(
        #                                     # nn.BatchNorm1d(self.config.hidden_size),
        #                                     # nn.ReLU(),
        #     nn.BatchNorm1d(self.config.hidden_size),###########
        #     nn.ReLU(), #nn.ReLU(),
        #     nn.Linear(
        #         self.config.hidden_size,   # nn.BatchNorm1d(self.config.hidden_size),
                                            # nn.ReLU(),
        #         self.config.hidden_size,
        #     ),
        #                                     # nn.BatchNorm1d(self.config.hidden_size),
        #                                     # reeeal nn.ReLU(),
        #     nn.BatchNorm1d(self.config.hidden_size),###########
        #     nn.ReLU(), #nn.ReLU(),
        #     nn.Linear(
        #         self.config.hidden_size,
        #         self.cluster_num,
        #     )
        
    @timer
    def single_forward(self, x, pooler_type):
        # print(len(x), x[0][0], type(x[0]), len(x[0]), type(x[0][0]), print(x[0][0]), sep='\n###')
        # print(x,end="\n" + 30 * "$")
        # @timer
        # def do_mlm_stuff(_tokens):
        #     mlm_input_ids, mlm_labels = self.mlm_mix_head.mask_tokens(
        #         inputs=_tokens.input_ids, tokenizer=self.tokenizer
        #     )

        #     mlm_last_hidden_state = self.backbone(
        #         input_ids=mlm_input_ids.to("cuda"),
        #         attention_mask=mask,
        #         output_hidden_states=False,
        #         return_dict=True,
        #     ).last_hidden_state
        #     del mlm_input_ids
        #     return mlm_last_hidden_state, mlm_labels

        torch.cuda.empty_cache()
        tokens = self.tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        # del x
        mask = tokens["attention_mask"].to("cuda")

        if self.mlm_mix_head is not None:
            # mlm_last_hidden_state, mlm_labels = do_mlm_stuff(tokens)
            torch.cuda.empty_cache()

        tokens.input_ids = tokens.input_ids.to("cuda")

        model_output = self.backbone(
            **tokens.to("cuda"),
            output_hidden_states=False,
            return_dict=True,  # made it false #TODO
        )
        # del tokens

        # pooled = mean_pooling(model_output=model_output,
        #
        #                 attention_mask=tokens['attention_mask'])
        h = self.pooler(
            outputs=model_output,
            pooler_type=pooler_type,
            attention_mask=mask,
        )
        # del model_output
        torch.cuda.empty_cache()

        z = normalize(self.instance_projector(h), dim=1)

        c = self.cluster_projector(h)
        # log_it(*c[:3].tolist(), c.shape, h.shape, sep='\n')
        c = nn.functional.softmax(c, dim=1)
        # log_it(*c[:3].tolist(), c.shape, h.shape, sep='\n')
        # log_it("#" * 30)

        return z, c, 0, 0
       
    def pooler(self, outputs, pooler_type, attention_mask):
        # outputs = model(**batch, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        # Apply different poolers
        if pooler_type == "cls":
            pass
            # There is a linear+activation layer after CLS representation
            # return pooler_output
        elif pooler_type == "cls_before_pooler":
            return last_hidden[:, 0]
        elif pooler_type == "avg":
            return self.mean_pooling(
                model_output=outputs, attention_mask=attention_mask
            )
            # return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        elif pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden)
                / 2.0
                * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden)
                / 2.0
                * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

    @timer
    def hf_forward(self, x, pooler_type):
        torch.cuda.empty_cache()
        tokens = self.tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        ).to("cuda")

        model_output = self.backbone(
            **tokens, output_hidden_states=True, return_dict=True
        )
        # pooled = mean_pooling(model_output=model_output,
        #                attention_mask=tokens['attention_mask'])
        return self.pooler(
            outputs=model_output,
            pooler_type=pooler_type,
            attention_mask=tokens["attention_mask"],
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(
                -1
            )  # wandb sync C:\Users\ComInSys\Desktop\vahidi-workspace\2022-IJCV-TCL-fork\tcl-official-repo\wandb\offline-run-20230921_010356-b3uga5x8
            .expand(token_embeddings.size())
            .float()
        )
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @timer
    def mlm_forward(self, x, pooler_type):
        # @timer
        # def do_mlm_stuff(_tokens):
        #     mlm_input_ids, mlm_labels = self.mlm_mix_head.mask_tokens(
        #         inputs=_tokens.input_ids, tokenizer=self.tokenizer
        #     )

        #     mlm_last_hidden_state = self.backbone(
        #         input_ids=mlm_input_ids.to("cuda"),
        #         attention_mask=mask,
        #         output_hidden_states=False,
        #         return_dict=True,
        #     ).last_hidden_state
        #     del mlm_input_ids
        #     return mlm_last_hidden_state, mlm_labels

        torch.cuda.empty_cache()
        tokens = self.tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )
        del x
        mask = tokens["attention_mask"].to("cuda")

        if self.mlm_mix_head is not None:
            # mlm_last_hidden_state, mlm_labels = do_mlm_stuff(tokens)
            torch.cuda.empty_cache()

        tokens.input_ids = tokens.input_ids.to("cuda")

        model_output = self.backbone(
            **tokens.to("cuda"),
            output_hidden_states=False,
            return_dict=True,  # made it false #TODO
        )
        del tokens

        # pooled = mean_pooling(model_output=model_output,
        #                attention_mask=tokens['attention_mask'])
        h = self.pooler(
            outputs=model_output,
            pooler_type=pooler_type,
            attention_mask=mask,
        )
        del model_output
        if self.mlm_mix_head is not None:
            raise ValueError("MLM WTF?????")
            # return h, mlm_last_hidden_state, mlm_labels.to("cuda")
        else:
            return h

    @timer
    def forward(self, x_i, x_j):
        """h_i = self.backbone.encode(x_i, batch_size=len(x_i),
                                   convert_to_numpy=False,
                                   convert_to_tensor=True)
        h_j = self.backbone.encode(x_j, batch_size=len(x_j),
                                   convert_to_numpy=False,
                                   convert_to_tensor=True)"""

        # h_i = self.hf_forward(x_i, pooler_type="avg")
        # h_j = self.hf_forward(x_j, pooler_type="avg")

        out_i = self.mlm_forward(x_i, pooler_type="avg")

        torch.cuda.empty_cache()

        out_j = self.mlm_forward(x_j, pooler_type="avg")

        if self.mlm_mix_head is not None:
            h_i, mlm_last_hidden_i, mlm_labels_i = out_i
            h_j, mlm_last_hidden_j, mlm_labels_j = out_j
            torch.cuda.empty_cache()

            mixcse_loss, scaled_mlm_loss = self.mlm_mix_head(
                h1=h_i,
                h2=h_j,
                mlm_last_hidden1=mlm_last_hidden_i,
                mlm_last_hidden2=mlm_last_hidden_j,
                mlm_labels1=mlm_labels_i,
                mlm_labels2=mlm_labels_j,
            )
        else:
            h_i = out_i
            h_j = out_j
        torch.cuda.empty_cache()

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        if self.mlm_mix_head is not None:
            return z_i, z_j, c_i, c_j, mixcse_loss, scaled_mlm_loss
        return z_i, z_j, c_i, c_j, 0, 0

    # @timer
    def forward_c(self, x):
        h = self.backbone.encode(
            x,
            batch_size=len(x),
            convert_to_numpy=False,
            convert_to_tensor=True,
        )
        c = self.cluster_projector(h)
        c = torch.nn.functional.softmax(c, dim=1)
        return c

    # @timer
    def forward_c_psd(self, x_j, pseudo_index):
        x = []
        size = len(x_j)
        for i in range(size):
            if pseudo_index[i]:
                x.append(x_j[i])
        h = self.backbone.encode(
            x,
            batch_size=len(x),
            convert_to_numpy=False,
            convert_to_tensor=True,
        )
        c = self.cluster_projector(h)
        c = torch.nn.functional.softmax(c, dim=1)
        return c

    # @timer
    def forward_cluster(self, x):
        h = self.backbone.encode(
            x,
            batch_size=len(x),
            convert_to_numpy=False,
            convert_to_tensor=True,
        )
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

    @torch.no_grad()
    @timer
    def forward_cluster_feature_return(self, x):
        """h = self.backbone.encode(x, batch_size=len(x),
        convert_to_numpy=False, "cls_before_pooler"
        convert_to_tensor=True)"""

        h = self.hf_forward(x, pooler_type="avg")
        c_features = self.cluster_projector(h)
        clusters = torch.argmax(c_features, dim=1)
        return clusters, c_features
