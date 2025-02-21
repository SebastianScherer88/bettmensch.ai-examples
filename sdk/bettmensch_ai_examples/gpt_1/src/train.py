import pickle
import sys
from datetime import datetime as dt
import yaml
from yaml import Loader
from typing import Any, List, Tuple, Dict, Optional, Literal

import GPUtil
import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
import torch.utils.tensorboard.summary
from .model import GPT1Pretrain
from transformers import OpenAIGPTConfig, OpenAIGPTTokenizerFast
from jaxtyping import Float

config = OpenAIGPTConfig()
config


def size_in_gb(object: Any) -> float:
    return sys.getsizeof(object) / 1024**3


def now() -> str:
    return dt.now().strftime("%Y-%m-%d %H:%M:%S")


def tokenize_for_language_modelling(
    samples: List[str],
    tokenizer: OpenAIGPTTokenizerFast,
    batch_size: int = 100,
    length: int = 512,
    display_step: int = 100,
    **tokenizer_kwargs,
) -> List[str]:
    """Tokenization utility to convert a list of strings into a list
    of sequences of tokens of size length+1.

    Args:
        samples (List[str]): The corpus that needs tokenization.
        tokenizer (OpenAIGPTTokenizerFast): The huggingface tokenizer used.
            Should be the OpenAIGPTTokenizerFast, but can be any tokenizer.
        batch_size (int, optional): The batch used during tokenization for
            speed reasons. Defaults to 100.
        length (int, optional): The size of the context window. Defaults to 
            512.
        display_step (int, optional): The number of batches between progress
            displays. Defaults to 100.

    Returns:
        List[str]: The tokenized corpus.
    """
    tokenized_text_samples = []
    tokenized_text_sample = []

    n_steps = len(samples) // batch_size

    for step, start in enumerate(range(0, len(samples), batch_size)):
        if step % display_step == 0:
            print(f"{now()} | Step {step+1}/{n_steps+1}")
        end = start + batch_size
        batch = samples[start:end]
        tokenized_batch = tokenizer(batch, **tokenizer_kwargs)["input_ids"]

        for tokenized_sample in tokenized_batch:
            tokenized_text_sample.extend(tokenized_sample)

        while len(tokenized_text_sample) >= (length + 1):
            tokenized_text_samples.append(tokenized_text_sample[:(length + 1)])
            tokenized_text_sample = tokenized_text_sample[(length + 1):]

    return tokenized_text_samples


class TokenizedBookCorpusOpenSplit(torch.utils.data.Dataset):
    def __init__(self, token_file_path):

        with open(token_file_path, "rb") as token_file:
            self.token_data = pickle.load(token_file)

    def __len__(self) -> int:
        return len(self.token_data)

    def __getitem__(self, idx) -> List[int]:
        return self.token_data[idx]
    
    @staticmethod
    def collate_batch(token_samples: List[List[int]]) -> Tuple[
        Float[torch.Tensor, "n_batch n_tokens"],
        Float[torch.Tensor, "n_batch n_tokens"],
        Float[torch.Tensor, "n_batch n_tokens"],
    ]:
        
        inputs_list = [torch.tensor(token_sample[:-1], dtype=torch.int) for token_sample in token_samples]
        inputs_tensor = torch.stack(inputs_list)

        mask_tensor = torch.ones(inputs_tensor.size(),dtype=torch.bool)

        target_list = [torch.tensor(token_sample[1:], dtype=torch.int64) for token_sample in token_samples]
        target_tensor = torch.stack(target_list)

        return inputs_tensor, mask_tensor, target_tensor


class GPT1PretrainConfig:

    def __init__(self,config: Dict):
        self.tokenizer = config['tokenizer']
        self.data = config['data']
        self.model = config['model']
        self.trainer = config['trainer']
        
    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path,"r") as yaml_file:
            config_dict = yaml.load(yaml_file,Loader=Loader)

        return cls(config_dict)

class GPT1Trainer:

    model: GPT1Pretrain
    optimizer: torch.optim.Optimizer
    summary_writer = torch.utils.tensorboard.writer.SummaryWriter(
        "runs/gpt1_pretrain_{}".format(dt.now().strftime("%Y%m%d_%H%M%S"))
    )
    
    def __init__(self, config: Dict):
        self.config = config
    
    def tb_log(self,*args,**kwargs):
        self.summary_writer.add_scalars(*args,**kwargs)

    def log_step(self, epoch_index: int, batch_index: int, batch_size: int, n_batches: int, step_loss: float):
        """Displays step level progress on console and logs to tensorboard.

        Args:
            epoch_index (int): The index of the current epoch
            batch_index (int): The index of the current batch
            batch_size (int): The batch size used during training
            n_batches (int): The number of batches in one epoch
            step_loss (float): The loss averaged over the last `step_size`
                batches.
        """
        n_records = n_batches * batch_size
        record_index_start = batch_index * batch_size
        record_index_end = record_index_start + batch_size
        
        print(
            f"{now()} | Batch {batch_index+1}/{n_batches}"
            f" (observations {record_index_start}-{record_index_end}/"
            f"{n_records}) loss: {step_loss}"
        )

        global_batch_index = (
            epoch_index * n_batches + batch_index + 1
        )
        self.tb_log(
            "Batch train loss", {"Training": step_loss}, global_batch_index
        )
        gpu = GPUtil.getGPUs()[0]
        self.tb_log(
            "Batch GPU utilization",
            {"Load": gpu.load, "Memory": gpu.memoryUtil},
            global_batch_index,
        )

    def log_epoch(self, epoch_index: int, train_loss: float, validation_loss: float):
        """Displays epoch level progress on console and logs to tensorboard.

        Args:
            epoch_index (int): The index of the current epoch
            train_loss (float): The training loss for the current epoch
            validation_loss (float): The validation loss for the current epoch
        """
        
        # display and log evaluation data
        print(
            f"{now()} | Train loss {train_loss} for last step of epoch"
            f" {epoch_index+1}"
        )
        print(
            f"{now()} | Validation loss {validation_loss} for epoch {epoch_index+1}"
        )

        self.tb_log(
            "Epoch train vs. validation loss",
            {"Training": train_loss, "Validation": validation_loss},
            epoch_index + 1,
        )
        self.summary_writer.flush()

    def run_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        train: bool = True,
        epoch_index: int = 0,
    ) -> float:

        assert self.model is not None
        self.model.to(0)

        if train:
            assert self.optimizer is not None
            self.model.train()
            self.model.init_weights()
        else:
            self.model.eval()

        n_batches = len(data_loader)
        max_batches = n_batches if self.config['training']["n_batches"] == -1 else self.config['training']["n_batches"]
        batch_size = data_loader.batch_size
        display_step = self.config['training']["display_step"]

        total_loss = 0.0
        step_loss = 0.0
        running_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for batch_index, (inputs, mask, target) in enumerate(data_loader):

            if batch_index > max_batches:
                break

            # Every data instance is an input + label pair
            inputs = inputs.to(0)
            mask = mask.to(0)
            target = target.to(0)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            logits, loss = self.model(inputs, mask, target)

            if train:
                # Compute the loss and its gradients
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1)

                # update model weights and learning rate
                self.optimizer.step()
                
            # Gather data and report
            running_loss += loss.item()

            if batch_index % display_step == (display_step - 1):

                # average loss over last display_step batches
                step_loss = running_loss / display_step
                total_loss += step_loss

                self.log_step(epoch_index,batch_index,batch_size,n_batches,step_loss)

                running_loss = 0.0

        return total_loss

    def train(
            self,
            model: GPT1Pretrain,
            train_loader: torch.utils.data.DataLoader,
            validation_loader: Optional[torch.utils.data.DataLoader] = None
        ):
        """Main training loop.

        Args:
            model (GPT1Pretrain): _description_
            train_loader (torch.utils.data.DataLoader): _description_
            validation_loader (torch.utils.data.DataLoader): _description_
        """

        # set attributes needed for training
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            **self.config['optimizer']
        )

        for epoch_index in range(self.config['training']["n_epochs"]):
            print(f"EPOCH {epoch_index + 1}:")

            # run training and retrieve loss averaged over last 1000 batches of
            # training split
            train_loss = self.run_epoch(
                data_loader=train_loader,
                train=True,
                epoch_index=epoch_index,
            )

            if validation_loader is not None:
                validation_loss = self.run_epoch(
                    data_loader=validation_loader,
                    train=False,
                    epoch_index=epoch_index
                )
            else:
                validation_loss = -1

            self.log_epoch(epoch_index, train_loss, validation_loss)

def pretrain(
    pretrain_config: GPT1PretrainConfig,
    # # dataset
    # train_data_path: str,
    # validation_data_path: str,
    # # dataloader
    # batch_size: int,
    # # tokenizer
    # tokenizer_path: str,
    # # model
    # sequence_length: int = 512,
    # dim_embed: int = 768,
    # n_decoder_layers: int = 12,
    # n_heads: int = 12,
    # dropout: float = 0.1,
    # # optimizer
    # learning_rate: float = 0.00025,
    # weight_decay: float = 0.01,
    # # training
    # n_epochs: int = 10,
    # display_step: int = 10,
    # verbose: bool = False,
):

    train_data = TokenizedBookCorpusOpenSplit(pretrain_config.data['train']['path'])
    validation_data = TokenizedBookCorpusOpenSplit(pretrain_config.data['validation']['path'])

    train_loader = torch.utils.data.DataLoader(
        train_data, collate_fn=TokenizedBookCorpusOpenSplit.collate_batch, **pretrain_config.data['train']['dataloader'],
    )

    if pretrain_config.data['validation']['use']:
        validation_loader = torch.utils.data.DataLoader(
            validation_data, collate_fn=TokenizedBookCorpusOpenSplit.collate_batch, **pretrain_config.data['validation']['dataloader'],
        )
    else:
        validation_loader = None

    tokenizer = OpenAIGPTTokenizerFast.from_pretrained(pretrain_config.tokenizer['path'])
    gpt1 = GPT1Pretrain(
        n_vocab=tokenizer.vocab_size,
        **pretrain_config.model['architecture'],
        id="GPT1Pretrain",
    )
    gpt1.set_io_verbosity(pretrain_config.model['misc']['verbose'])    

    trainer = GPT1Trainer(pretrain_config.trainer)

    trainer.train(
        model=gpt1,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )
