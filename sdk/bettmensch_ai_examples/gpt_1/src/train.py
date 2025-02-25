import pickle
import sys
from datetime import datetime as dt
import torch.amp
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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# config = OpenAIGPTConfig()
# config

def seed(value: int):
    import numpy, random, torch
    numpy.random.seed(value)
    torch.manual_seed(value)
    random.seed(value)

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
        self.seed = config['seed']
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
    scaler: Optional[torch.amp.grad_scaler.GradScaler] = None
    scheduler: torch.optim.lr_scheduler.SequentialLR
    summary_writer: torch.utils.tensorboard.writer.SummaryWriter
    n_completed_epochs: int = 0
    train_loss_history: List[float] = []
    validation_loss_history: List[float] = []
    
    def __init__(self, config: Dict):
        self.config = config
        self.summary_writer = torch.utils.tensorboard.writer.SummaryWriter(
            "runs/gpt1_pretrain_{}".format(dt.now().strftime("%Y%m%d_%H%M%S"))
        )
    
    def tb_log(self,*args,**kwargs):
        self.summary_writer.add_scalars(*args,**kwargs)

    def log_step(self, batch_index: int, batch_size: int, n_batches: int, step_loss: float):
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
            self.n_completed_epochs * n_batches + batch_index + 1
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

        self.tb_log(
            "Learning rate",
            {"LR": self.optimizer.param_groups[0]['lr']},
             global_batch_index
        )

    def log_epoch(self, train_loss: float, validation_loss: float):
        """Displays epoch level progress on console and logs to tensorboard.

        Args:
            train_loss (float): The training loss for the current epoch
            validation_loss (float): The validation loss for the current epoch
        """
        
        # display and log evaluation data
        print(
            f"{now()} | Train loss {train_loss} for last step of epoch"
            f" {self.n_completed_epochs+1}"
        )
        print(
            f"{now()} | Validation loss {validation_loss} for epoch {self.n_completed_epochs+1}"
        )

        self.tb_log(
            "Epoch train vs. validation loss",
            {"Training": train_loss, "Validation": validation_loss},
            self.n_completed_epochs + 1,
        )
        self.summary_writer.flush()

    def run_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        train: bool = True,
    ) -> float:

        assert self.model is not None
        self.model.to(0)

        if train:
            assert self.optimizer is not None
            self.model.train()
            if self.n_completed_epochs == 0:
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
        total_loss = []

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

            # Make predictions for this batch
            with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda",dtype=torch.float16,enabled=self.config['training']['use_amp']):
                logits, loss = self.model(inputs, mask, target)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

                if self.config['training']['scale_gradients']:
                    # Compute the loss and its gradients
                    self.scaler.scale(loss).backward()

                    # clip
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

                    # update model weights and learning rate
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:
                    # Compute the loss and its gradients
                    loss.backward()

                    # clip
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

                    # update model weights and learning rate
                    self.optimizer.step()
                    self.scheduler.step()
                
            # Gather data and report
            running_loss += loss.item()

            if batch_index % display_step == (display_step - 1):

                # average loss over last display_step batches
                step_loss = running_loss / display_step
                total_loss.append(step_loss)

                self.log_step(batch_index,batch_size,n_batches,step_loss)
                running_loss = 0.0

        return sum(total_loss) / len(total_loss)
    
    def save_checkpoint(self, path: str):
        """Creates checkpoint from which training can be resumed.

        Args:
            path (str): The location on disk to create the checkpoint.
        """
        torch.save(
            {
                'trainer_config':self.config,
                'n_completed_epochs': self.n_completed_epochs,
                'train_loss_history': self.train_loss_history,
                'validation_loss_history': self.validation_loss_history,
                'model_config': self.model.config,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            },
            path
        )

    def load_checkpoint(self, path: str):
        """Loads created checkpoint to resume training from.

        Args:
            path (str): The location on disk to load the checkpoint from.
        """
        checkpoint_dict = torch.load(path, weights_only=True)
        
        # trainer config & state
        self.config = checkpoint_dict['trainer_config']
        self.n_completed_epochs = checkpoint_dict['n_completed_epochs']
        self.train_loss_history = checkpoint_dict['train_loss_history']
        self.validation_loss_history = checkpoint_dict['validation_loss_history']
        
        # model
        self.model = GPT1Pretrain(**checkpoint_dict['model_config'])
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])

        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),**self.config['optimizer'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])

        # scaler
        if self.config['training']['scale_gradients']:
            self.scaler = torch.amp.grad_scaler.GradScaler("cuda")

        # scheduler
        linear_scheduler = LinearLR(self.optimizer, **self.config['scheduler']['linear']),#start_factor=0.0001, end_factor=1, total_iters=50)
        cosine_scheduler = CosineAnnealingLR(self.optimizer,**self.config['scheduler']['cosine'])#T_max=25,eta_min=0)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[linear_scheduler,cosine_scheduler],**self.config['scheduler']['sequential'])#milestones=[50])
        self.scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])

    def train(
            self,
            train_loader: torch.utils.data.DataLoader,
            model: Optional[GPT1Pretrain] = None,
            validation_loader: Optional[torch.utils.data.DataLoader] = None,
            checkpoint: Optional[str] = None,
        ):
        """Main training loop.

        Args:
            model (GPT1Pretrain): _description_
            train_loader (torch.utils.data.DataLoader): _description_
            validation_loader (torch.utils.data.DataLoader): _description_
        """

        # if training from scratch, set attributes needed for training.
        # if training from checkpoint, set attributes from checkpoint
        if model and (checkpoint is None):
            self.model = model
            self.optimizer = torch.optim.AdamW(
                model.parameters(), 
                **self.config['optimizer']
            )
            if self.config['training']['scale_gradients']:
                self.scaler = torch.amp.grad_scaler.GradScaler("cuda")
            linear_scheduler = LinearLR(self.optimizer, **self.config['scheduler']['linear'])#start_factor=0.0001, end_factor=1, total_iters=50)
            cosine_scheduler = CosineAnnealingLR(self.optimizer,**self.config['scheduler']['cosine'])#T_max=25,eta_min=0)
            self.scheduler = SequentialLR(self.optimizer, schedulers=[linear_scheduler,cosine_scheduler],**self.config['scheduler']['sequential'])#milestones=[50])
        elif (model is None) and checkpoint:
            self.load_checkpoint(checkpoint)
        else:
            raise ValueError(f"Only one of `model` and `checkpoint` arguments"
                             " can be provided.")

        while self.n_completed_epochs < self.config['training']["n_epochs"]:
            print(f"EPOCH {self.n_completed_epochs + 1}:")

            # run training and retrieve loss averaged over last 1000 batches of
            # training split
            train_loss = self.run_epoch(
                data_loader=train_loader,
                train=True,
            )
            self.train_loss_history.append(train_loss)

            if validation_loader is not None:
                validation_loss = self.run_epoch(
                    data_loader=validation_loader,
                    train=False,
                )
            else:
                validation_loss = -1
            self.validation_loss_history.append(validation_loss)

            # log progress using current epoch index
            self.log_epoch(train_loss, validation_loss)

            # update epoch counter to reflect newly completed epoch before 
            # checkpointing            
            self.n_completed_epochs += 1
            self.save_checkpoint(str(self.n_completed_epochs))

def pretrain(
    pretrain_config: GPT1PretrainConfig,
):

    seed(pretrain_config.seed)
    train_data = TokenizedBookCorpusOpenSplit(pretrain_config.data['train']['path'])

    train_loader = torch.utils.data.DataLoader(
        train_data, collate_fn=TokenizedBookCorpusOpenSplit.collate_batch, **pretrain_config.data['train']['dataloader']
    )

    if pretrain_config.data['validation']['use']:
        validation_data = TokenizedBookCorpusOpenSplit(pretrain_config.data['validation']['path'])
        validation_loader = torch.utils.data.DataLoader(
            validation_data, collate_fn=TokenizedBookCorpusOpenSplit.collate_batch, **pretrain_config.data['validation']['dataloader'],generator=torch.Generator(device="cuda")
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
