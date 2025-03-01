import pickle
import sys
from datetime import datetime as dt
import torch.amp
import yaml
from yaml import Loader
from typing import Any, List, Tuple, Dict, Optional, Union
import random

import GPUtil
import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
import torch.utils.tensorboard.summary
from .model import GPT
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


def tokenize_text(
    samples: List[str],
    tokenizer: OpenAIGPTTokenizerFast,
    batch_size: int = 100,
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
    tokenized_text = []

    n_steps = len(samples) // batch_size

    for step, start in enumerate(range(0, len(samples), batch_size)):
        if step % display_step == 0:
            print(f"{now()} | Step {step+1}/{n_steps+1}")
        end = start + batch_size
        batch = samples[start:end]
        tokenized_batch = tokenizer(batch, **tokenizer_kwargs)["input_ids"]

        for tokenized_sample in tokenized_batch:
            tokenized_text.extend(tokenized_sample)

    return tokenized_text

class TokenizedBookCorpusOpenSplit(torch.utils.data.Dataset):

    context_size: int

    def __init__(self, path, context_size: int = 512):

        self.context_size = context_size
        self.counter = 0

        with open(path, "rb") as token_file:
            self.tokenized_text = pickle.load(token_file)

    def __len__(self) -> int:
        return int(len(self.tokenized_text) / self.context_size)

    def __getitem__(self, idx) -> Tuple[List[int],List[int]]:
        
        input_start = idx * self.context_size
        target_start = input_start + 1
        input_tokens = self.tokenized_text[input_start:input_start+self.context_size]
        target_tokens = self.tokenized_text[target_start:target_start+self.context_size]
        return input_tokens, target_tokens

    @staticmethod
    def collate_batch(samples: List[Tuple[List[int],List[int]]]) -> Tuple[
        Float[torch.Tensor, "n_batch n_tokens"],
        Float[torch.Tensor, "n_batch n_tokens"],
    ]:
        
        inputs_list = [torch.tensor(sample[0], dtype=torch.int) for sample in samples]
        inputs_tensor = torch.stack(inputs_list)

        target_list = [torch.tensor(sample[1], dtype=torch.int64) for sample in samples]
        target_tensor = torch.stack(target_list)

        return inputs_tensor, target_tensor


class GPTPretrainConfig:

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

class GPTTrainer:

    # init
    model: GPT
    train_loader: torch.utils.data.DataLoader
    train_batch_size: int
    validation_loader: Optional[torch.utils.data.DataLoader] = None
    validation_batch_size: Optional[int] = None
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.SequentialLR
    summary_writer: torch.utils.tensorboard.writer.SummaryWriter
    
    # state
    train_batch_size: int
    validation_batch_size: int
    n_completed_steps: int = 0
    train_loss_history: List[float] = []
    validation_loss_history: List[float] = []
    
    def __init__(self, config: Dict, model: GPT, train_loader: torch.utils.data.DataLoader, validation_loader: Optional[torch.utils.data.DataLoader] = None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.train_batch_size = train_loader.batch_size
        if validation_loader:
            self.validation_loader = validation_loader
            self.validation_batch_size = validation_loader.batch_size
        self.summary_writer = torch.utils.tensorboard.writer.SummaryWriter(
            "runs/gpt1_pretrain_{}".format(dt.now().strftime("%Y%m%d_%H%M%S"))
        )
        self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                **self.config['optimizer']
            )
        linear_scheduler = LinearLR(self.optimizer, **self.config['scheduler']['linear'])
        cosine_scheduler = CosineAnnealingLR(self.optimizer,**self.config['scheduler']['cosine'])
        self.scheduler = SequentialLR(self.optimizer, schedulers=[linear_scheduler,cosine_scheduler],**self.config['scheduler']['sequential'])

    def run_step(
        self,
        batch,
        train: bool = True,
    ) -> float:

        if train:
            self.model.train()
        else:
            self.model.eval()

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        inputs, target = batch
            
        # Every data instance is an input + label pair
        inputs = inputs.to(0)
        target = target.to(0)

        # Make predictions for this batch
        logits, loss = self.model(inputs, target)

        if train:
            self.optimizer.zero_grad(set_to_none=True)

            # Compute the loss and its gradients
            loss.backward()

            # clip
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['training']['grad_clip_norm'])

            # update model weights and learning rate
            self.optimizer.step()
            self.scheduler.step()

        return loss.item()
    
    def eval_step(self):
        
        if self.n_completed_steps % self.config['training']['eval_steps'] != 0:
            return
        
        n_steps_eval = self.config['training']['n_steps_eval']
        train_iter = iter(self.train_loader)
        train_loss = []

        for i in range(n_steps_eval):
            batch = next(train_iter)
            train_loss.append(self.run_step(batch=batch,train=False))

        train_loss = sum(train_loss)/n_steps_eval
        self.train_loss_history.append(train_loss)

        if self.validation_loader:
            validation_iter = iter(self.validation_loader)
            validation_loss = []
            for i in range(n_steps_eval):
                batch = next(validation_iter)
                validation_loss.append(self.run_step(batch=batch,train=False))

            validation_loss = sum(validation_loss)/n_steps_eval
            self.validation_loss_history.append(validation_loss)
    
    def log_step(self):
        """Displays & logs step level progress.
        """

        if self.n_completed_steps % self.config['training']['eval_steps'] != 0:
            return
        
        train_loss = self.train_loss_history[-1]
        validation_loss = self.validation_loss_history[-1] if self.validation_loss_history else -1
        learning_rate = self.optimizer.param_groups[0]['lr']
        
        print(
            f"{now()} | Step {self.n_completed_steps}/{self.config['training']['n_steps']}"
            f" | Train loss: {train_loss} | Validation loss: {validation_loss} | Learning rate: {learning_rate}"
        )

        self.summary_writer.add_scalars(
            "Step loss", {"Train": train_loss, "Validation": validation_loss}, self.n_completed_steps
        )
        gpu = GPUtil.getGPUs()[0]
        self.summary_writer.add_scalars(
            "Batch GPU utilization",
            {"Load": gpu.load, "Memory": gpu.memoryUtil},
            self.n_completed_steps,
        )

        self.summary_writer.add_scalars(
            "Learning rate",
            {"LR": learning_rate},
             self.n_completed_steps
        )

    def checkpoint(self):
        """Creates checkpoint from which training can be resumed.

        Args:
            path (str): The location on disk to create the checkpoint.
        """

        if self.n_completed_steps % self.config['training']['checkpoint_steps'] != 0:
            return
        
        torch.save(
            {
                'trainer_config':self.config,
                'n_completed_steps': self.n_completed_steps,
                'train_loss_history': self.train_loss_history,
                'validation_loss_history': self.validation_loss_history,
                'model_config': self.model.config,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            },
            self.n_completed_steps
        )

    @classmethod
    def from_checkpoint(cls, path: str, train_loader: torch.utils.data.DataLoader, validation_loader: Optional[torch.utils.data.DataLoader] = None) -> 'GPTTrainer':
        """Loads created checkpoint to resume training from.

        Args:
            path (str): The location on disk to load the checkpoint from.
        """
        checkpoint_dict = torch.load(path, weights_only=True)
        
        # model
        model = GPT(**checkpoint_dict['model_config'])
        model.load_state_dict(checkpoint_dict['model_state_dict'])

        instance = cls(config=checkpoint_dict['trainer_config'], model=model, train_loader=train_loader, validation_loader=validation_loader)
        instance.n_completed_steps = checkpoint_dict['n_completed_steps']
        instance.train_loss_history = checkpoint_dict['train_loss_history']
        instance.validation_loss_history = checkpoint_dict['validation_loss_history']
        instance.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        instance.scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])

        return instance

    def train(self):
        """Main training loop.

        Args:
            model (GPT1Pretrain): _description_
            train_loader (torch.utils.data.DataLoader): _description_
            validation_loader (torch.utils.data.DataLoader): _description_
        """
        assert self.model is not None
        self.model.init_weights()
        self.model.to(0)

        train_iter = iter(self.train_loader)
        validation_iter = iter(self.validation_loader) if self.validation_loader else None

        while self.n_completed_steps < self.config['training']["n_steps"]:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
            finally:
                if self.n_completed_steps >= self.config['training']['n_steps']:
                    break

            # run step in training mode
            loss = self.run_step(
                batch=batch,
                train=True,
            )
            #self.train_loss_history.append(loss)

            self.n_completed_steps += 1
            self.eval_step()
            self.log_step()
            self.checkpoint()

def pretrain(
    pretrain_config: GPTPretrainConfig,
):

    seed(pretrain_config.seed)
    train_data = TokenizedBookCorpusOpenSplit(**pretrain_config.data['train']['dataset'])

    train_loader = torch.utils.data.DataLoader(
        train_data, collate_fn=TokenizedBookCorpusOpenSplit.collate_batch, **pretrain_config.data['train']['dataloader']
    )

    if pretrain_config.data['validation']['use']:
        validation_data = TokenizedBookCorpusOpenSplit(**pretrain_config.data['validation']['dataset'])
        validation_loader = torch.utils.data.DataLoader(
            validation_data, collate_fn=TokenizedBookCorpusOpenSplit.collate_batch, **pretrain_config.data['validation']['dataloader'],
        )
    else:
        validation_loader = None

    tokenizer = OpenAIGPTTokenizerFast.from_pretrained(pretrain_config.tokenizer['path'])
    
    gpt1 = GPT(
        n_vocab=tokenizer.vocab_size,
        **pretrain_config.model['architecture'],
    )

    trainer = GPTTrainer(
        pretrain_config.trainer,
        model=gpt1,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )

    trainer.train()
