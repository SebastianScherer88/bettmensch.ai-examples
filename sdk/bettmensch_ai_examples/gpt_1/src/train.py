import pickle
import sys
from datetime import datetime as dt
from typing import Any, List, Tuple

import GPUtil
import torch
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
import torch.utils.tensorboard.summary
from sdk.bettmensch_ai_examples.gpt_1.src.model import GPT1Pretrain
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
        
        inputs_list = [torch.tensor(token_sample[:-1], dtype=torch.long) for token_sample in token_samples]
        inputs_tensor = torch.stack(inputs_list)

        mask_tensor = torch.ones(inputs_tensor.size(),dtype=torch.bool)

        target_list = [torch.tensor(token_sample[1:], dtype=torch.long) for token_sample in token_samples]
        target_tensor = torch.stack(target_list)

        return inputs_tensor, mask_tensor, target_tensor

def train_epoch(
    epoch_index: int,
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    model: GPT1Pretrain,
    loss_module: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    summary_writer: torch.utils.tensorboard.writer.SummaryWriter,
    display_step: int = 10,
) -> float:

    model.train()
    running_loss = 0.0
    train_step_loss = 0.0
    n_records = len(train_loader) * train_loader.batch_size

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for batch_index, (inputs, mask, target) in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs = inputs.to(0)
        mask = mask.to(0)
        target = target.to(0)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs, mask)

        # Compute the loss and its gradients
        loss_node = loss_module(outputs, target)
        loss_node.backward()

        # update model weights and learning rate
        optimizer.step()
        scheduler.step()

        # Gather data and report
        running_loss += loss_node.item()
        if batch_index % display_step == (display_step - 1):

            train_step_loss = (
                running_loss / display_step
            )  # average loss over last display_step batches

            record_index_start = batch_index * train_loader.batch_size
            record_index_end = (
                batch_index + display_step
            ) * train_loader.batch_size
            print(
                f"{now()} | Batch {batch_index+1}/{len(train_loader)}"
                f" (observations {record_index_start}-{record_index_end}/"
                f"{n_records}) loss: {train_step_loss}"
            )

            global_batch_index = (
                epoch_index * len(train_loader) + batch_index + 1
            )
            summary_writer.add_scalar(
                "Batch train loss", train_step_loss, global_batch_index
            )

            gpu = GPUtil.getGPUs()[0]
            summary_writer.add_scalars(
                "Batch GPU utilization",
                {"Load": gpu.load, "Memory": gpu.memoryUtil},
                global_batch_index,
            )

            running_loss = 0.0

    # calculate validation loss
    model.eval()

    validation_running_loss = 0

    with torch.no_grad():
        for batch_index, (inputs_val, mask_val, targets_val) in enumerate(validation_loader):
            inputs_val = inputs_val.to(0)
            mask_val = mask_val.to(0)
            targets_val = targets_val.to(0)
            output_val = model(inputs_val, mask_val)
            loss_node = loss_module(output_val, targets_val)
            validation_running_loss += loss_node.item()

    validation_loss = validation_running_loss / (batch_index + 1)

    return train_step_loss, validation_loss


def pretrain(
    # dataset
    train_data_path: str,
    validation_data_path: str,
    # dataloader
    batch_size: int,
    # tokenizer
    tokenizer_path: str,
    # model
    sequence_length: int = 512,
    dim_embed: int = 768,
    n_decoder_layers: int = 12,
    n_heads: int = 12,
    dropout: float = 0.1,
    # training
    n_epochs: int = 10,
    display_step: int = 10,
    verbose: bool = False,
):

    train_data = TokenizedBookCorpusOpenSplit(train_data_path)
    validation_data = TokenizedBookCorpusOpenSplit(validation_data_path)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=TokenizedBookCorpusOpenSplit.collate_batch,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, collate_fn=TokenizedBookCorpusOpenSplit.collate_batch,
    )

    tokenizer = OpenAIGPTTokenizerFast.from_pretrained(tokenizer_path)
    model = GPT1Pretrain(
        n_vocab=tokenizer.vocab_size,
        n_tokens=sequence_length,
        dim_embed=dim_embed,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        dropout=dropout,
        id="GPT1Pretrain",
    )
    model.set_io_verbosity(verbose)
    model.to(0)
    loss_fn = torch.nn.CrossEntropyLoss().to(0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00025, 
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=0.00025,
        total_steps=250000,
        anneal_strategy='cos',
        pct_start=0.10
    )
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    summary_writer = torch.utils.tensorboard.writer.SummaryWriter(
        "runs/gpt1_pretrain_{}".format(timestamp)
    )

    for epoch in range(n_epochs):
        print(f"EPOCH {epoch + 1}:")

        # run training and retrieve loss averaged over last 1000 batches of
        # training split
        train_loss, validation_loss = train_epoch(
            epoch,
            train_loader,
            validation_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            summary_writer,
            display_step,
        )

        # display and log evaluation data
        print(
            f"{now()} | Train loss {train_loss} for last step of epoch"
            f" {epoch+1}"
        )
        print(
            f"{now()} | Validation loss {validation_loss} for epoch {epoch+1}"
        )

        summary_writer.add_scalars(
            "Epoch train vs. validation loss",
            {"Training": train_loss, "Validation": validation_loss},
            epoch + 1,
        )
        summary_writer.flush()
