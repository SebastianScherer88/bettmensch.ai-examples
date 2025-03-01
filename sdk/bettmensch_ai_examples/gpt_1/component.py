from bettmensch_ai.pipelines.component import (
    as_component,
    as_torch_ddp_component,
)
from bettmensch_ai.pipelines.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
)
from .src.train import tokenize_text, pretrain, size_in_gb


def get_source_data_split(
    data_out: OutputArtifact = None,
):
    from datasets import load_dataset

    data = load_dataset(
        "bookcorpus/bookcorpus", split="train", trust_remote_code=True
    )
    data.save_to_disk(data_out.path)


def get_tokenized_data_split_and_tokenizer(
    source_data_split: InputArtifact,
    start_index: InputParameter = 0,
    end_index: InputParameter = -1,
    sequence_length: InputParameter = 512,
    unk_token: InputParameter = "<unk>",
    bos_token: InputParameter = "<s>",  # only needed for fine-tuning tasks
    eos_token: InputParameter = "<e>",  # only needed for fine-tuning tasks
    sep_token: InputParameter = "<$>",  # only needed for fine-tuning tasks
    pad_token: InputParameter = "<p>",  # only needed for fine-tuning tasks
    batch_size: InputParameter = 50,
    display_step: InputParameter = 2000,
    tokenized_data_out: OutputArtifact = None,
    tokenizer_out: OutputArtifact = None,
):

    import pickle

    from datasets import Dataset
    from transformers import OpenAIGPTConfig, OpenAIGPTTokenizerFast

    data = Dataset.load_from_disk(source_data_split.path)

    print(f"Number of observations in data set: {len(data)}")
    print(f"Size of data set in memory (in GB): {size_in_gb(data)}")

    config = OpenAIGPTConfig()
    tokenizer = OpenAIGPTTokenizerFast.from_pretrained(config.model_type)
    tokenizer.add_special_tokens(
        {
            "unk_token": unk_token,
            "bos_token": bos_token,  # only needed for fine-tuning tasks
            "eos_token": eos_token,  # only needed for fine-tuning tasks
            "sep_token": sep_token,  # only needed for fine-tuning tasks
            "pad_token": pad_token,  # only needed for fine-tuning tasks
        }
    )

    tokenized_data = tokenize_text(
        data[start_index:end_index]["text"],
        tokenizer=tokenizer,
        batch_size=batch_size,
        display_step=display_step,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    print(
        f"Number of observations in tokenized data set: {len(tokenized_data)}"
    )
    print(
        "Size of tokenized data set in memory (in GB): "
        f"{size_in_gb(tokenized_data)}"
    )

    with open(tokenized_data_out.path, "wb") as token_file:
        pickle.dump(tokenized_data, token_file)

    tokenizer.save_pretrained(tokenizer_out.path)


def pretrain_and_checkpoint(
    tokenized_train: InputArtifact,
    tokenized_validation: InputArtifact,
    tokenizer: InputArtifact,
    architecture: InputParameter,
    sequence_length: InputParameter = 512,
    dim_embed: InputParameter = 768,
    n_decoder_layers: InputParameter = 12,
    n_heads: InputParameter = 12,
    dropout: InputParameter = 0.1,
    learning_rate: InputParameter = 0.00025,
    weight_decay: InputParameter = 0.01,
    n_epochs: InputParameter = 10,
    batch_size: InputParameter = 2,
    display_step: InputParameter = 20,
    verbose: InputParameter = False,
):

    pretrain(
        train_data_path=tokenized_train.path,
        validation_data_path=tokenized_validation.path,
        tokenizer_path=tokenizer.path,
        architecture=architecture,
        sequence_length=sequence_length,
        dim_embed=dim_embed,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        batch_size=batch_size,
        display_step=display_step,
        verbose=verbose,
    )


get_source_data_split_factory = as_component(get_source_data_split)

get_tokenized_data_split_and_tokenizer_factory = as_component(
    get_tokenized_data_split_and_tokenizer
)

pretrain_and_checkpoint_factory = as_torch_ddp_component(
    pretrain_and_checkpoint
)
