
from io import BytesIO
import boto3
import redis
import torch
import os
import botocore.config
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional
import lz4.frame

class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.model_name = checkpoint_dir.stem
        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        # some checkpoints have both files, `.json` takes precedence
        if (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                eos_token = config.get("eos_token")
                if bos_token is not None and isinstance(bos_token, dict):
                    bos_token = bos_token.get("content")
                if eos_token is not None and isinstance(eos_token, dict):
                    eos_token = eos_token.get("content")
                self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
                self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")

        elif (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()
        else:
            raise NotImplementedError

        # NOTE: A temporary fix until it's resolved on Tokenizers side.
        # LlaMA tokenizer strips leading spaces if to decode a single token at a time.
        # https://github.com/huggingface/transformers/issues/31643
        self.apply_decoding_fix = None
        if (config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            with open(config_path, encoding="utf-8") as fp:
                self.apply_decoding_fix = "LlamaTokenizer" in json.load(fp)["tokenizer_class"]

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if not (tokenizer_config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False
        with open(tokenizer_config_path, encoding="utf-8") as fp:
            config = json.load(fp)
        # for LlaMA-3 tokenizer there is no `add_bos_token` at all and `tokenizer_class` is only
        # `PreTrainedTokenizerFast`
        if checkpoint_dir.stem.startswith("Meta-Llama-3"):
            return True
        if "add_bos_token" in config:
            return config["add_bos_token"]
        # if `add_bos_token` isn't in the config file, but LLaMA tokenizer is used - return True.
        # ex: https://huggingface.co/stabilityai/StableBeluga2/blob/main/tokenizer_config.json#L2
        return config.get("tokenizer_class") == "LlamaTokenizer"

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string).ids
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError(f"`{self.backend}` is not supported.")
        if tokens is None:
            raise ValueError("`self.processor` returned tokens of None value.")

        if bos or (bos is None and self.use_bos):
            if self.bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined bos token.")
            if not tokens or tokens[0] != self.bos_id:
                tokens = [self.bos_id] + tokens
        # if the processor misbehaves and adds `bos` token no matter what
        elif tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]

        if eos and (not tokens or tokens[-1] != self.eos_id):
            tokens = tokens + [self.eos_id]

        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        if len(tokens) == 1 and self.apply_decoding_fix:
            dummy_token_id = 33  # \x1e
            dummy_token = self.processor.decode([dummy_token_id])
            return self.processor.decode([dummy_token_id] + tokens)[len(dummy_token) :]
        return self.processor.decode(tokens)


# Create the S3 client with the custom config
s3_client = boto3.client('s3', config=botocore.config.Config(
    max_pool_connections=51
))
redis_client = None

# # Set the TRANSFORMERS_CACHE environment variable
# os.environ['HF_HOME'] = '/tmp'
# tokenizer=AutoTokenizer.from_pretrained('pythia-14m-tokenizer')

tokenizer = Tokenizer("/var/task/pythia-14m-tokenizer")

def gen_samples(tokenized_data, block_size=512):
        # Create a list to hold samples of size self.context_size
        samples = []
        for item in tokenized_data:
            chunk_size = block_size + 1  # Define the chunk size
            # Split ids into chunks of size block_size+1
            for i in range(0, item.size(0), chunk_size):
                # Extract a chunk from the ids
                chunk = item[i:i + chunk_size]
                # Pad the last chunk if it is smaller than block_size+1
                if chunk.size(0) < chunk_size:
                    padding_length = chunk_size - chunk.size(0)
                    padding = torch.full((padding_length,), fill_value=0, dtype=torch.long)
                    chunk = torch.cat((chunk, padding))

                input_ids = chunk[0:block_size].contiguous().long()
                targets = chunk[1:block_size + 1].contiguous().long()
                samples.append((input_ids, targets))
        return samples

def prepare_data_chunk(bucket_name, data_path, s3_client):
    global tokenizer
    response = s3_client.get_object(Bucket=bucket_name, Key=data_path)
    data_chunk = response['Body'].read()
    data_chunk = BytesIO(data_chunk)
    tokenized_docs = []
    index = 0
    while True:
        offset = (1 + (index - 0) if index >= 0 else index + 1) * 4
        # Read the entire content of the binary file
        data_chunk.seek(offset)
        pair = data_chunk.read(8)
        begin, end = np.frombuffer(pair, np.uint32)
        if begin.item() == len(data_chunk.getvalue()):
            break
        data_chunk.seek(begin)
        raw_item_data = data_chunk.read(end - begin)

        shift_idx = 4
        sizes = np.frombuffer(raw_item_data[:shift_idx], np.uint32)
        data = ""
        for size, data_format in zip(sizes, 'str'):
            # size = size.item()
            data_bytes = raw_item_data[shift_idx : shift_idx + size]
            data += data_bytes.decode('utf-8')
            shift_idx += size
        index += 1
        tokenized_docs.append(tokenizer.encode(data, eos=True))
    
    tokenized_samples = gen_samples(tokenized_docs, block_size=512)
    
    return tokenized_samples

def _torch_tenors_to_bytes(tokenized_data) -> tuple:
    with BytesIO() as buffer:
            torch.save(tokenized_data, buffer)
            compressed_byte_data = lz4.frame.compress(buffer.getvalue())
    return compressed_byte_data


def lambda_handler(event, context):
    global s3_client
    global redis_client
    try:
        task = event.get('task')
        if task == 'warmup':
            return {'success': True, 'message': 'function warmed'}
        
        bucket_name = event.get('bucket_name')
        batch_samples = event.get('batch_samples')
        chunk_id = event.get('batch_id')
        cache_address = event.get('cache_address', None)
        
        if not all([bucket_name, batch_samples, chunk_id, cache_address]):
            return {'success': False, 'message': 'Missing parameters'}
        
        cache_host, cache_port = cache_address.split(":")
        if redis_client is None:
            redis_client = redis.StrictRedis(host=cache_host, port=int(cache_port))

        for sample in batch_samples:
            data_path, _ = sample
            tokenized_samples = prepare_data_chunk(bucket_name, data_path, s3_client)
            tokens_as_bytes = _torch_tenors_to_bytes(tokenized_samples)
            redis_client.set(chunk_id, tokens_as_bytes)

        return {
            'success': True,
            'is_cached': True,
            'message': f"Successfully cached '{chunk_id}'"
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }