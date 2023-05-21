import logging
from functools import partial
from threading import Thread
from typing import Any, Dict, List, Mapping, Optional

import torch
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import cached_assets_path, snapshot_download
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from pydantic import Field
from tqdm.auto import tqdm
from transformers import TextIteratorStreamer

logger = logging.getLogger(__name__)


class MosaicML(LLM):  # mypy: allow-uninitialized
    model_id: str = "mosaicml/mpt-7b"
    tokenizer_id: str = "EleutherAI/gpt-neox-20b"

    accelerator: Any = None
    tokenizer: Any = None
    config: Any = None
    model: Any = None
    device: int = 0

    max_new_tokens: Optional[int] = Field(10000, alias="max_new_tokens")
    """The maximum number of tokens to generate."""

    do_sample: Optional[bool] = Field(True, alias="do_sample")
    """Whether to sample or not."""

    temperature: Optional[float] = Field(0.8, alias="temperature")
    """The temperature to use for sampling."""

    @property
    def _llm_type(self) -> str:
        return "mpt_7b"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_id": self.model_id},
            **{"tokenizer_id": self.tokenizer_id},
        }

    def setup(self) -> LLM:
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        download_location = snapshot_download(
            repo_id=self.model_id, use_auth_token=True, local_files_only=True
        )
        logger.info(f"[{self.model_id}] Model location: {str(download_location)}")

        offload_cache_location = cached_assets_path(
            library_name="langchain",
            namespace=self.model_id,
            subfolder="offload",
        )
        logger.info(
            f"[{self.model_id}] Offload cache location: {str(offload_cache_location)}"
        )

        self.accelerator = Accelerator(mixed_precision="bf16")

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

        self.config = AutoConfig.from_pretrained(
            "mosaicml/mpt-7b", trust_remote_code=True
        )
        # config.attn_config["attn_impl"] = "flash"

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                config=self.config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        self.model.tie_weights()

        self.model = load_checkpoint_and_dispatch(
            self.model,
            download_location,
            device_map="auto",
            no_split_module_classes=["MPTBlock"],
            offload_folder=offload_cache_location,
        )

        return self

    def _mpt_default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=True)

        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(device=f"cuda:{self.device}")

        streamer = TextIteratorStreamer(tokenizer=self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            inputs, streamer=streamer, **self._mpt_default_params()
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        text = ""
        pbar = tqdm(total=self.max_new_tokens, desc="Thinking", leave=False)
        for new_text in streamer:
            if text_callback:
                text_callback(new_text)
            text += new_text
            pbar.update(1)
        pbar.close()

        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to
            # enforce stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text
