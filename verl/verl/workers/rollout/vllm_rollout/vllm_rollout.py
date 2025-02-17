# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
from copy import deepcopy
import torch
import torch.distributed
from tensordict import TensorDict
import traceback
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.utils.python_tool import batch_apply
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.actor_module = actor_module
        self.config = config
        self.tokenizer = tokenizer
        self.model_hf_config = model_hf_config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        self.tensor_parallel_size = tensor_parallel_size
        
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)
        
        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(actor_module,
                                    tokenizer=tokenizer,
                                    model_hf_config=model_hf_config,
                                    tensor_parallel_size=tensor_parallel_size,
                                    dtype=config.dtype,
                                    enforce_eager=config.enforce_eager,
                                    gpu_memory_utilization=config.gpu_memory_utilization,
                                    skip_tokenizer_init=False,
                                    max_model_len=config.prompt_length + config.response_length,
                                    max_num_batched_tokens=max_num_batched_tokens,
                                    enable_chunked_prefill=config.enable_chunked_prefill,
                                    load_format=config.load_format)
        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, max_retries: int = 1e9, **kwargs) -> DataProto:
        """Generate sequences using vLLM engine with retry logic for failures.

        Args:
            prompts (DataProto): Input prompts containing batch data with input_ids, attention_mask,
                position_ids and meta_info.
            max_retries (int, optional): Maximum number of retries on failure. Defaults to 1e9.
            **kwargs: Additional sampling parameters to override defaults.

        Returns:
            DataProto: Generated sequences containing:
                - prompts: Original input token ids
                - responses: Generated response token ids
                - input_ids: Concatenated prompt and response tokens
                - attention_mask: Attention mask for full sequence
                - position_ids: Position ids for full sequence

        Raises:
            RuntimeError: If generation fails after max_retries attempts.
        """
        max_retries = int(max_retries)
        for attempt in range(max_retries):
            try:
                # Rebuild vLLM cache engine if configured
                if self.config.free_cache_engine:
                    self.inference_engine.init_cache_engine()

                # Extract input tensors from prompt batch
                idx = prompts.batch['input_ids']
                attention_mask = prompts.batch['attention_mask']
                position_ids = prompts.batch['position_ids']
                eos_token_id = prompts.meta_info['eos_token_id']
                batch_size = idx.size(0)

                # Pre-process input token ids
                idx_list = [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ]

                # Configure sampling parameters
                do_sample = prompts.meta_info.get('do_sample', True)
                if not do_sample:
                    kwargs = {
                        'best_of': 1,
                        'top_p': 1.0,
                        'top_k': -1,
                        'min_p': 0.0,
                        'temperature': 0,
                        'n': 1
                    }
                if prompts.meta_info.get('val_temperature', None):
                    kwargs['temperature'] = prompts.meta_info['val_temperature']
                # Generate sequences
                with self.update_sampling_params(**kwargs):
                    print("-" * 20)
                    print(self.sampling_params)
                    max_func_call = 2
                    n = self.sampling_params.n
                    idx_list_with_id = []
                    i = 0
                    for idx_ in idx_list:
                        for _ in range(n):
                            idx_list_with_id.append((i, idx_, None))
                            i += 1
                    call_sampling_params = deepcopy(self.sampling_params)
                    call_sampling_params.n = 1
                    call_sampling_params.skip_special_tokens = False
                    #call_sampling_params.stop = ["<end_of_code>", "<end_of_answer>"]
                    call_sampling_params.stop_token_ids = [151667, 151671]
                    # call_sampling_params.logprobs = None
                    
                    end_outs = []
                    for epoch in range(max_func_call):
                        current_idxs = idx_list_with_id
                        if len(current_idxs) == 0:
                            break

                        prompt_ids = [item[1] for item in current_idxs]
                        output = self.inference_engine.generate(
                            prompts=None,
                            sampling_params=call_sampling_params,
                            prompt_token_ids=prompt_ids,
                            use_tqdm=False)
                        # print("这个output 到底是什么")
                        # print(output[0])
                        idx_list_with_id = []
                        for (i, idx_list, _), out in zip(current_idxs, output):
                            if out.outputs[0].stop_reason and out.outputs[0].stop_reason == 151667:
                                all_idx = idx_list + list(out.outputs[0].token_ids)
                                idx_list_with_id.append((i, all_idx, out))
                            else:
                                end_outs.append((i, [], out))
                        
                        remain_res = batch_apply([item[1] for item in idx_list_with_id], self.tokenizer)
                        #remain_res = []
                        idx_list_with_id_new = deepcopy(idx_list_with_id)
                        for k in range(len(idx_list_with_id)):
                            i, idx_, out = idx_list_with_id[k]
                            res = remain_res[k]
                            idx_ = idx_ + self.tokenizer.encode(res)
                            # if epoch == max_func_call - 1:
                            #     idx_ += self.tokenizer.encode("\nReach max function call limit.")
                            idx_list_with_id_new[k] = (i, idx_, out)
                        idx_list_with_id = idx_list_with_id_new
                        
                    end_outs.extend(idx_list_with_id)

                    def takeFirst(elem):
                        return elem[0]
                    end_outs.sort(key=takeFirst)


                    output_token_ids = []
                    for request_output in end_outs:  # List[RequestOutput]
                        prompt_token_ids = request_output[2].prompt_token_ids
                        res_token_ids = request_output[2].outputs[0].token_ids
                        prompt_end_text = "\n<|assistant|>: Let's think step by step and solve the problem with code."
                        # 使用 self.tokenizer 将提示文本转换为 token id 序列（不添加特殊 tokens）
                        prompt_end_ids = self.tokenizer.encode(prompt_end_text, add_special_tokens=False)

                        # 在 token_ids 列表中查找 prompt_end_ids 所在的位置
                        token_ids = list(prompt_token_ids) + list(res_token_ids)
                        start_index = 0
                        for i in range(len(token_ids) - len(prompt_end_ids) + 1):
                            # 如果从 i 开始的子列表与 prompt_end_ids 相同，则找到了提示结束的标识位置
                            if token_ids[i:i + len(prompt_end_ids)] == prompt_end_ids:
                                # 响应部分的起始位置为该位置后一个 token
                                start_index = i + len(prompt_end_ids)
                                break

                        # 截取响应部分的 token ids
                        resp_ids = token_ids[start_index:]
                        output_token_ids.append(torch.tensor(resp_ids))
                
                    
                    pad_token_id = (self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                    else self.tokenizer.eos_token_id)
                    
                    output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=pad_token_id)
    
                # Process outputs
                response = output_token_ids.to(idx.device)
                # log_probs = output[1].to(idx.device)

                # Pad sequences if needed
                if response.shape[1] < self.config.response_length:
                    response = pad_sequence_to_length(
                        response, self.config.response_length, self.pad_token_id)
                    # log_probs = pad_sequence_to_length(
                    #     log_probs, self.config.response_length, self.pad_token_id)

                # Handle multiple samples per prompt
                if self.config.n > 1 and do_sample:
                    idx = idx.repeat_interleave(self.config.n, dim=0)
                    attention_mask = attention_mask.repeat_interleave(
                        self.config.n, dim=0)
                    position_ids = position_ids.repeat_interleave(
                        self.config.n, dim=0)
                    batch_size = batch_size * self.config.n

                # Concatenate prompt and response
                seq = torch.cat([idx, response], dim=-1)

                # Create position IDs and attention mask for full sequence
                response_length = response.size(1)
                delta_position_id = torch.arange(
                    1, response_length + 1, device=position_ids.device)
                delta_position_id = delta_position_id.unsqueeze(0).repeat(
                    batch_size, 1)

                response_position_ids = position_ids[:, -1:] + delta_position_id
                position_ids = torch.cat([position_ids, response_position_ids],
                                       dim=-1)
                response_attention_mask = get_eos_mask(
                    response_id=response,
                    eos_token=eos_token_id,
                    dtype=attention_mask.dtype)
                attention_mask = torch.cat(
                    (attention_mask, response_attention_mask), dim=-1)

                # Construct output batch
                batch = TensorDict(
                    {
                        'prompts': idx,
                        'responses': response,
                        'input_ids': seq,
                        'attention_mask': attention_mask,
                        'position_ids': position_ids
                    },
                    batch_size=batch_size)

                # Free cache if configured
                if self.config.free_cache_engine:
                    self.inference_engine.free_cache_engine()

                return DataProto(batch=batch)

            except Exception as e:
                print(e)
                print("报错了")
                # traceback.print_exc()
                # print("Restarting vLLM due to error: ", e)
                # print("Retrying...")

                # # Clean up and restart engine
                # torch.cuda.empty_cache()
                # if hasattr(self.inference_engine, 'free_cache_engine'):
                #     self.inference_engine.free_cache_engine()
                # del self.inference_engine

                # # Reinitialize engine with same parameters
                # self.inference_engine = LLM(
                #     self.actor_module,
                #     tokenizer=self.tokenizer,
                #     model_hf_config=self.model_hf_config,
                #     tensor_parallel_size=self.tensor_parallel_size,
                #     dtype=self.config.dtype,
                #     enforce_eager=self.config.enforce_eager,
                #     gpu_memory_utilization=self.config.gpu_memory_utilization,
                #     skip_tokenizer_init=False,
                #     max_model_len=self.config.prompt_length +
                #     self.config.response_length,
                #     load_format=self.config.load_format)
                # print("vLLM is ready to roll!")

                # if attempt < max_retries - 1:
                #     continue

        raise RuntimeError(
            f"Failed to generate sequences after {max_retries} attempts")
