"""LLM processing and cost calculation."""
import logging
import time
import threading
from pathlib import Path
from typing import Callable, Any

from pattern import diff_example, rewrite_example
import sys
from .config import config, TOKENS_PER_CHAR_ESTIMATE, AVAILABLE_MODELS
from .fs import is_image_file, file_cache, get_display_path, get_mime_type, encode_image

logger = logging.getLogger(__name__)

class CancelledError(Exception): pass
class GenerationError(Exception): pass

OutputFunc = Callable[..., None]

def _calc_tiered_cost(token_count: int, pricing: dict, key_prefix: str) -> float:
    base_cost = (token_count / 1_000_000) * pricing.get(key_prefix, 0)
    over_key = f"{key_prefix}_over_200k"
    
    if over_key in pricing and token_count > 200_000:
        base_tokens = 200_000
        over_tokens = token_count - 200_000
        cost = (base_tokens / 1_000_000) * pricing[key_prefix]
        cost += (over_tokens / 1_000_000) * pricing[over_key]
        return cost
        
    return base_cost

def _get_model_pricing(model_name: str) -> dict | None:
    return AVAILABLE_MODELS.get(model_name)

def calculate_input_cost(token_count: int, model_name: str) -> tuple[float, str]:
    pricing = _get_model_pricing(model_name)
    if not pricing:
        return 0.0, "???"
    cost = _calc_tiered_cost(token_count, pricing, "input")
    return cost, f"${cost:.4f}"

def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> tuple[float, str]:
    pricing = _get_model_pricing(model_name)
    if not pricing:
        return 0.0, " | Cost: (unknown model pricing)"
    input_cost = _calc_tiered_cost(input_tokens, pricing, "input")
    output_cost = _calc_tiered_cost(output_tokens, pricing, "output")
    total_cost = input_cost + output_cost
    return total_cost, f" | Est. Cost: ${total_cost:.4f}"

# Backward compatibility for tests
_calculate_cost = calculate_cost

def build_file_contents(filenames: list[str]) -> str:
    parts = []
    cwd = Path.cwd()
    sorted_filenames = sorted(filenames, key=lambda f: str(Path(f).resolve()))

    for filename in sorted_filenames:
        if is_image_file(filename):
            continue
        try:
            p = Path(filename)
            if p.exists() and p.is_file():
                content = file_cache.get_or_read(str(p.resolve()))
                display_path = get_display_path(p, cwd)
                parts.append(f"\n--- START OF FILE: {display_path} ---\n{content}\n--- END OF FILE: {display_path} ---\n")
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
    return "".join(parts)

def build_system_message(filenames: list[str], ask_mode: bool = False, plan_mode: bool = False) -> str:
    sorted_names = sorted(filenames)
    msg = ""
    if plan_mode:
        msg = f"""You have been given the following files: {", ".join(sorted_names)}

You are in Planning Mode.
Create a detailed implementation plan for the user's request.
Break it down into a sequence of tractable sub-tasks.
For each task, provide a clear 'Title' and a specific 'Prompt' for a code-editing LLM.

Output strictly using this format for every step:
<<<<<<< PLAN
Title: <short title>
Prompt: <detailed instructions for the LLM>
>>>>>>> END
"""
    elif ask_mode:
        msg = f"""You have been given the following files: {", ".join(sorted_names)}

You are in ask mode - answer questions about the code without making changes.
Provide helpful explanations, analysis, and insights about the codebase.
"""
    else:
        msg = f"""You have been given the following files: {", ".join(sorted_names)}

Use the following diff format to specify changes to files, including the surrounding backticks.
If you are writing a new file, leave the original text blank.
Ensure the filename is on its own line, not nestled with the backticks.
The original text must match exactly with no differences -- This means no annotations of any kind. 

{diff_example}

Ensure you surround your diff with triple backticks on their own lines.
Include a brief human-readable overview of the changes you plan to make at the start, AND a recap of the changes you made at the end.
"""
    if config.allow_rewrite:
        msg += f"\nTo replace a file completely (or delete it), you may use:\n\n{rewrite_example}\n"

    if config.extra_system_prompt:
        msg += f"\n\n=== Custom Instructions ===\n{config.extra_system_prompt}"
    return msg

def _create_openai_client():
    from openai import OpenAI
    # Access module directly to get latest values
    core_cfg = sys.modules["core.config"]
    return OpenAI(base_url=core_cfg.API_BASE_URL, api_key=core_cfg.API_KEY)

def generate(
    in_filenames: list[str],
    prompt: str,
    preuploaded_files: list | None = None,
    output_func_override: OutputFunc | None = None,
    raw_stream_output_func: OutputFunc | None = None,
    conversation_history: list[dict] | None = None,
    ask_mode: bool = False,
    plan_mode: bool = False,
    cancel_event: threading.Event | None = None,
    on_stream_start: Callable[[], None] | None = None,
) -> str:
    output_func = output_func_override or print
    client = _create_openai_client()
    
    system_message = build_system_message(in_filenames, ask_mode=ask_mode, plan_mode=plan_mode)
    
    text_files = [f for f in in_filenames if not is_image_file(f)]
    image_files = [f for f in in_filenames if is_image_file(f)]
    file_contents_str = build_file_contents(text_files)

    header = "File Contents"
    egress_msg = f"{header}:\n{file_contents_str}"
    if image_files:
        egress_msg += f"\n\n(Plus {len(image_files)} image file(s) attached)"
    egress_msg += f"\n\nRequest:\n{prompt}"

    api_user_content: str | list[dict[str, Any]] = egress_msg
    if image_files:
        api_user_content = [{"type": "text", "text": egress_msg}]
        for img_path in image_files:
            try:
                if isinstance(api_user_content, list):
                    api_user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{get_mime_type(img_path)};base64,{encode_image(img_path)}"}
                    })
            except Exception as e:
                logger.error(f"Failed to encode image {img_path}: {e}")

    history_user_content: str | list[dict[str, Any]] = prompt
    if image_files:
        history_user_content = [{"type": "text", "text": prompt}]
        for img_path in image_files:
            try:
                if isinstance(history_user_content, list):
                    history_user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{get_mime_type(img_path)};base64,{encode_image(img_path)}"}
                    })
            except Exception: pass

    base_messages: list[dict[str, Any]] = [{"role": "system", "content": system_message}]
    if conversation_history:
        base_messages.extend(conversation_history)
    base_messages.append({"role": "user", "content": api_user_content})
    
    full_result = ""
    start_time = time.time()
    last_progress_time = start_time
    total_input_tokens = 0
    total_output_tokens = 0
    
    current_messages = list(base_messages)
    
    extra_body = {}
    if "claude" in config.model.lower() or "anthropic" in config.model.lower():
        extra_body["provider"] = {"order": ["Anthropic"]}

    soft_limit_chars = int((config.output_sharding_limit * TOKENS_PER_CHAR_ESTIMATE) * config.sharding_ratio)

    logger.info(f"Generating with model: {config.model}")

    if on_stream_start:
        on_stream_start()

    try:
        for shard_idx in range(config.max_shards):
            shard_response = ""
            shard_finish_reason = None
            shard_input_tokens = 0
            shard_output_tokens = 0

            stream = client.chat.completions.create(
                model=config.model, 
                messages=current_messages, 
                stream=True,
                max_tokens=config.output_sharding_limit,
                extra_body=extra_body or None
            )

            for chunk in stream:
                if cancel_event and cancel_event.is_set():
                    output_func("\nGeneration cancelled.\n")
                    logger.info("Generation cancelled by user")
                    raise CancelledError("Cancelled by user")

                if not chunk.choices: continue

                content = chunk.choices[0].delta.content
                if content:
                    shard_response += content
                    if raw_stream_output_func:
                        raw_stream_output_func(content, end="")
                    elif time.time() - last_progress_time > 1:
                        last_progress_time = time.time()
                        output_func(".", end="", flush=True)

                    if len(shard_response) > soft_limit_chars:
                        if shard_response.endswith("\n>>>>>>> REPLACE\n") or shard_response.endswith("\n```\n"):
                            shard_finish_reason = "length"
                            break
                
                if chunk.choices[0].finish_reason:
                    shard_finish_reason = chunk.choices[0].finish_reason
                
                if hasattr(chunk, 'usage') and chunk.usage:
                    shard_input_tokens = getattr(chunk.usage, 'prompt_tokens', shard_input_tokens)
                    shard_output_tokens = getattr(chunk.usage, 'completion_tokens', shard_output_tokens)
            
            full_result += shard_response
            
            if shard_input_tokens == 0:
                shard_input_tokens = len(str(current_messages)) // TOKENS_PER_CHAR_ESTIMATE
            if shard_output_tokens == 0:
                shard_output_tokens = len(shard_response) // TOKENS_PER_CHAR_ESTIMATE
            
            total_input_tokens += shard_input_tokens
            total_output_tokens += shard_output_tokens

            if shard_finish_reason == "length":
                output_func(f" [Shard {shard_idx + 1} limit. Continuing...]", end="", flush=True)
                logger.debug(f"Shard {shard_idx+1} limit reached. Continuing...")
                
                current_messages.append({"role": "assistant", "content": shard_response})
                current_messages.append({
                    "role": "user", 
                    "content": "Output limit reached. Please continue generating EXACTLY where you left off. Do not repeat the last sentence, just continue the stream of text."
                })
            else:
                break
        else:
            logger.error("Error: Maximum shard limit reached.")
            raise GenerationError(f"Generation exceeded max shards ({config.max_shards})")

    except Exception as e:
        if not isinstance(e, (CancelledError, GenerationError)):
            logger.exception(f"Error during stream processing: {e}")
        raise e

    if conversation_history is not None:
        conversation_history.append({"role": "user", "content": history_user_content})
        conversation_history.append({"role": "assistant", "content": full_result})

    elapsed_time = time.time() - start_time
    _, cost_str = calculate_cost(total_input_tokens, total_output_tokens, config.model)
    
    logger.info(f"Tokens: {total_input_tokens} in / {total_output_tokens} out{cost_str}")
    logger.info(f"Time: {elapsed_time:.2f}s")
    return full_result