# SPDX-License-Identifier: Apache-2.0
"""Batch inference + minADE/ADE over the PAI-AV long-tail reasoning val set (Alpamayo 2 Super).

Mirrors ``eval_pai_av_val.py`` (1.5) / ``eval_pai_av_val_r1.py`` (R1) from the alpamayo1.5
repo: same event list (reasoning/ood_reasoning.parquet), same t0 convention, same 1.6s
history / 6.4s future window, same K=6 sampling, and the same metrics + output schema
(shared via ``eval_common``), so all three models' parquets parse uniformly with that
repo's ``parse_results.py`` / ``compute_metrics.py`` / ``score_reasoning_multigpu.py``.

Protocol difference to keep in mind when comparing: 1.5/R1 saw 4 cameras x 4 frames (16
images); Alpamayo 2's trajectory task profile is 6 cameras x 4 frames (24 images, camera
ids [0,1,2,3,5,6] -- the canonical 7-camera ring minus rear_tele). That is the model's
trained/validated input contract, so we use it rather than forcing the 1.5 camera set.

--no_coc: 1.5 injects ``coc_text=""`` and R1 needs a subclass hack; Alpamayo 2 supports it
cleanly by dropping "cot" from ``components_prompt`` when building the conversation (the
same mechanism as ``truckdrive_eval_val.py``; verified there that ``extra["cot"]`` comes
back empty rather than merely unreported). Caveat: the released expert was trained with
CoT tokens always present in the KV-cache before the trajectory, so CoT-off is outside its
validated conditioning distribution.

Usage
-----
    python examples/eval_pai_av_val_a2.py --out outputs/pai_av_val_a2.parquet
    python examples/eval_pai_av_val_a2.py --no_coc --out outputs/pai_av_val_a2_nococ.parquet
    python examples/eval_pai_av_val_a2.py --limit 4 --num_traj_samples 2   # smoke test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_common as ec  # noqa: E402

from alpamayo2_super.common.constants import PUBLIC_MODEL_ID  # noqa: E402


def _prepare_model_inputs(data, model_config, tokenizer, helper_mod, enable_cot: bool):
    """``helper.prepare_model_inputs`` with a switch to drop CoT from the prompt.

    ``helper.create_messages`` hardcodes ``components_prompt=["cot", "traj_future"]`` with
    no exposed switch, hence this local copy (same as ``truckdrive_eval_val.py``).
    """
    import torch

    from alpamayo2_super.chat_template.conversation import build_conversation

    components_prompt = ["cot", "traj_future"] if enable_cot else ["traj_future"]
    messages = build_conversation(
        data=data,
        num_tokens_per_history_traj=model_config.tokens_per_history_traj,
        num_tokens_per_future_traj=model_config.tokens_per_future_traj,
        components_order=["image", "traj_history", "prompt"],
        components_prompt=components_prompt,
        generation_mode=True,
        include_camera_ids=model_config.include_camera_ids,
        camera_ids=data["camera_indices"],
        include_frame_nums=model_config.frame_label == "frame_num",
    )
    if messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages = messages[:-1]
    processor = helper_mod.get_processor(tokenizer, model_config)
    has_assistant_content = messages[-1]["role"] == "assistant" and bool(messages[-1]["content"])
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not has_assistant_content,
        add_vision_id=False,
        continue_final_message=has_assistant_content,
    )
    images = data["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
    tokenized_data = dict(
        processor(
            text=text, images=images, videos=None, padding=False,
            return_tensors="pt", do_rescale=False,
        )
    )
    if tokenized_data["input_ids"].shape[0] != 1:
        raise ValueError("expected one sample at a time")
    return {
        "tokenized_data": tokenized_data,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ec.add_common_args(parser)  # shared: --split/--limit/--num_traj_samples/--out/shards/...
    parser.add_argument(
        "--model_id",
        default=os.environ.get("ALPAMAYO2_SUPER_MODEL_ID", PUBLIC_MODEL_ID),
        help="HF id or local checkpoint path for Alpamayo 2 Super.",
    )
    parser.add_argument(
        "--task_profile", default="trajectory",
        help="Input profile selected from the canonical 7-camera ring (trajectory = 6 cams).",
    )
    parser.add_argument("--diffusion_steps", type=int, default=10,
                        help="Expert diffusion inference steps.")
    parser.add_argument(
        "--sample_chunk_size", type=int, default=3,
        help="Max trajectory samples per forward pass; K is drawn in sequential chunks of "
             "this size and concatenated. Measured on 80GB H100: K=3 fits (~73GB peak), "
             "K>=4 OOMs, so the default keeps K=6 runs alive. Statistically equivalent to "
             "one K-sample call (the samples are i.i.d. given fixed conditioning).",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import physical_ai_av
    import torch

    from alpamayo2_super import helper
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    if not torch.cuda.is_available():
        raise RuntimeError("PAI-AV eval requires a CUDA GPU.")
    torch.cuda.manual_seed_all(args.seed)

    rows = ec.select_rows(args)

    print(f"Loading model {args.model_id} ...")
    model = Alpamayo2Super.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda:0"
    )

    def loader(clip_id, t0_us, avdi):
        # The loader returns the canonical 7-camera source ring; narrow it to the task's
        # trained profile (trajectory -> camera ids [0,1,2,3,5,6], 4 frames each).
        source = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
        return select_task_input(source, args.task_profile)

    chunk = max(1, min(args.sample_chunk_size, args.num_traj_samples))
    chunks = [
        min(chunk, args.num_traj_samples - i)
        for i in range(0, args.num_traj_samples, chunk)
    ]

    def predict(data: dict) -> dict:
        model_inputs = _prepare_model_inputs(
            data, model.config, model.tokenizer, helper, enable_cot=not args.no_coc
        )
        model_inputs = helper.to_device(model_inputs, "cuda")
        # Draw K in sequential chunks: peak memory is set by the largest single call, so
        # this is what makes K=6 feasible on an 80GB card (see --sample_chunk_size).
        xyz_parts, cot_parts = [], []
        for n in chunks:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, _, _, extra = model.sample_trajectories_from_data(
                    data=model_inputs,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    num_traj_samples=n,
                    max_generation_length=args.max_generation_length,
                    diffusion_kwargs={"inference_step": args.diffusion_steps},
                    return_extra=True,
                )
            xyz_parts.append(pred_xyz.detach().float().cpu())
            cot_parts.append(np.asarray(extra["cot"]))  # (b, n_sets, n)
            del pred_xyz, extra
            torch.cuda.empty_cache()
        # Concatenate along the sample axis -> exactly the (b, n_sets, K, ...) layout a
        # single K-sample call would have produced.
        merged_xyz = torch.cat(xyz_parts, dim=2)
        merged_extra = {"cot": np.concatenate(cot_parts, axis=2)}
        return ec.build_pred_record(merged_xyz, data["ego_future_xyz"].cpu(), merged_extra)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    ec.run_eval(rows, loader, predict, avdi, args)


if __name__ == "__main__":
    main()
