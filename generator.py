import os
import random
import sys
import json
import argparse
import contextlib
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        if args is None or args.comfyui_directory is None:
            path = os.getcwd()
        else:
            path = args.comfyui_directory

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes(init_custom_nodes=True)


def save_image_wrapper(context, cls):
    if args.output is None:
        return cls

    from PIL import Image, ImageOps, ImageSequence
    from PIL.PngImagePlugin import PngInfo

    import numpy as np

    class WrappedSaveImage(cls):
        counter = 0

        def save_images(
            self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
        ):
            if args.output is None:
                return super().save_images(
                    images, filename_prefix, prompt, extra_pnginfo
                )
            else:
                if len(images) > 1 and args.output == "-":
                    raise ValueError("Cannot save multiple images to stdout")
                filename_prefix += self.prefix_append

                results = list()
                for batch_number, image in enumerate(images):
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                    metadata = None
                    if not args.disable_metadata:
                        metadata = PngInfo()
                        if prompt is not None:
                            metadata.add_text("prompt", json.dumps(prompt))
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    if args.output == "-":
                        # Hack to briefly restore stdout
                        if context is not None:
                            context.__exit__(None, None, None)
                        try:
                            img.save(
                                sys.stdout.buffer,
                                format="png",
                                pnginfo=metadata,
                                compress_level=self.compress_level,
                            )
                        finally:
                            if context is not None:
                                context.__enter__()
                    else:
                        subfolder = ""
                        if len(images) == 1:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = "output.png"
                            else:
                                subfolder, file = os.path.split(args.output)
                                if subfolder == "":
                                    subfolder = os.getcwd()
                        else:
                            if os.path.isdir(args.output):
                                subfolder = args.output
                                file = filename_prefix
                            else:
                                subfolder, file = os.path.split(args.output)

                            if subfolder == "":
                                subfolder = os.getcwd()

                            files = os.listdir(subfolder)
                            file_pattern = file
                            while True:
                                filename_with_batch_num = file_pattern.replace(
                                    "%batch_num%", str(batch_number)
                                )
                                file = (
                                    f"{filename_with_batch_num}_{self.counter:05}.png"
                                )
                                self.counter += 1

                                if file not in files:
                                    break

                        img.save(
                            os.path.join(subfolder, file),
                            pnginfo=metadata,
                            compress_level=self.compress_level,
                        )
                        print("Saved image to", os.path.join(subfolder, file))
                        results.append(
                            {
                                "filename": file,
                                "subfolder": subfolder,
                                "type": self.type,
                            }
                        )

                return {"ui": {"images": results}}

    return WrappedSaveImage


def parse_arg(s: Any):
    """Parses a JSON string, returning it unchanged if the parsing fails."""
    if __name__ == "__main__" or not isinstance(s, str):
        return s

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


parser = argparse.ArgumentParser(
    description="A converted ComfyUI workflow. Required inputs listed below. Values passed should be valid JSON (assumes string if not valid JSON)."
)
parser.add_argument(
    "--queue-size",
    "-q",
    type=int,
    default=1,
    help="How many times the workflow will be executed (default: 1)",
)

parser.add_argument(
    "--comfyui-directory",
    "-c",
    default=None,
    help="Where to look for ComfyUI (default: current directory)",
)

parser.add_argument(
    "--output",
    "-o",
    default=None,
    help="The location to save the output image. Either a file path, a directory, or - for stdout (default: the ComfyUI output directory)",
)

parser.add_argument(
    "--disable-metadata",
    action="store_true",
    help="Disables writing workflow metadata to the outputs",
)


comfy_args = [sys.argv[0]]
if __name__ == "__main__" and "--" in sys.argv:
    idx = sys.argv.index("--")
    comfy_args += sys.argv[idx + 1 :]
    sys.argv = sys.argv[:idx]

args = None
if __name__ == "__main__":
    args = parser.parse_args()
    sys.argv = comfy_args
if args is not None and args.output is not None and args.output == "-":
    ctx = contextlib.redirect_stdout(sys.stderr)
else:
    ctx = contextlib.nullcontext()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes(init_custom_nodes=True)


_custom_nodes_imported = False
_custom_path_added = False


def generate_video(image_path, audio_path, prompt, neg_prompt):
    global args, _custom_nodes_imported, _custom_path_added
    args = parser.parse_args()

    with ctx:
        if not _custom_path_added:
            add_comfyui_directory_to_sys_path()
            add_extra_model_paths()

            _custom_path_added = True

        if not _custom_nodes_imported:
            import_custom_nodes()

            _custom_nodes_imported = True

        from nodes import NODE_CLASS_MAPPINGS

    with torch.inference_mode(), ctx:
        multitalkmodelloader = NODE_CLASS_MAPPINGS["MultiTalkModelLoader"]()
        multitalkmodelloader_120 = multitalkmodelloader.loadmodel(
            model="multitalk.safetensors", base_precision="fp16"
        )

        loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
        loadaudio_125 = loadaudio.load(audio=audio_path)

        wanvideovaeloader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        wanvideovaeloader_129 = wanvideovaeloader.loadmodel(
            model_name="wan_2.1_vae.safetensors", precision="bf16"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_133 = loadimage.load_image(image=image_path)

        wanvideoblockswap = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
        wanvideoblockswap_134 = wanvideoblockswap.setargs(
            blocks_to_swap=10,
            offload_img_emb=False,
            offload_txt_emb=False,
            use_non_blocking=True,
            vace_blocks_to_swap=0,
        )

        loadwanvideot5textencoder = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
        loadwanvideot5textencoder_136 = loadwanvideot5textencoder.loadmodel(
            model_name="umt5-xxl-enc-bf16.safetensors",
            precision="bf16",
            load_device="offload_device",
            quantization="disabled",
        )

        downloadandloadwav2vecmodel = NODE_CLASS_MAPPINGS[
            "DownloadAndLoadWav2VecModel"
        ]()
        downloadandloadwav2vecmodel_137 = downloadandloadwav2vecmodel.loadmodel(
            model="TencentGameMate/chinese-wav2vec2-base",
            base_precision="fp16",
            load_device="main_device",
        )

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_173 = clipvisionloader.load_clip(
            clip_name="clip_vision_h.safetensors"
        )

        wanvideocontextoptions = NODE_CLASS_MAPPINGS["WanVideoContextOptions"]()
        wanvideocontextoptions_176 = wanvideocontextoptions.process(
            context_schedule="static_standard",
            context_frames=81,
            context_stride=4,
            context_overlap=32,
            freenoise=True,
            verbose=False,
        )

        wanvideotorchcompilesettings = NODE_CLASS_MAPPINGS[
            "WanVideoTorchCompileSettings"
        ]()
        wanvideotorchcompilesettings_177 = wanvideotorchcompilesettings.set_args(
            backend="inductor",
            fullgraph=False,
            mode="default",
            dynamic=False,
            dynamo_cache_size_limit=64,
            compile_transformer_blocks_only=True,
            dynamo_recompile_limit=128,
        )

        wanvideouni3c_controlnetloader = NODE_CLASS_MAPPINGS[
            "WanVideoUni3C_ControlnetLoader"
        ]()
        wanvideouni3c_controlnetloader_183 = wanvideouni3c_controlnetloader.loadmodel(
            model="Wan21_Uni3C_controlnet_fp16.safetensors",
            base_precision="fp16",
            quantization="fp8_e5m2",
            load_device="main_device",
            attention_mode="sageattn",
        )

        imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        imageresizekjv2_171 = imageresizekjv2.resize(
            width=400,
            height=720,
            upscale_method="lanczos",
            keep_proportion="crop",
            pad_color="0, 0, 0",
            crop_position="center",
            divisible_by=32,
            image=get_value_at_index(loadimage_133, 0),
        )

        vhs_duplicateimages = NODE_CLASS_MAPPINGS["VHS_DuplicateImages"]()
        vhs_duplicateimages_186 = vhs_duplicateimages.duplicate_input(
            multiply_by=81, images=get_value_at_index(imageresizekjv2_171, 0)
        )

        wanvideoencode = NODE_CLASS_MAPPINGS["WanVideoEncode"]()
        wanvideoencode_184 = wanvideoencode.encode(
            enable_vae_tiling=True,
            tile_x=272,
            tile_y=272,
            tile_stride_x=144,
            tile_stride_y=128,
            noise_aug_strength=0,
            latent_strength=1,
            vae=get_value_at_index(wanvideovaeloader_129, 0),
            image=get_value_at_index(vhs_duplicateimages_186, 0),
        )

        wanvideoloraselect = NODE_CLASS_MAPPINGS["WanVideoLoraSelect"]()
        wanvideoloraselect_203 = wanvideoloraselect.getlorapath(
            lora="wan/bounceV_01.safetensors",
            strength=1.1000000000000003,
            low_mem_load=False,
        )

        wanvideomodelloader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        audiocrop = NODE_CLASS_MAPPINGS["AudioCrop"]()
        multitalkwav2vecembeds = NODE_CLASS_MAPPINGS["MultiTalkWav2VecEmbeds"]()
        wanvideoclipvisionencode = NODE_CLASS_MAPPINGS["WanVideoClipVisionEncode"]()
        wanvideoimagetovideoencode = NODE_CLASS_MAPPINGS["WanVideoImageToVideoEncode"]()
        wanvideotextencode = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        wanvideouni3c_embeds = NODE_CLASS_MAPPINGS["WanVideoUni3C_embeds"]()
        wanvideosampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
        wanvideodecode = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        notif_playsound = NODE_CLASS_MAPPINGS["Notif-PlaySound"]()
        for q in range(args.queue_size):
            wanvideoloraselect_138 = wanvideoloraselect.getlorapath(
                lora="wan/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
                strength=1.2000000000000002,
                low_mem_load=False,
                prev_lora=get_value_at_index(wanvideoloraselect_203, 0),
            )

            wanvideomodelloader_122 = wanvideomodelloader.loadmodel(
                model="wan2.1_i2v_480p_14B_fp16.safetensors",
                base_precision="fp16",
                quantization="fp8_e5m2",
                load_device="offload_device",
                attention_mode="sageattn",
                compile_args=get_value_at_index(wanvideotorchcompilesettings_177, 0),
                block_swap_args=get_value_at_index(wanvideoblockswap_134, 0),
                lora=get_value_at_index(wanvideoloraselect_138, 0),
                multitalk_model=get_value_at_index(multitalkmodelloader_120, 0),
            )

            audiocrop_159 = audiocrop.main(
                start_time="0",
                end_time="0:11",
                audio=get_value_at_index(loadaudio_125, 0),
            )

            multitalkwav2vecembeds_123 = multitalkwav2vecembeds.process(
                normalize_loudness=True,
                num_frames=306,
                fps=25,
                audio_scale=1,
                audio_cfg_scale=3.3000000000000003,
                multi_audio_type="para",
                wav2vec_model=get_value_at_index(downloadandloadwav2vecmodel_137, 0),
                audio_1=get_value_at_index(audiocrop_159, 0),
            )

            wanvideoclipvisionencode_172 = wanvideoclipvisionencode.process(
                strength_1=1,
                strength_2=1,
                crop="center",
                combine_embeds="average",
                force_offload=True,
                tiles=0,
                ratio=0.5,
                clip_vision=get_value_at_index(clipvisionloader_173, 0),
                image_1=get_value_at_index(imageresizekjv2_171, 0),
            )

            wanvideoimagetovideoencode_132 = wanvideoimagetovideoencode.process(
                width=get_value_at_index(imageresizekjv2_171, 1),
                height=get_value_at_index(imageresizekjv2_171, 2),
                num_frames=305,
                noise_aug_strength=0,
                start_latent_strength=1,
                end_latent_strength=1,
                force_offload=True,
                fun_or_fl2v_model=False,
                tiled_vae=False,
                vae=get_value_at_index(wanvideovaeloader_129, 0),
                clip_embeds=get_value_at_index(wanvideoclipvisionencode_172, 0),
                start_image=get_value_at_index(imageresizekjv2_171, 0),
            )

            wanvideotextencode_135 = wanvideotextencode.process(
                positive_prompt=prompt,
                negative_prompt=neg_prompt,
                force_offload=True,
                t5=get_value_at_index(loadwanvideot5textencoder_136, 0),
            )

            wanvideouni3c_embeds_182 = wanvideouni3c_embeds.process(
                strength=1,
                start_percent=0,
                end_percent=0.10000000000000002,
                controlnet=get_value_at_index(wanvideouni3c_controlnetloader_183, 0),
                render_latent=get_value_at_index(wanvideoencode_184, 0),
            )

            wanvideosampler_128 = wanvideosampler.process(
                steps=5,
                cfg=1.0000000000000002,
                shift=5.000000000000001,
                seed=random.randint(1, 2**64),
                force_offload=True,
                scheduler="dpm++_sde",
                riflex_freq_index=0,
                denoise_strength=1,
                batched_cfg=False,
                rope_function="comfy",
                model=get_value_at_index(wanvideomodelloader_122, 0),
                image_embeds=get_value_at_index(wanvideoimagetovideoencode_132, 0),
                text_embeds=get_value_at_index(wanvideotextencode_135, 0),
                context_options=get_value_at_index(wanvideocontextoptions_176, 0),
                uni3c_embeds=get_value_at_index(wanvideouni3c_embeds_182, 0),
                multitalk_embeds=get_value_at_index(multitalkwav2vecembeds_123, 0),
            )

            wanvideodecode_130 = wanvideodecode.decode(
                enable_vae_tiling=False,
                tile_x=272,
                tile_y=272,
                tile_stride_x=144,
                tile_stride_y=128,
                normalization="default",
                vae=get_value_at_index(wanvideovaeloader_129, 0),
                samples=get_value_at_index(wanvideosampler_128, 0),
            )

            vhs_videocombine_131 = vhs_videocombine.combine_video(
                frame_rate=25,
                loop_count=0,
                filename_prefix="multitalk",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(wanvideodecode_130, 0),
                audio=get_value_at_index(audiocrop_159, 0),
            )

            notif_playsound_188 = notif_playsound.nop(
                mode="always",
                volume=0.5,
                file="notify.mp3",
                any=get_value_at_index(vhs_videocombine_131, 0),
            )
