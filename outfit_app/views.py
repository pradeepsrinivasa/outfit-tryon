import torch
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from PIL import Image
import numpy as np
from diffusers import AutoPipelineForInpainting, AutoencoderKL
from diffusers.utils import load_image
from sentence_transformers import SentenceTransformer, util
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import tempfile
import os
import pandas as pd


def generate_upper_lower_masks(image_path):
    ontology_dict = {
        "upper-body clothing": "upper-body",
        "pant": "lower-body"
    }
    ontology = CaptionOntology(ontology_dict)
    base_model = GroundedSAM(ontology=ontology)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    detections = base_model.predict(image)

    upper_mask = np.zeros((height, width), dtype=np.uint8)
    lower_mask = np.zeros((height, width), dtype=np.uint8)
    prompts = list(ontology_dict.values())

    if hasattr(detections, "mask") and detections.mask is not None:
        for label, mask in zip(prompts, detections.mask):
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            if label == "upper-body":
                upper_mask[mask.astype(bool)] = 255
            elif label == "lower-body":
                lower_mask[mask.astype(bool)] = 255

    # Return masks as NumPy arrays for inpainting
    return upper_mask, lower_mask


def load_inpainting_pipeline():
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
        low_cpu_mem_usage=True
    )
    pipeline.set_ip_adapter_scale(1.0)
    return pipeline


def find_similar_image(df, user_prompt):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    combined_text = df['description'] + " " + df['occasion']
    desc_embeddings = model.encode(combined_text, convert_to_tensor=True)
    prompt_embedding = model.encode(user_prompt, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(prompt_embedding, desc_embeddings)[0]
    top_results = cosine_scores.topk(1)

    for score, idx in zip(top_results[0], top_results[1]):
        print(f"Match: {df.iloc[idx.item()]['description']} (Score: {score:.4f})")
        return df.iloc[idx.item()]['image_path']


class GenerateOutfitAPIView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        prompt = request.data.get('prompt')
        image_file = request.FILES.get('image')

        if not prompt or not image_file:
            return Response({'error': 'Missing prompt or image'}, status=status.HTTP_400_BAD_REQUEST)

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            temp_image.write(image_file.read())
            temp_image_path = temp_image.name

        # Generate masks
        upper_mask, lower_mask = generate_upper_lower_masks(temp_image_path)

        # Load inpainting pipeline
        pipeline = load_inpainting_pipeline()

        # Prepare image for inpainting
        input_image = load_image(temp_image_path).convert("RGB").resize((512, 512))
        mask_image = Image.fromarray(upper_mask).convert("L").resize((512, 512))
        
        # Read dataset for finding similar image
        df = pd.read_csv("dataset6.csv")
        ip_image_path = find_similar_image(df, prompt)
        ip_image = load_image(ip_image_path).convert("RGB").resize((512, 512))

        # Run inpainting
        result = pipeline(
            prompt="a man perfect body ,realistic skin, natural skin,remove shirt",
            negative_prompt="ugly, bad quality, bad anatomy",
            image=input_image,
            mask_image=mask_image,
            ip_adapter_image=ip_image,
            strength=0.99,
            guidance_scale=7.5,
            num_inference_steps=120,
        ).images[0]

        # Save and return result as a response
        result_path = os.path.join(tempfile.gettempdir(), "output.jpg")
        result.save(result_path)

        with open(result_path, "rb") as f:
            return Response(f.read(), content_type="image/jpeg")
