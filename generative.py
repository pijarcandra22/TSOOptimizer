import os
import re
import json

import base64
import mimetypes

from funct.cloud_storage import upload_blob

import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.preview.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Part
)

import subprocess
import requests

from google.auth import default
from google.auth.transport.requests import Request

from dotenv import load_dotenv

load_dotenv()


safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
}

def byte_to_part(file_base64,mime_type):
    return Part.from_data(
        mime_type=mime_type,
        data=file_base64,
    )

def json_parse(text):
    text = text.split("```")[1]
    text = text.replace("json","")
    try:
        json.loads(text)
    except:
        start = text.find('{')
        end = text.rfind('}') + 1
        json_text = text[start:end]
        json_text = json_text.replace("`","")
        text = json_text
    return text
    # with open("llama-output.txt","w") as f:
    #     f.write(text)
    matches = re.findall(r".*?(\{.*?\})", text, re.DOTALL)
    
    return "".join(matches)

def multimodal_part(file_uri, mode="Tools"):
    if mode == "Tools":
        with open(file_uri, "rb") as file:
            file_base64 = file.read()

        # Use mimetypes to determine the MIME type based on the file extension
        mime_type, _ = mimetypes.guess_type(file_uri)
        
        if not mime_type:
            raise ValueError(f"Unsupported file type for {file_uri}")

        return byte_to_part(file_base64,mime_type)
    elif mode == "Cloud_Stourage":
        project_id = os.getenv("project_id")
        bucket_name = os.getenv("gcs_bucket")
        source_file_name = file_uri
        destination_folder = os.getenv("gcs_folder")

        destination_blob_name = f"{destination_folder}/{file_uri}"
        return upload_blob(project_id, bucket_name, source_file_name, destination_blob_name)

def gemini(
        prompts,
        response_schema,
        gen_model,
        location,
        project_id,
        temperature,
        top_p,
        top_k,
        candidate_count,
        max_output_tokens,
        stream = False
    ):

    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(
        gen_model,
    )

    return model.generate_content(
        prompts,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            candidate_count=candidate_count,
            max_output_tokens=max_output_tokens,
        ),
        safety_settings=safety_settings,
        stream=stream,
    )

def llama_multimodal(
        prompts,
        gen_model,
        location,
        project_id,
        temperature,
        top_p,
        top_k=10,
        n=1,
        stream=False,
    ):
    endpoint = "us-central1-aiplatform.googleapis.com"
    location = location or "us-central1"
    print(stream)
    
    request_data = {
        "model": f"meta/{gen_model}",
        "stream": stream,
        "max_tokens": 4096,  # Matching the curl request
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "n": n,
        "messages":prompts
    }

    with open('request.json', 'w') as f:
        json.dump(request_data, f, indent=4)

    credentials, project = default()
    credentials.refresh(Request())
    access_token = credentials.token

    # Set the headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    url = f"https://{endpoint}/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"

    with open('request.json', 'r') as f:
        response = requests.post(url, headers=headers, data=f)
    
    print(response)
    return response

def llama(
        prompts,
        gen_model,
        location,
        project_id,
        temperature,
        top_p,
        stream = False
    ):

    endpoint = "us-central1-aiplatform.googleapis.com"
    location = "us-central1"

    prompts = " ".join(prompts)

    request_data = {
        "model": f"meta/{gen_model}",
        "stream": stream,
        "max_tokens": 4096,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompts
                    }
                ]
            }
        ]
    }

    with open('request.json', 'w') as f:
        json.dump(request_data, f, indent=4)

    credentials, project = default()
    credentials.refresh(Request())
    access_token = credentials.token

    # Set the headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    url = f"https://{endpoint}/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions"
    with open('request.json', 'r') as f:
        response = requests.post(url, headers=headers, data=f)
        
    return response

def generate_json(
    prompt,
    response_schema,
    gen_model,
    location,
    project_id,
    temperature,
    top_p,
    top_k,
    candidate_count,
    max_output_tokens,
    stream = False
):

    prompts = []
    if isinstance(prompt, list):
        prompts = prompt
    else:
        prompts.append(prompt)

    responses = None

    if gen_model in [
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro-002",
        "gemini-1.5-flash-002"
    ]:
        responses = gemini(
            prompts,
            response_schema,
            gen_model,
            location,
            project_id,
            temperature,
            top_p,
            top_k,
            candidate_count,
            max_output_tokens,
            stream = stream
        )
    elif gen_model in [
        "llama-3.1-70b-instruct-maas"
    ]:
        responses = llama(
            prompts,
            gen_model,
            location,
            project_id,
            temperature,
            top_p,
            stream = stream
        )
    
    elif gen_model in [
        "llama-3.2-90b-vision-instruct-maas"
    ]:
        responses = llama_multimodal(
            prompts = prompts,
            gen_model = gen_model,
            location = location,
            project_id = project_id,
            temperature=0.7,
            top_p=top_p,
            top_k=10,
            n=1,
            stream=False,
        )

    return responses

