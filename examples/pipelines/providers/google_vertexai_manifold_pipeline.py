"""
title: Google GenAI (Vertex AI) Manifold Pipeline
author: Hiromasa Kakehashi
date: 2024-09-19
version: 1.0
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: vertexai, langchain==0.3.3, langchain-community==0.3.2, langchain-google-vertexai==2.0.4, pydantic==2.7.4
environment_variables: GOOGLE_PROJECT_ID, GOOGLE_CLOUD_REGION
usage_instructions:
  To use Gemini with the Vertex AI API, a service account with the appropriate role (e.g., `roles/aiplatform.user`) is required.
  - For deployment on Google Cloud: Associate the service account with the deployment.
  - For use outside of Google Cloud: Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of the service account key file.
"""

import os
from typing import Iterator, List, Union

import vertexai
from langchain.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import BigQueryVectorSearch
from langchain_google_vertexai import VertexAIEmbeddings
from pydantic import BaseModel, Field
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)


class Pipeline:
    """Google GenAI pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_PROJECT_ID: str = ""
        GOOGLE_CLOUD_REGION: str = ""
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.name = "vertexai: "

        self.valves = self.Valves(
            **{
                "GOOGLE_PROJECT_ID": os.getenv("GOOGLE_PROJECT_ID", ""),
                "GOOGLE_CLOUD_REGION": os.getenv("GOOGLE_CLOUD_REGION", ""),
                "USE_PERMISSIVE_SAFETY": False,
            }
        )
        self.pipelines = [
            {"id": "gemini-1.5-flash-001", "name": "Gemini 1.5 Flash"},
            {"id": "gemini-1.5-pro-001", "name": "Gemini 1.5 Pro"},
        ]


    async def on_startup(self) -> None:
        """This function is called when the server is started."""

        print(f"on_startup:{__name__}")
        vertexai.init(
            project=self.valves.GOOGLE_PROJECT_ID,
            location=self.valves.GOOGLE_CLOUD_REGION,
        )

        embedding = VertexAIEmbeddings(
            model_name="textembedding-gecko-multilingual@latest",
            project=self.valves.GOOGLE_PROJECT_ID,
        )

        self.store = BigQueryVectorSearch(
            project_id=self.valves.GOOGLE_PROJECT_ID,
            dataset_name="law_db",
            table_name="personal_info",
            embedding=embedding,
            distance_strategy=DistanceStrategy.COSINE,
        )

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        vertexai.init(
            project=self.valves.GOOGLE_PROJECT_ID,
            location=self.valves.GOOGLE_CLOUD_REGION,
        )

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Iterator]:
        try:
            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            print(f"Pipe function called for model: {model_id}")
            print(f"Stream mode: {body.get('stream', False)}")
            print(f"Messages: {messages}")
            print(f"Body: {body}")

            # ユーザーの質問から関連する法令情報を取得
            retrieved_laws = self.retrieve_relevant_laws(user_message, k=3)
            print(f"Retrieved laws: {retrieved_laws}")
            if retrieved_laws:
                # 取得した法令情報をテキストに整形
                laws_context = "以下はユーザーの質問に関連する法令情報です:\n"
                laws_context += (
                    "ユーザーからの質問に対して、法令をもとに回答してください。\n"
                )
                laws_context += (
                    "この時、参照した法令番号と法令名を明記してください。\n\n"
                )
                for law in retrieved_laws:
                    laws_context += f"法令名: {law['metadata']['law_name']}\n"
                    laws_context += f"法令番号: {law['metadata']['law_number']}\n"
                    laws_context += f"法令内容: {law['content']}\n\n"

                # システムメッセージとして先頭に追加する
                messages.insert(0, {"role": "system", "content": laws_context})

            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            print(f"System message: {system_message}")

            model = GenerativeModel(
                model_name=model_id,
                system_instruction=system_message,
            )

            if body.get("title", False):  # If chat title generation is requested
                contents = [Content(role="user", parts=[Part.from_text(user_message)])]
            else:
                contents = self.build_conversation_history(messages)

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = body.get("safety_settings")

            print("Generating content...")

            response = model.generate_content(
                contents,
                stream=body.get("stream", False),
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            if body.get("stream", False):
                return self.stream_response(response)
            else:
                return response.text

        except Exception as e:
            print(f"Error generating content: {e}")
            return f"An error occurred: {str(e)}"

    def stream_response(self, response):
        for chunk in response:
            if chunk.text:
                print(f"Chunk: {chunk.text}")
                yield chunk.text

    def build_conversation_history(self, messages: List[dict]) -> List[Content]:
        contents = []

        for message in messages:
            if message["role"] == "system":
                continue

            parts = []

            if isinstance(message.get("content"), list):
                for content in message["content"]:
                    if content["type"] == "text":
                        parts.append(Part.from_text(content["text"]))
                    elif content["type"] == "image_url":
                        image_url = content["image_url"]["url"]
                        if image_url.startswith("data:image"):
                            image_data = image_url.split(",")[1]
                            parts.append(Part.from_image(image_data))
                        else:
                            parts.append(Part.from_uri(image_url))
            else:
                parts = [Part.from_text(message["content"])]

            role = "user" if message["role"] == "user" else "model"
            contents.append(Content(role=role, parts=parts))

        return contents

    def retrieve_relevant_laws(self, query: str, k: int = 3) -> list[dict]:
        # 1. ユーザーの質問に関連するドキュメントを取得
        # retrieved_docs = self.store.similarity_search(query, k)
        retrieved_docs_mock = [
            {
                "page_content": "法令の内容",
                "metadata": {
                    "law_name": "法令名",
                    "law_number": "法令番号",
                },
            }
        ]

        # 2. 取得したドキュメントの整形
        # Relevant Docs Format:
        ## {
        ##    "content": "TEXT",
        ##    "metadata": {
        ##        "law_name": "TEXT",
        ##        "law_number": "TEXT"
        ##    }
        ## }
        relevant_docs = []

        for doc in retrieved_docs_mock:
            response_doc = {
                "content": doc['page_content'],
                "metadata": doc['metadata'],
            }
            relevant_docs.append(response_doc)

        return relevant_docs
