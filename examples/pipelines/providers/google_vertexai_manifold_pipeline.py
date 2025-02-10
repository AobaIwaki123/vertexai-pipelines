"""
title: Google GenAI (Vertex AI) Manifold Pipeline
author: Hiromasa Kakehashi
date: 2024-09-19
version: 1.0
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: vertexai
environment_variables: GOOGLE_PROJECT_ID, GOOGLE_CLOUD_REGION
usage_instructions:
  To use Gemini with the Vertex AI API, a service account with the appropriate role (e.g., `roles/aiplatform.user`) is required.
  - For deployment on Google Cloud: Associate the service account with the deployment.
  - For use outside of Google Cloud: Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of the service account key file.
"""

import os
from typing import Iterator, List, Union

import numpy as np
import vertexai
from google.cloud import bigquery
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
                laws_context = "以下は関連法令の詳細情報です\n取得した法令情報をもとにユーザーの質問に回答をしてください:\n"
                for law in retrieved_laws:
                    laws_context += f"【{law['law_name']}】\n{law['law_content']}\n\n"

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

            # print(f"Response: {response}")

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

    # ※ここでは、ユーザーの質問から埋め込みを得る関数 compute_embedding() を仮定します。

    def compute_embedding(self, text: str) -> np.ndarray:
        # ここにVertex AIや他の埋め込みモデルを用いた処理を実装してください
        # 例: embedding = embedding_model.embed(text)
        pass

    def retrieve_relevant_laws(self, query: str, k: int = 3) -> list:
        # 1. ユーザーの質問からベクトルを計算
#         query_embedding = self.compute_embedding(query)  # numpy array 等
# 
#         # 2. BigQueryクライアントの初期化（環境変数で設定済みのプロジェクトIDを使用）
#         client = bigquery.Client(project=self.valves.GOOGLE_PROJECT_ID)
# 
#         # 3. ベクトル検索のクエリ（実際のSQLはBigQueryでの実装方法に合わせて調整してください）
#         query_string = """
#         WITH LawData AS (
#         SELECT
#             law_name,
#             law_content,
#             embedding
#         FROM `your_project.your_dataset.laws_detailed`
#         )
#         SELECT
#         law_name,
#         law_content,
#         ML.PREDICT_VECTOR_SIMILARITY(embedding, @query_embedding) AS similarity
#         FROM LawData
#         WHERE ML.PREDICT_VECTOR_SIMILARITY(embedding, @query_embedding) > 0.8
#         ORDER BY similarity DESC
#         LIMIT @k
#         """
# 
#         job_config = bigquery.QueryJobConfig(
#             query_parameters=[
#                 bigquery.ArrayQueryParameter(
#                     "query_embedding", "FLOAT64", query_embedding.tolist()
#                 ),
#                 bigquery.ScalarQueryParameter("k", "INT64", k),
#             ]
#         )
# 
#         query_job = client.query(query_string, job_config=job_config)
#         results = query_job.result()
        # return [dict(row) for row in results]
        # Return　Mock Data
        return [
            {
                "law_name": "個人情報保護法",
                "law_content": "個人情報保護法は、個人情報の適切な取り扱いを義務付ける法律です。",
            },
            {
                "law_name": "著作権法",
                "law_content": "著作権法は、著作物の権利を保護する法律です。",
            },
            {
                "law_name": "労働基準法",
                "law_content": "労働基準法は、労働者の権利を保護する法律です。",
            },
        ]
