"""
Multi-agent system trace evaluation module with semantic caching.
"""
import os
import json
import pickle
import hashlib
import asyncio
import datetime
import argparse
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import faiss
from openai import AsyncOpenAI


def _cache_text_from_trace(trace: str, max_chars: int = 5000) -> str:
    """Build cache text from trace, keeping head and tail if very long."""
    if len(trace) <= max_chars:
        return trace
    head = trace[:4000]
    tail = trace[-1000:]
    return head + "\n...\n" + tail


class SemanticCache:
    """Semantic cache for LLM evaluation results using FAISS and sentence transformers."""
    
    def __init__(
        self,
        cache_dir: str,
        threshold: float = 0.93,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        enabled: bool = True,
    ):
        self.cache_dir = cache_dir
        self.index_path = os.path.join(cache_dir, "faiss.index")
        self.meta_path = os.path.join(cache_dir, "meta.pkl")
        self.threshold = threshold
        self.enabled = enabled
        self.embedder = None
        self.index = None  # faiss index (cosine via normalized embeddings + inner product)
        self.items = []    # list of dicts: {"hash": str, "response": str, "text_len": int}
        self.by_hash = {}  # exact match map
        self._lock = asyncio.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)
        if self.enabled:
            self.embedder = SentenceTransformer(embedding_model_name)
            self._load_if_exists()

    def _load_if_exists(self):
        """Load existing cache if available."""
        if not self.enabled:
            return
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                meta = pickle.load(f)
            # backwards/forwards simple schema
            self.items = meta.get("items", [])
            self.threshold = meta.get("threshold", self.threshold)
            for i, it in enumerate(self.items):
                h = it.get("hash")
                if h:
                    self.by_hash[h] = i

    def save(self):
        """Persist cache to disk."""
        if (not self.enabled) or (self.index is None):
            return
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(
                {"items": self.items, "threshold": self.threshold},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate SHA-256 hash for exact matching."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text in a worker thread."""
        vec = await asyncio.to_thread(
            self.embedder.encode,
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        # encode may return 1D (good) or 2D (batch); ensure 1D
        if getattr(vec, "ndim", 1) > 1:
            vec = vec[0]
        return vec.astype("float32")

    async def get(self, text: str) -> Optional[str]:
        """Get cached result for text (exact match first, then semantic)."""
        if not self.enabled:
            return None
        key_hash = self._hash_text(text)
        async with self._lock:
            # Exact match first
            if key_hash in self.by_hash:
                idx = self.by_hash[key_hash]
                return self.items[idx]["response"]

            # Semantic match via FAISS
            if self.index is None or self.index.ntotal == 0:
                return None

        # Embed outside lock
        q = await self._embed(text)
        q = np.expand_dims(q, axis=0)

        async with self._lock:
            D, I = self.index.search(q, 1)  # top-1 neighbor
            score = float(D[0][0]) if I[0][0] != -1 else -1.0
            if score >= self.threshold:
                idx = int(I[0][0])
                return self.items[idx]["response"]
            return None

    async def put(self, text: str, response: str):
        """Store result in cache."""
        if not self.enabled:
            return
        key_hash = self._hash_text(text)
        vec = await self._embed(text)
        vec = np.expand_dims(vec, axis=0)

        async with self._lock:
            # Lazily initialize index when we know the dim
            if self.index is None:
                dim = vec.shape[1]
                self.index = faiss.IndexFlatIP(dim)  # cosine via normalized IP
            self.index.add(vec)
            self.items.append({"hash": key_hash, "response": response, "text_len": len(text)})
            self.by_hash[key_hash] = len(self.items) - 1


class ModelRouter(ABC):
    """Abstract interface for routing requests to different models."""
    
    @abstractmethod
    async def select_model(self, models: List[str], **kwargs) -> str:
        """
        Select a model from the available models list.
        
        Args:
            models: List of available model names
            **kwargs: Additional context for routing decisions
            
        Returns:
            Selected model name
        """
        pass


class RoundRobinRouter(ModelRouter):
    """Round-robin router that cycles through models sequentially."""
    
    def __init__(self):
        self._current_index = 0
        self._lock = asyncio.Lock()
    
    async def select_model(self, models: List[str], **kwargs) -> str:
        """Select the next model in round-robin fashion."""
        if not models:
            raise ValueError("Models list cannot be empty")
        
        async with self._lock:
            selected = models[self._current_index]
            self._current_index = (self._current_index + 1) % len(models)
            return selected


class LLMEvaluator:
    """LLM-based evaluator for multi-agent system traces."""
    
    def __init__(
        self,
        model: Union[str, List[str]] = "openai/gpt-oss-20b",
        base_url: str = "http://localhost:8000/v1",
        max_model_length: int = 130000,
        temperature: float = 0,
        api_key: str = "KEY",
        routing_logic: str = "roundrobin"
    ):
        # Handle single model or list of models
        if isinstance(model, str):
            self.models = model.strip().split(",")
        else:
            if not model or len(model) == 0:
                raise ValueError("At least one model must be provided")
            self.models = model
        
        self.base_url = base_url
        self.max_model_length = max_model_length
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=7200)
        
        # Set up router (default to round-robin if multiple models, None if single model)
        if routing_logic == "roundrobin":
            self.router = RoundRobinRouter()
        else:
            raise ValueError(f"Invalid routing logic: {routing_logic}")
        
        # Load definitions and examples
        self.definitions = self._load_definitions()
        self.examples = self._load_examples()
        
    def _load_definitions(self) -> str:
        """Load taxonomy definitions."""
        with open("MAST/taxonomy_definitions_examples/definitions.txt", "r") as f:
            return f.read()
    
    def _load_examples(self) -> str:
        """Load taxonomy examples."""
        with open("MAST/taxonomy_definitions_examples/examples.txt", "r") as f:
            return f.read()

    async def _select_model(self) -> str:
        """Select a model using the router if available, otherwise use the first model."""
        if self.router is not None:
            return await self.router.select_model(self.models)
        return self.models[0]
    
    async def chat_completion_request_openai(self, prompt: str) -> Optional[str]:
        """Make OpenAI API request."""
        messages = [{"role": "user", "content": prompt}]
        
        # Select model using router
        selected_model = await self._select_model()
        
        chat_response = await self.client.chat.completions.create(
            model=selected_model,
            temperature=self.temperature,
            messages=messages,
            timeout=7200,
            max_tokens=self.max_model_length,
        )
        
        if chat_response.choices:
            return chat_response.choices[0].message.content
        return None

    async def evaluate_trace(self, trace: str, examples: str = None) -> Optional[str]:
        """Evaluate a single trace."""
        if examples is None:
            examples = self.examples
            
        prompt = (
            "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
            "In the traces, analyze the system behaviour."
            "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
            "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
            "Also tell me whether the task is successfully completed or not, as a binary yes or no."
            "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
            "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
            "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
            "*** begin of things you should answer *** @@"
            "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
            "B. Whether the task is successfully completed or not: <yes or no>"
            "C. Whether you encounter any of the failure modes or inefficiencies:"
            "1.1 Disobey Task Specification: <yes or no>"
            "1.2 Disobey Role Specification: <yes or no>"
            "1.3 Step Repetition: <yes or no>"
            "1.4 Loss of Conversation History: <yes or no>"
            "1.5 Unaware of Termination Conditions: <yes or no>"
            "2.1 Conversation Reset: <yes or no>"
            "2.2 Fail to Ask for Clarification: <yes or no>"
            "2.3 Task Derailment: <yes or no>"
            "2.4 Information Withholding: <yes or no>"
            "2.5 Ignored Other Agent's Input: <yes or no>"
            "2.6 Action-Reasoning Mismatch: <yes or no>"
            "3.1 Premature Termination: <yes or no>"
            "3.2 No or Incorrect Verification: <yes or no>"
            "3.3 Weak Verification: <yes or no>"
            "@@*** end of your answer ***"
            "An example answer is: \n"
            "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
            "B. no \n"
            "C. \n"
            "1.1 no \n"
            "1.2 no \n"
            "1.3 no \n"
            "1.4 no \n"
            "1.5 no \n"
            "1.6 yes \n"
            "2.1 no \n"
            "2.2 no \n"
            "2.3 yes \n"
            "2.4 no \n"
            "2.5 no \n"
            "2.6 yes \n"
            "2.7 no \n"
            "3.1 no \n"
            "3.2 yes \n"
            "3.3 no \n"   
            "Here is the trace: \n"
            f"{trace}"
            "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
            f"{self.definitions} \n"
            "Here are some examples of the failure modes and inefficiencies: \n"
            f"{examples}"
        )
        return await self.chat_completion_request_openai(prompt)

    async def evaluate_traces(
        self, 
        full_trace_list: List[dict], 
        examples: str = None, 
        cache_dir: str = "saved_results",
        cache_enabled: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """
        Evaluate multiple traces with semantic caching.
        
        Returns:
            Tuple of (times_info_list, results)
        """
        if examples is None:
            examples = self.examples
            
        # Initialize semantic cache
        cache_path = os.path.join(cache_dir, "semantic_cache")
        cache = SemanticCache(cache_dir=cache_path, threshold=0.93, enabled=cache_enabled)
        
        # Use index-based assignment for results to avoid race conditions with append
        times = [None] * len(full_trace_list)
        results = [None] * len(full_trace_list)

        async def evaluate_trace(i):
            # Build the cache text (semantic key). We cache on the trace only.
            cache_text = _cache_text_from_trace(full_trace_list[i]["trace"]["trajectory"], max_chars=self.max_model_length)

            # Respect model input length (trim only what you send to model; cache uses head+tail)
            trace_to_evaluate = full_trace_list[i]["trace"]["trajectory"]
            system_prompt_length = len(examples + self.definitions) + 1000 # 1000 is a buffer
            if len(trace_to_evaluate) + system_prompt_length > self.max_model_length:
                trace_to_evaluate = trace_to_evaluate[:self.max_model_length - system_prompt_length]

            start_time = datetime.datetime.now()
            try:
                # Check cache first
                cached = await cache.get(cache_text)
                if cached is not None:
                    times[i] = {
                        'index': i,
                        'start': start_time,
                        'end': datetime.datetime.now(),
                        'duration_seconds': (datetime.datetime.now() - start_time).total_seconds(),
                        'error': None,
                        'cached': True,
                    }
                    results[i] = cached
                    print(f"[CACHE HIT] Evaluation {i+1}/{len(full_trace_list)} reused at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    return

                # Otherwise, call the model
                print(f"Evaluation {i+1}/{len(full_trace_list)} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                openai_evaluation = await self.evaluate_trace(trace_to_evaluate, examples=examples)

                times[i] = {
                    'index': i,
                    'start': start_time,
                    'end': datetime.datetime.now(),
                    'duration_seconds': (datetime.datetime.now() - start_time).total_seconds(),
                    'error': None,
                    'cached': False
                }
                results[i] = openai_evaluation

                # Populate cache
                if openai_evaluation is not None:
                    await cache.put(cache_text, openai_evaluation)

                print(f"Evaluation {i+1}/{len(full_trace_list)} completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            except Exception as e:
                error_time = datetime.datetime.now()
                times[i] = {
                    'index': i,
                    'start': start_time,
                    'end': error_time,
                    'duration_seconds': (error_time - start_time).total_seconds(),
                    'error': str(e),
                    'cached': False
                }
                results[i] = None
                print(f"Evaluation {i+1}/{len(full_trace_list)} failed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Error: {str(e)}")
                print(f"Trace length: {len(trace_to_evaluate)}")
                print(f"Examples length: {len(examples)}")

        tasks = [evaluate_trace(i) for i in range(len(full_trace_list))]
        await asyncio.gather(*tasks)

        # Persist the cache so future runs benefit immediately
        cache.save()

        return times, results


def parse_args():
    """Parse command-line arguments for running evaluation."""
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation on multi-agent system traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model name(s). Can be comma-separated for multiple models (e.g., 'model1,model2')"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the API endpoint"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="KEY",
        help="API key for authentication"
    )
    parser.add_argument(
        "--max-model-length",
        type=int,
        default=130000,
        help="Maximum model input length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model generation"
    )
    parser.add_argument(
        "--routing-logic",
        type=str,
        default="roundrobin",
        choices=["roundrobin"],
        help="Routing logic for multiple models"
    )
    
    # Cache configuration
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="saved_results",
        help="Directory for caching evaluation results"
    )
    parser.add_argument(
        "--cache-enabled",
        action="store_true",
        help="Enable semantic caching (default: False)"
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local dataset JSON file. If not provided, will download from HuggingFace."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="mcemri/MAD",
        help="HuggingFace repository ID for dataset (used if --dataset-path not provided)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="MAD_full_dataset.json",
        help="Dataset filename in the repository (used if --dataset-path not provided)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (defaults to cache-dir if not specified)"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point for command-line execution."""
    args = parse_args()
    
    # Load dataset
    if args.dataset_path:
        print(f"Loading dataset from {args.dataset_path}")
        with open(args.dataset_path, "r") as f:
            full_trace_list = json.load(f)
    else:
        print(f"Downloading dataset from HuggingFace: {args.repo_id}/{args.filename}")
        try:
            from huggingface_hub import hf_hub_download
            file_path = hf_hub_download(
                repo_id=args.repo_id, 
                filename=args.filename, 
                repo_type="dataset"
            )
            with open(file_path, "r") as f:
                full_trace_list = json.load(f)
        except ImportError:
            raise ImportError("huggingface_hub is required when using --repo-id. Install it with: pip install huggingface_hub")
    
    print(f"Loaded {len(full_trace_list)} records.")
    
    # Initialize evaluator
    evaluator = LLMEvaluator(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_model_length=args.max_model_length,
        temperature=args.temperature,
        routing_logic=args.routing_logic,
    )
    
    # Run evaluation
    times_info_list, results = await evaluator.evaluate_traces(
        full_trace_list,
        cache_dir=args.cache_dir,
        cache_enabled=args.cache_enabled,
    )
    
    # Save results
    output_dir = args.output_dir if args.output_dir else args.cache_dir
    os.makedirs(output_dir, exist_ok=True)
    
    model_clean = args.model.replace("/", "_").replace(",", "_")
    results_path = os.path.join(output_dir, f'{model_clean}_{args.max_model_length}_{args.routing_logic}_results.pkl')
    times_path = os.path.join(output_dir, f'{model_clean}_{args.max_model_length}_{args.routing_logic}_times_info.pkl')
    
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    with open(times_path, 'wb') as f:
        pickle.dump(times_info_list, f)
    
    print(f"\nResults saved to:")
    print(f"  - {results_path}")
    print(f"  - {times_path}")
    
    # Print summary
    successful = sum(1 for t in times_info_list if t and t.get("error") is None)
    cached = sum(1 for t in times_info_list if t and t.get("cached", False))
    print(f"\nSummary:")
    print(f"  - Total evaluations: {len(times_info_list)}")
    print(f"  - Successful: {successful}")
    print(f"  - Cached: {cached}")
    print(f"  - Failed: {len(times_info_list) - successful}")
    
    return times_info_list, results


if __name__ == "__main__":
    asyncio.run(main())
