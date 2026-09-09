"""Core SOPM pipeline: embeddings, SOM training/caching, recommendation and LLM.

This module holds the pure (non-Streamlit) logic shared by the interactive app
(``sopmInterface.py``) and the comparison benchmark (``runner.py``). Keeping it
free of ``import streamlit`` lets the benchmark run headless.
"""

from __future__ import annotations

import os
import pickle
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

# Workaround for some Streamlit/protobuf version combinations on Windows.
# If you prefer the faster C++ implementation, pin `protobuf<=3.20.*` instead.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import requests
from minisom import MiniSom
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from embeddings_db import EmbeddingRecord, EmbeddingStore, make_embedding_key

# Compatibility shims for NumPy 2.x with older libs (e.g., Streamlit 0.82)
# that still reference deprecated aliases like np.object.
with warnings.catch_warnings():
	warnings.simplefilter("ignore", FutureWarning)
	if not hasattr(np, "object"):
		np.object = object  # type: ignore[attr-defined]
	if not hasattr(np, "bool"):
		np.bool = bool  # type: ignore[attr-defined]
	if not hasattr(np, "int"):
		np.int = int  # type: ignore[attr-defined]
	if not hasattr(np, "float"):
		np.float = float  # type: ignore[attr-defined]
	if not hasattr(np, "str"):
		np.str = str  # type: ignore[attr-defined]


DATA_CSV_DEFAULT = "ag_news_prompts.csv"
ARTIFACT_DIR_DEFAULT = "artifacts"
EMBEDDINGS_DB_FILENAME = "embeddings.sqlite"


@dataclass(frozen=True)
class SomParams:
	som_x: int = 5
	som_y: int = 10
	learning_rate: float = 0.1
	topology: str = "rectangular"
	num_iterations: int = 10_000
	sigma: float = 2.0
	random_seed: int = 42
	neighborhood_function: str = "gaussian"
	# Embeddings are L2-normalized, so cosine keeps the SOM's notion of
	# similarity consistent with KNNRetriever's (dot product on unit vectors).
	# Euclidean would drift from that once neuron weights stop being unit-norm
	# during training (weight updates are convex combinations of inputs).
	activation_distance: str = "cosine"


@dataclass
class SomBundle:
	df: pd.DataFrame
	embs: np.ndarray
	som: MiniSom
	params: SomParams
	embed_model_name: str


def _safe_mkdir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(x, axis=1, keepdims=True)
	norms = np.where(norms == 0, 1.0, norms)
	return x / norms


@lru_cache(maxsize=4)
def _load_embedding_model(model_name: str) -> SentenceTransformer:
	# Jina v3+/v5 ship custom code (trust_remote_code) and route through a task
	# adapter chosen at load time; we use the retrieval adapter for the benchmark.
	if _is_asymmetric_model(model_name):
		return SentenceTransformer(
			model_name,
			trust_remote_code=True,
			model_kwargs={"default_task": "retrieval"},
		)
	return SentenceTransformer(model_name)


def _is_asymmetric_model(model_name: str) -> bool:
	"""Models (e.g. Jina retrieval) that embed a query and a document differently
	via asymmetric prompts, so query/document roles must be tracked separately.
	"""
	return "jina" in model_name.lower()


def _embedding_cache_model_key(model_name: str, role: Optional[str]) -> str:
	"""Cache-key namespace for a (model, role) pair.

	Asymmetric models embed the same text differently as query vs document, so
	the role is folded into the key. Symmetric models (MiniLM) ignore the role,
	keeping previously cached embeddings valid.
	"""
	if role and _is_asymmetric_model(model_name):
		return f"{model_name}::{role}"
	return model_name


def _model_prompt_name(model: SentenceTransformer, role: Optional[str]) -> Optional[str]:
	"""Return the role as a prompt name only if the model defines that prompt
	(e.g. Jina's query/document); otherwise None so symmetric models are untouched.
	"""
	if not role:
		return None
	prompts = getattr(model, "prompts", None) or {}
	return role if role in prompts else None


@lru_cache(maxsize=4)
def _load_ollama_client(base_url: str) -> OpenAI:
	return OpenAI(api_key="ollama", base_url=base_url)


@lru_cache(maxsize=4)
def _load_df(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	required = {"id", "prompt_name", "input", "target"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"CSV missing required columns: {sorted(missing)}")
	return df


def _encode_texts(
	model: SentenceTransformer, texts: list[str], prompt_role: Optional[str] = None
) -> np.ndarray:
	pname = _model_prompt_name(model, prompt_role)
	if pname:
		embs = model.encode(texts, batch_size=32, show_progress_bar=False, prompt_name=pname)
	else:
		embs = model.encode(texts, batch_size=32, show_progress_bar=False)
	embs = np.asarray(embs, dtype=np.float32)
	return _normalize_rows(embs)


def _embeddings_db_path(artifact_dir: str) -> str:
	return os.path.join(artifact_dir, EMBEDDINGS_DB_FILENAME)


def _encode_texts_cached(
	*,
	model: Optional[SentenceTransformer] = None,
	model_name: str,
	texts: list[str],
	db_path: str,
	role: Optional[str] = None,
	dataset_tag: Optional[str] = None,
	prompt_ids: Optional[list[int]] = None,
) -> np.ndarray:
	"""Encode texts, reusing/storing embeddings in a SQLite DB.

	``role`` ("query"/"document") is tracked for asymmetric models (Jina), which
	embed the same text differently per role. ``model`` is loaded lazily: when
	every text is already cached, the (possibly heavy) model is never loaded.
	"""
	store = EmbeddingStore(db_path)
	key_model = _embedding_cache_model_key(model_name, role)
	keys = [make_embedding_key(key_model, t) for t in texts]
	found = store.get_many(keys)

	missing_idx = [i for i, k in enumerate(keys) if k not in found]
	computed_by_key: dict[str, np.ndarray] = {}
	if missing_idx:
		if model is None:
			model = _load_embedding_model(model_name)
		missing_texts = [texts[i] for i in missing_idx]
		missing_embs = _encode_texts(model, missing_texts, prompt_role=role)
		records: list[EmbeddingRecord] = []
		for local_i, i in enumerate(missing_idx):
			key = keys[i]
			emb = np.asarray(missing_embs[local_i], dtype=np.float32)
			computed_by_key[key] = emb
			records.append(
				EmbeddingRecord(
					key=key,
					model_name=key_model,
					text=texts[i],
					dim=int(emb.shape[0]),
					normalized=True,
					embedding=emb,
					dataset_tag=dataset_tag,
					prompt_id=(prompt_ids[i] if prompt_ids is not None else None),
				)
			)
		store.upsert_many(records)

	out = []
	for k in keys:
		emb = found.get(k)
		if emb is None:
			emb = computed_by_key[k]
		out.append(np.asarray(emb, dtype=np.float32))

	return _normalize_rows(np.vstack(out))


def _backfill_embeddings_db_from_bundle(*, bundle: SomBundle, csv_path: str, db_path: str) -> None:
	"""Best-effort: populate the embeddings DB using an already loaded bundle.

	This avoids recomputing embeddings when a SOM artifact exists but the DB is empty.
	"""
	try:
		texts = bundle.df["input"].astype(str).tolist()
		if not texts:
			return
		store = EmbeddingStore(db_path)
		# Corpus rows are documents; key them under the document role so asymmetric
		# models (Jina) stay consistent with _encode_texts_cached.
		key_model = _embedding_cache_model_key(bundle.embed_model_name, "document")
		key0 = make_embedding_key(key_model, texts[0])
		if key0 in store.get_many([key0]):
			return

		dataset_tag = os.path.basename(csv_path)
		prompt_ids = bundle.df["id"].astype(int).tolist() if "id" in bundle.df.columns else None
		records: list[EmbeddingRecord] = []
		for i, text in enumerate(texts):
			emb = np.asarray(bundle.embs[i], dtype=np.float32)
			records.append(
				EmbeddingRecord(
					key=make_embedding_key(key_model, text),
					model_name=key_model,
					text=text,
					dim=int(emb.shape[0]),
					normalized=True,
					embedding=emb,
					dataset_tag=dataset_tag,
					prompt_id=(prompt_ids[i] if prompt_ids is not None else None),
				)
			)
			if len(records) >= 500:
				store.upsert_many(records)
				records.clear()
		if records:
			store.upsert_many(records)
	except Exception:
		# Non-fatal: DB backfill is an optimization.
		return


def _som_artifact_path(
	artifact_dir: str,
	embed_model_name: str,
	params: SomParams,
	dataset_tag: Optional[str] = None,
) -> str:
	safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", embed_model_name)
	# Include dataset tag and random_seed so bundles for different datasets/seeds
	# never collide onto the same artifact file (required by the benchmark).
	safe_tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", dataset_tag) if dataset_tag else "default"
	# activation_distance is part of the key too: it changes what the trained
	# weights mean, so a cosine run must never load an old euclidean artifact.
	return os.path.join(
		artifact_dir,
		(
			f"som_{safe_tag}_{safe_model}_{params.som_x}x{params.som_y}"
			f"_it{params.num_iterations}_sig{params.sigma}_lr{params.learning_rate}"
			f"_seed{params.random_seed}_{params.activation_distance}.pkl"
		),
	)


def _save_bundle(path: str, bundle: SomBundle) -> None:
	payload = {
		"embed_model_name": bundle.embed_model_name,
		"params": {
			"som_x": bundle.params.som_x,
			"som_y": bundle.params.som_y,
			"learning_rate": bundle.params.learning_rate,
			"topology": bundle.params.topology,
			"num_iterations": bundle.params.num_iterations,
			"sigma": bundle.params.sigma,
			"random_seed": bundle.params.random_seed,
			"neighborhood_function": bundle.params.neighborhood_function,
			"activation_distance": bundle.params.activation_distance,
		},
		"df": bundle.df,
		"embs": bundle.embs,
		"som_weights": bundle.som.get_weights(),
	}
	with open(path, "wb") as f:
		pickle.dump(payload, f)


def _load_bundle(path: str) -> SomBundle:
	with open(path, "rb") as f:
		payload = pickle.load(f)

	raw_params = payload["params"]
	if isinstance(raw_params, dict):
		params = SomParams(**raw_params)
	else:
		# Backward-compatibility if an older artifact stored the dataclass.
		params = raw_params
	df: pd.DataFrame = payload["df"]
	embs: np.ndarray = payload["embs"]
	som_weights: np.ndarray = payload["som_weights"]
	embed_model_name: str = payload["embed_model_name"]

	som = MiniSom(
		params.som_x,
		params.som_y,
		int(embs.shape[1]),
		sigma=float(params.sigma),
		learning_rate=float(params.learning_rate),
		neighborhood_function=params.neighborhood_function,
		topology=params.topology,
		random_seed=int(params.random_seed),
		activation_distance=params.activation_distance,
	)
	som._weights = som_weights  # MiniSom doesn't expose a public setter

	return SomBundle(df=df, embs=embs, som=som, params=params, embed_model_name=embed_model_name)


def build_or_load_som_bundle(
	*,
	csv_path: str,
	artifact_dir: str,
	embed_model_name: str,
	params: SomParams,
) -> SomBundle:
	_safe_mkdir(artifact_dir)
	dataset_tag = os.path.basename(csv_path)
	artifact_path = _som_artifact_path(artifact_dir, embed_model_name, params, dataset_tag)
	db_path = _embeddings_db_path(artifact_dir)

	if os.path.exists(artifact_path):
		try:
			bundle = _load_bundle(artifact_path)
			_backfill_embeddings_db_from_bundle(bundle=bundle, csv_path=csv_path, db_path=db_path)
			return bundle
		except Exception:
			# If artifact is corrupted or incompatible with current code, rebuild.
			try:
				os.remove(artifact_path)
			except OSError:
				pass

	df = _load_df(csv_path).reset_index(drop=True)
	texts = df["input"].astype(str).tolist()
	prompt_ids = df["id"].astype(int).tolist()
	# The corpus rows are the "documents". model is loaded lazily inside — if all
	# embeddings are already cached, the (heavy) model is never loaded here.
	embs = _encode_texts_cached(
		model_name=embed_model_name,
		texts=texts,
		db_path=db_path,
		role="document",
		dataset_tag=dataset_tag,
		prompt_ids=prompt_ids,
	)

	som = MiniSom(
		params.som_x,
		params.som_y,
		embs.shape[1],
		sigma=params.sigma,
		learning_rate=params.learning_rate,
		neighborhood_function=params.neighborhood_function,
		topology=params.topology,
		random_seed=params.random_seed,
		activation_distance=params.activation_distance,
	)
	som.random_weights_init(embs)
	som.train_random(embs, params.num_iterations)

	bmus = [som.winner(v) for v in embs]
	df["som_x"] = [b[0] for b in bmus]
	df["som_y"] = [b[1] for b in bmus]

	bundle = SomBundle(df=df, embs=embs, som=som, params=params, embed_model_name=embed_model_name)
	_save_bundle(artifact_path, bundle)
	return bundle


@lru_cache(maxsize=8)
def _get_bundle_cached(csv_path: str, artifact_dir: str, embed_model_name: str, params: SomParams) -> SomBundle:
	return build_or_load_som_bundle(
		csv_path=csv_path,
		artifact_dir=artifact_dir,
		embed_model_name=embed_model_name,
		params=params,
	)


def compute_som_quality(bundle: SomBundle) -> dict[str, float]:
	"""Standard, single-source-of-truth SOM quality metrics.

	QE = mean distance between each sample and its BMU (quantization error).
	TE = fraction of samples whose 1st and 2nd BMUs are not adjacent (topographic error).
	"""
	return {
		"QE": float(bundle.som.quantization_error(bundle.embs)),
		"TE": float(bundle.som.topographic_error(bundle.embs)),
	}


_POS_COLUMNS = ["id", "prompt_name", "input", "target", "similarity", "som_x", "som_y"]


def _select_positives_topk(
	*, df: pd.DataFrame, embs: np.ndarray, prompt_emb: np.ndarray,
	pos_idx: np.ndarray, k_pos: int,
) -> pd.DataFrame:
	"""Top-k by cosine within the BMU neighborhood window.

	This is the original SOPM behavior: with a wide window it degenerates into a
	(lossy) KNN, since it just ranks the window's members by similarity.
	"""
	if len(pos_idx) == 0 or k_pos <= 0:
		return pd.DataFrame()
	sims = np.dot(embs[pos_idx], prompt_emb)
	topk_local = np.argsort(sims)[-min(k_pos, len(pos_idx)) :][::-1]
	topk_idx = pos_idx[topk_local]
	positives = df.iloc[topk_idx].copy()
	positives["similarity"] = sims[topk_local]
	return positives[_POS_COLUMNS]


def _select_positives_diverse(
	*, df: pd.DataFrame, embs: np.ndarray, prompt_emb: np.ndarray,
	bmu_x: int, bmu_y: int, k_pos: int, radius_pos: int, som_x: int, som_y: int,
) -> pd.DataFrame:
	"""One representative per SOM cell, ranked by relevance (the diversity mode).

	For each occupied cell in a Chebyshev window around the BMU, keep only its
	single best (most similar) example, then take the top ``k_pos`` cell
	representatives by similarity. Every positive therefore comes from a distinct
	SOM cell -> demonstrations that are related (same region of the map) yet
	*spatially diverse*, which is exactly what a plain KNN cannot give (KNN tends
	to return near-duplicates from one dense pocket). The window starts at
	``radius_pos`` and expands only if fewer than ``k_pos`` occupied cells are
	found, so a small radius on a larger map stays robust.
	"""
	if k_pos <= 0 or len(df) == 0:
		return pd.DataFrame()

	sims_all = np.dot(embs, prompt_emb)
	sx = df["som_x"].values
	sy = df["som_y"].values

	radius = max(0, radius_pos)
	reps: dict[tuple, tuple] = {}
	while True:
		x0 = max(0, bmu_x - radius)
		x1 = min(som_x - 1, bmu_x + radius)
		y0 = max(0, bmu_y - radius)
		y1 = min(som_y - 1, bmu_y + radius)
		mask = (sx >= x0) & (sx <= x1) & (sy >= y0) & (sy <= y1)
		reps = {}
		for i in np.flatnonzero(mask):
			cell = (int(sx[i]), int(sy[i]))
			if cell not in reps or sims_all[i] > reps[cell][1]:
				reps[cell] = (int(i), float(sims_all[i]))
		covers_all = x0 == 0 and y0 == 0 and x1 == som_x - 1 and y1 == som_y - 1
		if len(reps) >= k_pos or covers_all:
			break
		radius += 1

	if not reps:
		return pd.DataFrame()

	top = sorted(reps.values(), key=lambda t: t[1], reverse=True)[:k_pos]
	sel_idx = np.array([t[0] for t in top], dtype=int)
	positives = df.iloc[sel_idx].copy()
	positives["similarity"] = [t[1] for t in top]
	return positives[_POS_COLUMNS]


def _select_positives_spread(
	*, df: pd.DataFrame, embs: np.ndarray, prompt_emb: np.ndarray,
	bmu_x: int, bmu_y: int, k_pos: int,
) -> pd.DataFrame:
	"""Guaranteed topological spread: BMU cell, then adjacent, then far cells.

	Draws each example from a band of Chebyshev distance to the BMU rather than
	from a similarity ranking — one from the BMU's own cell, one from the ring of
	immediate neighbours, and the rest from the most distant occupied cells. The
	spread is therefore a property of the map, not a side effect of the scores:
	with three examples this is the "1 near / 1 adjacent / 1 far" design.

	Every tie is broken by similarity (and, failing that, by row order), so the
	selection is deterministic — a randomly chosen far cell would add variance
	between seeds and make the comparison against KNN harder to read.

	Distinct from ``diverse``: that mode still ranks *all* candidates by
	similarity and merely forbids repeating a cell, so its picks stay clustered
	near the BMU. This one spends its budget across the map on purpose.
	"""
	if k_pos <= 0 or len(df) == 0:
		return pd.DataFrame()

	sims_all = np.dot(embs, prompt_emb)
	sx = df["som_x"].values
	sy = df["som_y"].values
	cheb = np.maximum(np.abs(sx - bmu_x), np.abs(sy - bmu_y))

	# Best example of each occupied cell, with that cell's distance to the BMU.
	best: dict[tuple, tuple] = {}
	for i in range(len(df)):
		cell = (int(sx[i]), int(sy[i]))
		if cell not in best or sims_all[i] > best[cell][1]:
			best[cell] = (int(i), float(sims_all[i]), int(cheb[i]))

	chosen: list[tuple] = []
	used: set[tuple] = set()

	def take(candidates: list[tuple]) -> bool:
		"""Take the most similar of ``candidates``; True when one was added."""
		pool = [(c, v) for c, v in candidates if c not in used]
		if not pool:
			return False
		cell, val = max(pool, key=lambda cv: (cv[1][1], -cv[1][0]))
		used.add(cell)
		chosen.append(val)
		return True

	items = list(best.items())
	# 1) the BMU's own cell (nearest available if the BMU cell itself is empty)
	if items:
		dmin = min(v[2] for _, v in items)
		take([(c, v) for c, v in items if v[2] == dmin])
	# 2) the ring of immediate neighbours, falling back outward if it is empty
	if len(chosen) < k_pos:
		for d in range(1, int(max((v[2] for _, v in items), default=0)) + 1):
			if take([(c, v) for c, v in items if v[2] == d]):
				break
	# 3) the remainder from the most distant occupied cells inward
	while len(chosen) < k_pos:
		remaining = [(c, v) for c, v in items if c not in used]
		if not remaining:
			break
		dmax = max(v[2] for _, v in remaining)
		if not take([(c, v) for c, v in remaining if v[2] == dmax]):
			break

	if not chosen:
		return pd.DataFrame()
	sel_idx = np.array([t[0] for t in chosen], dtype=int)
	positives = df.iloc[sel_idx].copy()
	positives["similarity"] = [t[1] for t in chosen]
	return positives[_POS_COLUMNS]


def _recommend_for_embedding(
	*,
	prompt_emb: np.ndarray,
	bundle: SomBundle,
	k_pos: int,
	k_neg: int,
	radius_pos: int,
	radius_neg: int,
	selection: str = "topk",
) -> tuple[Tuple[int, int], pd.DataFrame, pd.DataFrame]:
	"""Return (bmu, positives_df, negatives_df).

	``selection`` chooses how positives are drawn from the BMU neighborhood:
	  - "topk":    top-k by cosine within the window (original behavior).
	  - "diverse": one representative per occupied SOM cell (Via 1), which
	               exploits the map's topology instead of mimicking KNN.
	  - "spread":  one example per distance band — BMU cell, adjacent ring, then
	               the most distant occupied cells — for guaranteed topological
	               spread regardless of how the similarities happen to rank.
	"""
	df = bundle.df
	embs = bundle.embs
	som = bundle.som
	params = bundle.params

	bmu_x, bmu_y = som.winner(prompt_emb)

	if selection == "spread":
		positives = _select_positives_spread(
			df=df, embs=embs, prompt_emb=prompt_emb,
			bmu_x=bmu_x, bmu_y=bmu_y, k_pos=k_pos,
		)
	elif selection == "diverse":
		positives = _select_positives_diverse(
			df=df, embs=embs, prompt_emb=prompt_emb,
			bmu_x=bmu_x, bmu_y=bmu_y, k_pos=k_pos, radius_pos=radius_pos,
			som_x=params.som_x, som_y=params.som_y,
		)
	elif selection == "topk":
		x0 = max(0, bmu_x - radius_pos)
		x1 = min(params.som_x - 1, bmu_x + radius_pos)
		y0 = max(0, bmu_y - radius_pos)
		y1 = min(params.som_y - 1, bmu_y + radius_pos)
		mask_pos = (df["som_x"].between(x0, x1)) & (df["som_y"].between(y0, y1))
		pos_idx = np.flatnonzero(mask_pos.values)
		positives = _select_positives_topk(
			df=df, embs=embs, prompt_emb=prompt_emb, pos_idx=pos_idx, k_pos=k_pos,
		)
	else:
		raise ValueError(
			f"unknown selection {selection!r}; valid: 'topk', 'diverse', 'spread'"
		)

	manhattan = np.abs(df["som_x"].values - bmu_x) + np.abs(df["som_y"].values - bmu_y)
	neg_idx = np.flatnonzero(manhattan >= radius_neg)

	negatives = pd.DataFrame()
	if len(neg_idx) > 0 and k_neg > 0:
		sims = np.dot(embs[neg_idx], prompt_emb)
		bottomk_local = np.argsort(sims)[: min(k_neg, len(neg_idx))]
		bottomk_idx = neg_idx[bottomk_local]
		negatives = df.iloc[bottomk_idx].copy()
		negatives["similarity"] = sims[bottomk_local]
		negatives = negatives[["id", "prompt_name", "input", "target", "similarity", "som_x", "som_y"]]

	return (int(bmu_x), int(bmu_y)), positives, negatives


def _strip_think_blocks(text: str) -> str:
	text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
	return text.strip()


def _ollama_native_base(client: OpenAI) -> str:
	"""Derive Ollama's native base URL (e.g. http://host:11434) from the
	OpenAI-compat base_url (e.g. http://host:11434/v1/) configured on `client`.
	"""
	base = str(client.base_url).rstrip("/")
	if base.endswith("/v1"):
		base = base[: -len("/v1")]
	return base


def _ollama_chat_raw(
	*,
	client: OpenAI,
	model_name: str,
	messages: list[dict],
	max_tokens: int,
	temperature: float,
	think: bool = False,
) -> str:
	"""Call Ollama's native /api/chat endpoint instead of the OpenAI-compat
	/v1/chat/completions layer.

	Only the native endpoint honors `think=False` for hybrid-reasoning models
	like Qwen3 (verified against Ollama 0.15.2: the OpenAI-compat endpoint
	silently ignores a `think` field and still burns the whole `max_tokens`
	budget on the hidden reasoning trace, often returning empty content).
	Disabling it here cuts per-call latency drastically since we always strip
	<think> blocks anyway and never use them.
	"""
	url = _ollama_native_base(client) + "/api/chat"
	resp = requests.post(
		url,
		json={
			"model": model_name,
			"messages": messages,
			"think": think,
			"stream": False,
			"options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
		},
		timeout=300,
	)
	resp.raise_for_status()
	content = resp.json().get("message", {}).get("content", "") or ""
	return _strip_think_blocks(content)


def _normalize_for_compare(text: str) -> str:
	"""Normalize text for 'did it change?' comparisons."""
	return re.sub(r"\s+", " ", (text or "").strip())


# Meta-prompt template used by _create_personalized_prompt. Exposed as a module
# constant so the benchmark can log it verbatim (reproducibility requirement).
META_PROMPT_TEMPLATE = """You are a prompt engineering specialist. Your goal is to improve the original prompt by combining it with insights from the similar examples provided.

=== ORIGINAL PROMPT ===
{original_prompt}

=== REFERENCE EXAMPLES ===
{similar_prompts}
{contrasting_prompts}

=== TASK ===
Create an IMPROVED version of the original prompt that:
1. Maintains the same task/objective as the original
2. Incorporates successful patterns from similar examples
3. Is clearer and more specific
4. Uses consistent formatting

Return ONLY the improved prompt, without additional explanations."""


def _create_personalized_prompt(
	*,
	client: OpenAI,
	model_name: str,
	original_prompt: str,
	positives: pd.DataFrame,
	negatives: Optional[pd.DataFrame] = None,
) -> str:
	similar_prompts = ""
	if positives is not None and not positives.empty:
		similar_prompts += "\n--- SIMILAR PROMPTS (good examples to draw inspiration from) ---\n"
		for i, (_, row) in enumerate(positives.iterrows(), 1):
			similar_prompts += f"\nExample {i}:\n"
			similar_prompts += f"Input: {str(row['input'])[:300]}...\n"
			similar_prompts += f"Expected response: {row['target']}\n"

	contrasting_prompts = ""
	if negatives is not None and not negatives.empty:
		contrasting_prompts += "\n--- CONTRASTING PROMPTS (different examples to avoid confusion) ---\n"
		for i, (_, row) in enumerate(negatives.iterrows(), 1):
			contrasting_prompts += f"\nExample {i}:\n"
			contrasting_prompts += f"Input: {str(row['input'])[:300]}...\n"
			contrasting_prompts += f"Expected response: {row['target']}\n"

	meta_prompt = META_PROMPT_TEMPLATE.format(
		original_prompt=original_prompt,
		similar_prompts=similar_prompts,
		contrasting_prompts=contrasting_prompts,
	)

	def _call(meta: str, *, temperature: float) -> str:
		return _ollama_chat_raw(
			client=client,
			model_name=model_name,
			messages=[
				{
					"role": "system",
					"content": "You are an expert prompt engineer. Output only the improved prompt text.",
				},
				{"role": "user", "content": meta},
			],
			max_tokens=2000,
			temperature=float(temperature),
			think=False,
		)

	# 1) First attempt: standard improvement.
	content = _call(meta_prompt, temperature=0.5)

	# 2) Retry if model returned empty after stripping (common when it outputs only <think>).
	if not content:
		retry_prompt = (
			meta_prompt
			+ "\n\nIMPORTANT: Do not include <think> tags. Do not output an empty response. Output the improved prompt text only."
		)
		content = _call(retry_prompt, temperature=0.6)

	# 3) Retry if unchanged (force a rewrite with a structured template).
	if _normalize_for_compare(content) == _normalize_for_compare(original_prompt):
		force_rewrite = (
			meta_prompt
			+ "\n\nIMPORTANT: Rewrite the prompt so it is measurably different in wording and structure (do NOT return the same text), "
			  "while keeping the exact same objective. Use a clear template with short sections like: Context, Task, Output format, Constraints."
		)
		content = _call(force_rewrite, temperature=0.7)

	# Final fallback: if still empty, keep the original.
	if not content:
		return original_prompt
	return content


def _call_llm_chat(*, client: OpenAI, model_name: str, user_content: str, max_tokens: int = 800) -> str:
	user_content = user_content.strip()
	if not user_content:
		return ""

	return _ollama_chat_raw(
		client=client,
		model_name=model_name,
		messages=[{"role": "user", "content": user_content}],
		max_tokens=max_tokens,
		temperature=0.0,
		think=False,
	)
