
from __future__ import annotations

import os
import time
import textwrap
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sopm_core import (
	ARTIFACT_DIR_DEFAULT,
	DATA_CSV_DEFAULT,
	SomBundle,
	SomParams,
	_call_llm_chat,
	_create_personalized_prompt,
	_embeddings_db_path,
	_encode_texts_cached,
	_get_bundle_cached,
	_load_ollama_client,
	_recommend_for_embedding,
	build_or_load_som_bundle,
)


def _wrap_hover(text: str, width: int = 55) -> str:
	"""Wrap long text into multiple HTML lines for Plotly hover."""
	return "<br>".join(textwrap.wrap(str(text), width=width))


def _build_som_figure(
	*,
	bundle: SomBundle,
	user_bmu: Optional[Tuple[int, int]] = None,
	positives: Optional[pd.DataFrame] = None,
	negatives: Optional[pd.DataFrame] = None,
	seed: int = 0,
) -> go.Figure:
	df = bundle.df
	params = bundle.params

	u = bundle.som.distance_map().T

	# Explicit coordinates for cell centers (helps Plotly + older Streamlit render consistently)
	heat_x = np.arange(params.som_x) + 0.5
	heat_y = np.arange(params.som_y) + 0.5

	rng = np.random.default_rng(seed)
	jitter_x = rng.uniform(-0.15, 0.15, len(df))
	jitter_y = rng.uniform(-0.15, 0.15, len(df))
	x = df["som_x"].to_numpy(dtype=float) + 0.5 + jitter_x
	y = df["som_y"].to_numpy(dtype=float) + 0.5 + jitter_y

	hover = (
		"<b>id=</b>" + df["id"].astype(str)
		+ "  <b>type=</b>" + df["prompt_name"].astype(str)
		+ "  <b>som=</b>(" + df["som_x"].astype(str) + "," + df["som_y"].astype(str) + ")"
		+ "<br><b>input:</b><br>" + df["input"].astype(str).str.slice(0, 200).apply(lambda t: _wrap_hover(t, 55))
		+ "<br><b>target=</b>" + df["target"].astype(str)
	)

	fig = go.Figure()
	fig.add_trace(
		go.Heatmap(
			z=u,
			x=heat_x,
			y=heat_y,
			colorscale="rdbu",
			opacity=0.95,
			showscale=True,
			colorbar={"title": "U-Matrix"},
		)
	)
	fig.add_trace(
		go.Scatter(
			x=x,
			y=y,
			mode="markers",
			marker={
				"size": 9,
				"opacity": 0.85,
				# Use a neutral color to avoid confusion with the RdBu heatmap (red/blue).
				"color": "rgba(200, 200, 200, 0.90)",
				"line": {"width": 1, "color": "rgba(0,0,0,0.6)"},
			},
			hovertext=hover,
			hoverinfo="text",
			name="Prompts",
		)
	)

	if positives is not None and not positives.empty:
		fig.add_trace(
			go.Scatter(
				x=positives["som_x"].to_numpy(dtype=float) + 0.5,
				y=positives["som_y"].to_numpy(dtype=float) + 0.5,
				mode="markers",
				# Green is distinct from RdBu and reads as "positive".
				# Note: for *-open symbols Plotly typically uses `marker.color` for the outline.
				marker={
					"size": 14,
					"symbol": "circle-open",
					"color": "rgba(0, 200, 0, 1.0)",
					"line": {"width": 3, "color": "rgba(0, 200, 0, 1.0)"},
				},
				name="Similar",
				hovertext=(
					"similarity=" + positives["similarity"].round(3).astype(str)
					+ "<br>id=" + positives["id"].astype(str)
					+ "<br>type=" + positives["prompt_name"].astype(str)
				),
				hoverinfo="text",
			)
		)

	if negatives is not None and not negatives.empty:
		fig.add_trace(
			go.Scatter(
				x=negatives["som_x"].to_numpy(dtype=float) + 0.5,
				y=negatives["som_y"].to_numpy(dtype=float) + 0.5,
				mode="markers",
				# Purple is distinct from RdBu and avoids the heatmap's red/blue hues.
				# Note: for *-open symbols Plotly typically uses `marker.color` for the outline.
				marker={
					"size": 14,
					"symbol": "diamond-open",
					"color": "rgba(160, 0, 255, 1.0)",
					"line": {"width": 3, "color": "rgba(160, 0, 255, 1.0)"},
				},
				name="Different",
				hovertext=(
					"similarity=" + negatives["similarity"].round(3).astype(str)
					+ "<br>id=" + negatives["id"].astype(str)
					+ "<br>type=" + negatives["prompt_name"].astype(str)
				),
				hoverinfo="text",
			)
		)

	if user_bmu is not None:
		fig.add_trace(
			go.Scatter(
				x=[user_bmu[0] + 0.5],
				y=[user_bmu[1] + 0.5],
				mode="markers",
				# Gold stands out without overlapping the heatmap's red/blue palette.
				marker={"size": 18, "symbol": "x", "color": "rgba(255, 215, 0, 1.0)"},
				name="Your BMU",
				hovertext=f"BMU=({user_bmu[0]},{user_bmu[1]})",
				hoverinfo="text",
			)
		)

	fig.update_layout(
		height=620,
		margin={"l": 10, "r": 10, "t": 30, "b": 10},
		legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},

		xaxis={
			"title": "SOM X",
			"range": [0, params.som_x],
			"tickmode": "linear",
			"tick0": 0,
			"dtick": 1,
			"constrain": "domain",
		},
		yaxis={
			"title": "SOM Y",
			"range": [params.som_y, 0],
			"tickmode": "linear",
			"tick0": 0,
			"dtick": 1,
			"scaleanchor": "x",
		},
	)
	return fig


def main() -> None:
	st.set_page_config(page_title="SOPM Interface", layout="wide")
	st.title("Self-Organizing Prompt Maps for Lightweight Prompt Adaptation")

	# Session state wiring: keeps text areas/buttons connected across reruns.
	st.session_state.setdefault("user_prompt_text", "")
	st.session_state.setdefault("improved_prompt_text", "")
	st.session_state.setdefault("llm_test_prompt_text", "")
	st.session_state.setdefault("llm_response_text", "")

	with st.sidebar:
		st.header("Config")
		# Defaults come from the environment when set (Docker sets them so the
		# app points at the `ollama` service instead of localhost); otherwise the
		# original local values apply.
		csv_path = st.text_input("Dataset CSV", value=os.environ.get("SOPM_DATA_CSV", DATA_CSV_DEFAULT))
		artifact_dir = st.text_input("Artifacts dir", value=os.environ.get("SOPM_ARTIFACT_DIR", ARTIFACT_DIR_DEFAULT))
		embed_model_name = st.text_input("Embedding model", value=os.environ.get("SOPM_EMBED_MODEL", "all-MiniLM-L6-v2"))
		ollama_base_url = st.text_input("Ollama base_url", value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
		llm_model_name = st.text_input("LLM model (Ollama)", value=os.environ.get("OLLAMA_MODEL", "qwen3.5:4b"))

		st.subheader("Recommendations")
		k_pos = st.number_input("k similar", min_value=0, max_value=20, value=5, step=1)
		k_neg = st.number_input("k different", min_value=0, max_value=20, value=3, step=1)
		radius_pos = st.number_input("similar radius", min_value=0, max_value=10, value=3, step=1)
		radius_neg = st.number_input("different radius (Manhattan)", min_value=0, max_value=30, value=5, step=1)

		st.subheader("SOM")
		som_x = st.number_input("som_x", min_value=2, max_value=30, value=5, step=1)
		som_y = st.number_input("som_y", min_value=2, max_value=30, value=10, step=1)
		num_iterations = st.number_input("iterations", min_value=100, max_value=200_000, value=10_000, step=100)
		sigma = st.number_input("sigma", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
		learning_rate = st.number_input("learning_rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01)
		random_seed = st.number_input("random_seed", min_value=0, max_value=10_000, value=42, step=1)

		params = SomParams(
			som_x=int(som_x),
			som_y=int(som_y),
			num_iterations=int(num_iterations),
			sigma=float(sigma),
			learning_rate=float(learning_rate),
			random_seed=int(random_seed),
		)

	col_left, col_right = st.columns([1.05, 1])

	with col_left:
		st.subheader("1) Your prompt")
		user_prompt = st.text_area("Paste or type your prompt here", key="user_prompt_text", height=220)
		run_button = st.button("Improve prompt")

		st.subheader("2) Result")

		last_error = ""
		improved_prompt = ""
		bmu = None
		pos_df = pd.DataFrame()
		neg_df = pd.DataFrame()

		if run_button:
			if not user_prompt.strip():
				last_error = "Empty prompt. Enter some text to continue."
			else:
				try:
					with st.spinner("Loading/training SOM…"):
						t0 = time.time()
						bundle = _get_bundle_cached(csv_path, artifact_dir, embed_model_name, params)
						st.write(f"Dataset: {len(bundle.df)} prompts")
						st.write(f"Time (SOM bundle): {time.time() - t0:.2f}s")

					with st.spinner("Generating embedding and recommendations…"):
						db_path = _embeddings_db_path(artifact_dir)
						# The user's prompt is the "query" side for asymmetric models.
						prompt_emb = _encode_texts_cached(
							model_name=embed_model_name,
							texts=[user_prompt],
							db_path=db_path,
							role="query",
						)[0]
						bmu, pos_df, neg_df = _recommend_for_embedding(
							prompt_emb=prompt_emb,
							bundle=bundle,
							k_pos=int(k_pos),
							k_neg=int(k_neg),
							radius_pos=int(radius_pos),
							radius_neg=int(radius_neg),
						)

					with st.spinner("Calling LLM (Ollama) to improve the prompt…"):
						client = _load_ollama_client(ollama_base_url)
						improved_prompt = _create_personalized_prompt(
							client=client,
							model_name=llm_model_name,
							original_prompt=user_prompt,
							positives=pos_df,
							negatives=neg_df,
						)
						st.session_state["improved_prompt_text"] = improved_prompt

				except Exception as e:
					last_error = str(e)

		if last_error:
			st.error(last_error)

		st.text_area("Improved prompt", key="improved_prompt_text", height=260)

		st.subheader("3) Similar prompts (positives)")
		if isinstance(pos_df, pd.DataFrame) and not pos_df.empty:
			st.dataframe(pos_df)
		else:
			st.write("No similar prompts found (try increasing the radius).")

		st.subheader("4) Different prompts (negatives)")
		if isinstance(neg_df, pd.DataFrame) and not neg_df.empty:
			st.dataframe(neg_df)
		else:
			st.write("No different prompts found (try adjusting the radius).")

		st.subheader("5) Run prompt on the LLM (Ollama)")
		if not st.session_state.get("llm_test_prompt_text"):
			# Seed with improved prompt if available, else the original.
			seed_text = st.session_state.get("improved_prompt_text") or st.session_state.get("user_prompt_text")
			st.session_state["llm_test_prompt_text"] = seed_text
		test_prompt = st.text_area("Prompt to send to the LLM", key="llm_test_prompt_text", height=180)
		max_tokens = st.number_input("max_tokens", min_value=16, max_value=4096, value=800, step=16)
		run_llm = st.button("Run on Ollama")
		if run_llm:
			try:
				with st.spinner("Calling Ollama…"):
					client = _load_ollama_client(ollama_base_url)
					reply = _call_llm_chat(
						client=client,
						model_name=llm_model_name,
						user_content=test_prompt,
						max_tokens=int(max_tokens),
					)
					st.session_state["llm_response_text"] = reply
			except Exception as e:
				st.error(f"Error calling Ollama: {e}")

		st.text_area("LLM response", key="llm_response_text", height=180)
		if bmu is not None:
			st.caption(f"Your prompt's BMU on the SOM: ({bmu[0]}, {bmu[1]})")

	with col_right:
		st.markdown(
			"""
			<style>
			:root {
				--umatrix-card-bg: var(--secondary-background-color, var(--secondaryBackgroundColor, rgba(255, 255, 255, 0.98)));
				--umatrix-card-fg: var(--text-color, var(--textColor, rgba(49, 51, 63, 0.98)));
				--umatrix-card-border: rgba(49, 51, 63, 0.18);
				--umatrix-card-shadow: rgba(0, 0, 0, 0.10);
				--umatrix-icon-border: rgba(49, 51, 63, 0.25);
			}
			@media (prefers-color-scheme: dark) {
				:root {
					--umatrix-card-bg: var(--secondary-background-color, var(--secondaryBackgroundColor, rgba(38, 39, 48, 0.98)));
					--umatrix-card-fg: var(--text-color, var(--textColor, rgba(250, 250, 250, 0.98)));
					--umatrix-card-border: rgba(250, 250, 250, 0.14);
					--umatrix-card-shadow: rgba(0, 0, 0, 0.35);
					--umatrix-icon-border: rgba(250, 250, 250, 0.25);
				}
			}

			.umatrix-title-row { display: flex; align-items: center; gap: 0.5rem; margin: 0 0 0.25rem 0; }
			.umatrix-title-row h3 { margin: 0; }
			.umatrix-info { position: relative; display: inline-block; cursor: help; user-select: none; }
			.umatrix-info-icon { font-size: 0.95rem; padding: 0.1rem 0.35rem; border-radius: 999px; border: 1px solid var(--umatrix-icon-border); }
			.umatrix-card {
				visibility: hidden;
				opacity: 0;
				position: absolute;
				top: 1.55rem;
				left: -0.25rem;
				z-index: 9999;
				width: min(560px, 20vw);
				background: var(--umatrix-card-bg);
				color: var(--umatrix-card-fg);
				border: 1px solid var(--umatrix-card-border);
				border-radius: 0.6rem;
				padding: 0.75rem 0.9rem;
				box-shadow: 0 6px 18px var(--umatrix-card-shadow);
				transition: opacity 120ms ease-in-out;
			}
			.umatrix-info:hover .umatrix-card { visibility: visible; opacity: 1; }
			.umatrix-card p { margin: 0.35rem 0 0 0; }
			</style>

			<div class="umatrix-title-row">
			  <h3>SOM U-Matrix Map</h3>
			  <div class="umatrix-info" aria-label="Information about the U-Matrix">
				<span class="umatrix-info-icon">&#9432;</span>
				<div class="umatrix-card">
				  <div><b>What is the U-Matrix?</b></div>
				  <p>
					The U-Matrix (Unified Distance Matrix) shows the <b>average distance</b> between each SOM neuron's weights and its neighbors.
					It helps visualize the clustering structure learned by the map.
				  </p>
				  <p>
					In general: <b>higher values</b> (lighter regions) indicate <b>boundaries</b> between clusters, and <b>lower values</b> (darker regions)
					suggest <b>more homogeneous areas</b> where examples are more similar.
				  </p>
				</div>
			  </div>
			</div>
			""",
			unsafe_allow_html=True,
		)
		try:
			bundle = build_or_load_som_bundle(
				csv_path=csv_path,
				artifact_dir=artifact_dir,
				embed_model_name=embed_model_name,
				params=params,
			)
			fig = _build_som_figure(bundle=bundle, user_bmu=bmu, positives=pos_df, negatives=neg_df)
			st.plotly_chart(fig, use_container_width=True)

			st.subheader("Browse prompts on the map")
			counts = (
				bundle.df.groupby(["som_x", "som_y"], as_index=False)
				.size()
				.rename(columns={"size": "n_prompts"})
				.sort_values(["som_x", "som_y"], ascending=[True, True])
			)
			options = [f"({int(r.som_x)},{int(r.som_y)}) — {int(r.n_prompts)} prompts" for r in counts.itertuples(index=False)]
			option_to_node = {
				opt: (int(r.som_x), int(r.som_y))
				for opt, r in zip(options, counts.itertuples(index=False))
			}
			default_opt = options[0] if options else None
			if default_opt is not None:
				selected = st.selectbox("Select a neuron (x,y)", options=options, index=0)
				node = option_to_node[selected]
				subset = bundle.df[(bundle.df["som_x"] == node[0]) & (bundle.df["som_y"] == node[1])]
				st.dataframe(subset[["id", "prompt_name", "input", "target", "som_x", "som_y"]])
		except Exception as e:
			st.error(f"Error loading/plotting SOM: {e}")


if __name__ == "__main__":
	main()
