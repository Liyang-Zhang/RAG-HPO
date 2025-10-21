import click
import pandas as pd
import os
import sys

from .vectorization import build_knowledge_base
from .pipeline import run_rag_pipeline, load_prompts, LLMClient, system_message_I, system_message_II, check_and_initialize_llm
from .utils import logger, cleanup

@click.group()
def cli():
    """RAG-HPO: Retrieval-Augmented Generation for Human Phenotype Ontology annotation."""
    pass

@cli.command()
@click.option('--obo-url', default="https://purl.obolibrary.org/obo/hp.obo", help='URL to the HPO OBO file.')
@click.option('--obo-path', default="hp.obo", help='Local path to save the HPO OBO file.')
@click.option('--refresh-days', default=14, type=int, help='Refresh OBO file if older than this many days.')
@click.option('--meta-output', default="hpo_meta.json", help='Output path for HPO metadata JSON.')
@click.option('--vec-output', default="hpo_embedded.npz", help='Output path for HPO embeddings NPZ.')
@click.option('--hpo-full-csv', default="hpo_terms_full.csv", help='Output path for full HPO terms CSV.')
@click.option('--use-sbert/--no-sbert', default=True, help='Use SBERT for embeddings (default: True).')
@click.option('--sbert-model', default='pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb', help='SBERT model name.')
@click.option('--bge-model', default='BAAI/bge-small-en-v1.5', help='BGE model name.')
@click.option('--limit', type=int, help='Limit the number of HPO terms processed (for testing).')
def build_kb(obo_url, obo_path, refresh_days, meta_output, vec_output, hpo_full_csv, use_sbert, sbert_model, bge_model, limit):
    """Builds or updates the HPO knowledge base (metadata and embeddings)."""
    logger.log("Starting knowledge base construction...")
    try:
        build_knowledge_base(
            obo_url=obo_url,
            obo_path=obo_path,
            refresh_days=refresh_days,
            meta_output_path=meta_output,
            vec_output_path=vec_output,
            hpo_full_csv_path=hpo_full_csv,
            use_sbert=use_sbert,
            sbert_model=sbert_model,
            bge_model=bge_model,
            limit=limit
        )
        logger.log("Knowledge base construction completed successfully.")
    except Exception as e:
        logger.log(f"Error building knowledge base: {e}")
        sys.exit(1)

@cli.command()
@click.option('--input-csv', type=click.Path(exists=True), required=True, help='Path to input CSV file with clinical notes.')
@click.option('--output-csv', type=click.Path(), help='Path to save the output CSV with HPO annotations.')
@click.option('--output-json-raw', type=click.Path(), help='Path to save the raw JSON output from LLM.')
@click.option('--display/--no-display', default=False, help='Display results in the terminal.')
@click.option('--meta-path', default="hpo_meta.json", help='Path to HPO metadata JSON.')
@click.option('--vec-path', default="hpo_embedded.npz", help='Path to HPO embeddings NPZ.')
@click.option('--use-sbert/--no-sbert', default=True, help='Use SBERT for embeddings (default: True).')
@click.option('--sbert-model', default='pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb', help='SBERT model name.')
@click.option('--bge-model', default='BAAI/bge-small-en-v1.5', help='BGE model name.')
@click.option('--api-key', envvar='LLM_API_KEY', help='API key for the LLM. Can be set via LLM_API_KEY environment variable.')
@click.option('--base-url', envvar='LLM_BASE_URL', default="https://api.groq.com/openai/v1/chat/completions", help='Base URL for the LLM API. Can be set via LLM_BASE_URL environment variable.')
@click.option('--llm-model-name', envvar='LLM_MODEL_NAME', default="llama3-groq-70b-8192-tool-use-preview", help='LLM model name. Can be set via LLM_MODEL_NAME environment variable.')
@click.option('--max-tokens-per-day', default=500000, type=int, help='Max tokens per day for LLM.')
@click.option('--max-queries-per-minute', default=30, type=int, help='Max queries per minute for LLM.')
@click.option('--temperature', default=0.7, type=float, help='LLM temperature.')
@click.option('--system-prompts-file', default="system_prompts.json", help='Path to system prompts JSON file.')
def process(input_csv, output_csv, output_json_raw, display, meta_path, vec_path, use_sbert, sbert_model, bge_model,
            api_key, base_url, llm_model_name, max_tokens_per_day, max_queries_per_minute, temperature, system_prompts_file):
    """Processes clinical notes to extract HPO terms using the RAG pipeline."""
    if not api_key:
        logger.log("[FATAL] LLM API key is required. Please provide it via --api-key or LLM_API_KEY environment variable.")
        sys.exit(1)

    logger.log("Starting RAG pipeline processing...")
    try:
        input_df = pd.read_csv(input_csv)
        run_rag_pipeline(
            input_data=input_df,
            output_csv_path=output_csv,
            output_json_raw_path=output_json_raw,
            display_results=display,
            meta_path=meta_path,
            vec_path=vec_path,
            use_sbert=use_sbert,
            sbert_model=sbert_model,
            bge_model=bge_model,
            api_key=api_key,
            base_url=base_url,
            llm_model_name=llm_model_name,
            max_tokens_per_day=max_tokens_per_day,
            max_queries_per_minute=max_queries_per_minute,
            temperature=temperature,
            system_prompts_file=system_prompts_file
        )
        logger.log("RAG pipeline processing completed successfully.")
    except Exception as e:
        logger.log(f"Error during RAG pipeline processing: {e}")
        sys.exit(1)

def main():
    cli()

if __name__ == '__main__':
    main()
