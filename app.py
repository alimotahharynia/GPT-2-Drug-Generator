import os
import json
import torch
import logging
import tempfile
import gradio as gr
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

# Global logging setup
def setup_logging(output_file="app.log"):
    log_filename = os.path.splitext(output_file)[0] + ".log"
    logging.getLogger().handlers.clear()
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Load model and tokenizer
def load_model_and_tokenizer(model_name):
    logging.info(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            logging.info("Moving model to CUDA device.")
            model = model.to("cuda")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {e}")
        raise RuntimeError(f"Failed to load model and tokenizer: {e}")

# Load the dataset
def load_uniprot_dataset(dataset_name, dataset_key):
    try:
        dataset = load_dataset(dataset_name, dataset_key)
        uniprot_to_sequence = {row["UniProt_id"]: row["Sequence"] for row in dataset["uniprot_seq"]}
        logging.info("Dataset loaded and processed successfully.")
        return uniprot_to_sequence
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise RuntimeError(f"Failed to load dataset: {e}")

def save_smiles_to_file(results):
    file_path = os.path.join(tempfile.gettempdir(), "generated_smiles.json")
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    return file_path

# SMILES Generator
class SMILESGenerator:
    def __init__(self, model, tokenizer, uniprot_to_sequence):
        self.model = model
        self.tokenizer = tokenizer
        self.uniprot_to_sequence = uniprot_to_sequence
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.generation_kwargs = {
            "do_sample": True,
            "top_k": 9,
            "max_length": 1024,
            "top_p": 0.9,
            "num_return_sequences": 10,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }

    def generate_smiles(self, sequence, num_generated, progress_callback=None):
        generated_smiles_set = set()
        prompt = f"<|startoftext|><P>{sequence}<L>"
        encoded_prompt = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)

        logging.info(f"Generating SMILES for sequence: {sequence[:10]}...")
        retries = 0
        while len(generated_smiles_set) < num_generated:
            if retries >= 30:
                logging.warning("Max retries reached. Returning what has been generated so far.")
                break

            sample_outputs = self.model.generate(encoded_prompt, **self.generation_kwargs)
            for i, sample_output in enumerate(sample_outputs):
                output_decode = self.tokenizer.decode(sample_output, skip_special_tokens=False)
                try:
                    generated_smiles = output_decode.split("<L>")[1].split("<|endoftext|>")[0]
                    if generated_smiles not in generated_smiles_set:
                        generated_smiles_set.add(generated_smiles)
                except (IndexError, AttributeError) as e:
                    logging.warning(f"Failed to parse SMILES due to error: {str(e)}. Skipping.")
            
            if progress_callback:
                progress_callback((retries + 1) / 30)

            retries += 1

        logging.info(f"SMILES generation completed. Generated {len(generated_smiles_set)} SMILES.")
        return list(generated_smiles_set)

# Gradio interface
def generate_smiles_gradio(sequence_input=None, uniprot_id=None, num_generated=10):
    results = {}

    # Process protein sequences
    if sequence_input:
        sequences = [seq.strip() for seq in sequence_input.split(",") if seq.strip()]
        for seq in sequences:
            try:
                # Always attempt to generate SMILES from the sequence (regardless of validity)
                smiles = generator.generate_smiles(seq, num_generated)
                results[seq] = {"sequence": seq, "smiles": smiles}
            except Exception as e:
                results[seq] = {"sequence": seq, "error": f"Error generating SMILES: {str(e)}"}

    # Process UniProt IDs
    if uniprot_id:
        uniprot_ids = [uid.strip() for uid in uniprot_id.split(",") if uid.strip()]
        for uid in uniprot_ids:
            sequence = uniprot_to_sequence.get(uid, None)  # None if not found
            try:
                if sequence:
                    smiles = generator.generate_smiles(sequence, num_generated)
                    results[uid] = {"sequence": sequence, "smiles": smiles}
                else:
                    # UniProt ID not found
                    results[uid] = {"sequence": "N/A", "error": f"UniProt ID {uid} not found in dataset."}
            except Exception as e:
                results[uid] = {"sequence": "N/A", "error": f"Error generating SMILES: {str(e)}"}

    if not results:
        return {"error": "No valid input provided. Please try again with different sequences or UniProt IDs."}

    # Save
    file_path = save_smiles_to_file(results)
    return results, file_path


# Main initialization and Gradio setup
if __name__ == "__main__":
    setup_logging()
    model_name = "alimotahharynia/DrugGen"
    dataset_name = "alimotahharynia/approved_drug_target"
    dataset_key = "uniprot_sequence"

    model, tokenizer = load_model_and_tokenizer(model_name)
    uniprot_to_sequence = load_uniprot_dataset(dataset_name, dataset_key)

    generator = SMILESGenerator(model, tokenizer, uniprot_to_sequence)

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="teal")) as iface:
        custom_css = """
        html, body {
            background-color: #111111 !important;  /* Dark gray background */
            color: #ffffff !important;  /* White text */
            font-family: 'Roboto', sans-serif !important;
        }

        #app-title {
            text-align: center;
            font-size: 36px;
            font-weight: 700;
            color: #ffffff !important;  /* White title for dark background */
            margin-bottom: 20px;
        }

        #description {
            font-size: 18px;
            margin-bottom: 40px;
            text-align: center;
            color: #dddddd !important;  /* Lighter text color for description */
        }

        .gr-button {
            padding: 12px 24px;
            font-weight: bold;
            background-color: #0066cc !important;  /* Muted blue button */
            color: white !important;
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
        }

        .gr-button:hover {
            background-color: #005cbf !important;  /* Darker blue on hover */
            transform: translateY(-2px);
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
        }

        .gr-input:focus {
            border-color: #0066cc !important;
            box-shadow: 0 0 8px rgba(0, 102, 204, 0.3) !important;
        }

        .gr-output {
            background-color: #444444 !important;  /* Lighter gray background for outputs */
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            color: #ffffff !important;  /* White text for outputs */
        }

        .error-message {
            background-color: #f8d7da !important;
            border-color: #f5c6cb !important;
            color: #721c24 !important;
            padding: 15px;
            border-radius: 8px;
        }

        .success-message {
            background-color: #d4edda !important;
            border-color: #c3e6cb !important;
            color: #155724 !important;
            padding: 15px;
            border-radius: 8px;
        }

        .gr-row {
            margin-bottom: 20px;
        }
"""

        iface.css = custom_css
        gr.Markdown("## GPT-2 Drug Generator", elem_id="app-title")
        gr.Markdown(
            "Generate **drug-like SMILES structures** from protein sequences or UniProt IDs. "
            "Input data, specify parameters, and download the results.",
            elem_id="description"
        )

        with gr.Row():
            sequence_input = gr.Textbox(
                label="Protein Sequences",
                placeholder="Enter sequences separated by commas (e.g., MGAASGRRGP, MGETLGDSPI, ...)",
                lines=3,
            )
            uniprot_id_input = gr.Textbox(
                label="UniProt IDs",
                placeholder="Enter UniProt IDs separated by commas (e.g., P12821, P37231, ...)",
                lines=1,
            )

        num_generated_slider = gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            value=10,
            label="Number of Unique SMILES to Generate",
        )

        output = gr.JSON(label="Generated SMILES")
        file_output = gr.File(label="Download Results as JSON")

        generate_button = gr.Button("Generate SMILES", elem_id="generate-button")

        generate_button.click(
            generate_smiles_gradio,
            inputs=[sequence_input, uniprot_id_input, num_generated_slider],
            outputs=[output, file_output]
        )

        gr.Markdown("""
        ### How to Cite:
        If you use this tool in your research, please cite the following work:
        
        ```bibtex
        @misc{sheikholeslami2024druggenadvancingdrugdiscovery,
            title={DrugGen: Advancing Drug Discovery with Large Language Models and Reinforcement Learning Feedback}, 
            author={Mahsa Sheikholeslami and Navid Mazrouei and Yousof Gheisari and Afshin Fasihi and Matin Irajpour and Ali Motahharynia},
            year={2024},
            eprint={2411.14157},
            archivePrefix={arXiv},
            primaryClass={q-bio.QM},
            url={https://arxiv.org/abs/2411.14157}, 
        }
        ```

        This will help us maintain the tool and support future development!
        """)

        iface.launch(allowed_paths=["/tmp"])
