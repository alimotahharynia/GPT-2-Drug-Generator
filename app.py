import os
import json
import torch
import logging
import tempfile
import gradio as gr
from gradio.themes import Soft
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
    uniprot_counter = 0  # Counter for sequences without UniProt IDs

    # Process sequence inputs and include UniProt ID if found
    if sequence_input:
        sequences = [seq.strip() for seq in sequence_input.split(",") if seq.strip()]
        for seq in sequences:
            try:
                # Find the corresponding UniProt ID for the sequence
                uniprot_id_for_seq = [uid for uid, s in uniprot_to_sequence.items() if s == seq]
                if uniprot_id_for_seq:
                    uniprot_id_for_seq = uniprot_id_for_seq[0]
                else:
                    # Assign a number as the key for sequences without UniProt IDs
                    uniprot_id_for_seq = str(uniprot_counter)
                    uniprot_counter += 1
                
                # Generate SMILES for the sequence
                smiles = generator.generate_smiles(seq, num_generated)
                
                # UniProt ID or the numeric key as the key
                results[uniprot_id_for_seq] = {
                    "sequence": seq,
                    "smiles": smiles
                }

            except Exception as e:
                results[str(uniprot_counter)] = {
                    "sequence": seq,
                    "error": f"Error generating SMILES: {str(e)}"
                }
                uniprot_counter += 1

    # Process UniProt ID inputs and include sequence if found
    if uniprot_id:
        uniprot_ids = [uid.strip() for uid in uniprot_id.split(",") if uid.strip()]
        for uid in uniprot_ids:
            sequence = uniprot_to_sequence.get(uid, "N/A")
            try:
                # Generate SMILES for the sequence found
                if sequence != "N/A":
                    smiles = generator.generate_smiles(sequence, num_generated)
                    results[uid] = {
                        "sequence": sequence,
                        "smiles": smiles
                    }
                else:
                    results[uid] = {
                        "sequence": "N/A",
                        "error": f"UniProt ID {uid} not found in the dataset."
                    }
            except Exception as e:
                results[uid] = {
                    "sequence": "N/A",
                    "error": f"Error generating SMILES: {str(e)}"
                }

    # Check if no results were generated
    if not results:
        return {"error": "No SMILES generated. Please try again with different inputs."}

    # Save results to a file
    file_path = save_smiles_to_file(results)

    # Return both results (JSON) and the file path
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

    # Apply Gradio theme
    theme = Soft(primary_hue="indigo", secondary_hue="teal")

    css = """
        #app-title {
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
        }
        #description {
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
        }
        #file-output {
            height: 90px;
        }
        #generate-button {
            height: 40px;
            color: #333333 !important;
        }      
    """

    with gr.Blocks(theme=theme, css=css) as iface:
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
                lines=3,
            )

        num_generated_slider = gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            value=10,
            label="Number of Unique SMILES to Generate",
        )

        output = gr.JSON(label="Generated SMILES")
        file_output = gr.File(
        label="Download Results as JSON",
        elem_id=["file-output"]
)

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
