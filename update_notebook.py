import json
import os

notebook_path = 'c:/Users/aalam23/Documents/My Code/VLM/VLM_AV_NLP.ipynb'

# Define the new cells to append
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 7: Load Model\n",
            "\n",
            "Load the `google/paligemma-3b-mix-224` model with 4-bit quantization for efficient memory usage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [Step 7] Load Model\n",
            "model_id = \"google/paligemma-3b-mix-224\"\n",
            "\n",
            "quantization_config = BitsAndBytesConfig(\n",
            "    load_in_4bit=True,\n",
            "    bnb_4bit_quant_type=\"nf4\",\n",
            "    bnb_4bit_compute_dtype=torch.bfloat16\n",
            ")\n",
            "\n",
            "model = PaliGemmaForConditionalGeneration.from_pretrained(\n",
            "    model_id,\n",
            "    quantization_config=quantization_config,\n",
            "    device_map=\"auto\"\n",
            ")\n",
            "\n",
            "processor = AutoProcessor.from_pretrained(model_id)\n",
            "print(\"Model and Processor loaded.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 8: Fine-Tuning (LoRA)\n",
            "\n",
            "Configure Low-Rank Adaptation (LoRA) to fine-tune only a small subset of parameters, making training faster and less memory-intensive."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [Step 8] Fine-Tuning Setup\n",
            "from peft import get_peft_model, LoraConfig, TaskType\n",
            "\n",
            "lora_config = LoraConfig(\n",
            "    r=8,\n",
            "    target_modules=[\"q_proj\", \"o_proj\", \"k_proj\", \"v_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
            "    task_type=TaskType.CAUSAL_LM,\n",
            ")\n",
            "\n",
            "model = get_peft_model(model, lora_config)\n",
            "model.print_trainable_parameters()\n",
            "\n",
            "# Note: Actual training loop would go here using Trainer\n",
            "# For this demo, we will proceed to inference to show how to use the model."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 9: Inference\n",
            "\n",
            "Demonstrate how to use the model for Visual Question Answering. \n",
            "**Crucial:** During inference, we provide the **Image** and the **Question**, but NOT the answer. The model generates the answer."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [Step 9] Inference Demo\n",
            "from PIL import Image\n",
            "import requests\n",
            "\n",
            "# Load a sample image from the dataset\n",
            "sample_idx = 0\n",
            "image = dataset[sample_idx]['image']\n",
            "question = \"What is the weather condition?\"\n",
            "\n",
            "# Prepare input\n",
            "inputs = processor(text=question, images=image, return_tensors=\"pt\").to(\"cuda\")\n",
            "\n",
            "# Generate answer\n",
            "input_len = inputs[\"input_ids\"].shape[-1]\n",
            "with torch.inference_mode():\n",
            "    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)\n",
            "    generation = generation[0][input_len:]\n",
            "    decoded = processor.decode(generation, skip_special_tokens=True)\n",
            "\n",
            "print(f\"Question: {question}\")\n",
            "print(f\"Model Answer: {decoded}\")\n",
            "\n",
            "# Display image\n",
            "display(image)"
        ]
    }
]

# Load existing notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Append new cells
notebook['cells'].extend(new_cells)

# Save updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully.")
