from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate_answer(context: str, question: str) -> str:
    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 dtype=torch.float16,
                                                 device_map='auto')
    prompt = f"""You are a precise technical assistant. Answer the question using ONLY the provided Context. 
If the Context does not contain the answer, reply exactly with: "I cannot answer based on the provided context." Do not guess, do not add external information, and do not make up code.

Context:
---
{context}
---

Question: {question}

Answer:"""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    final_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return final_answer.strip()
