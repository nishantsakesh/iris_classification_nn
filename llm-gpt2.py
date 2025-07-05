from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_text(prompt, model, tokenizer, max_new_tokens, temperature, top_k):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        model.to('cuda')

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens, 
        do_sample=True,                
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id 
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    
    return generated_text

if __name__ == "__main__":
    model_name = "gpt2" 
    prompt_text = "Once upon a time in a faraway land," 
    num_generated_tokens = 50 

    # 1. Model and Tokenizer load 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer)) 

    # 2. Text generate in temperature = 0.7 
    print("Generating text with Temperature = 0.7...")
    generated_text_0_7 = generate_text(
        prompt_text,
        model,
        tokenizer,
        max_new_tokens=num_generated_tokens,
        temperature=0.7,
        top_k=50
    )

    # 3. Text generate in temperature = 1.0 
    print("Generating text with Temperature = 1.0...")
    generated_text_1_0 = generate_text(
        prompt_text,
        model,
        tokenizer,
        max_new_tokens=num_generated_tokens,
        temperature=1.0,
        top_k=50
    )

    # 4. Outputs in text file 
    output_filename = "generated_samples.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt_text}\n\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"Generated Text (Temperature = 0.7):\n{generated_text_0_7}\n\n")
        f.write("-" * 50 + "\n\n")
        f.write(f"Generated Text (Temperature = 1.0):\n{generated_text_1_0}\n")

    print(f"Generated texts saved to {output_filename}")