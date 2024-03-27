# %%
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# %% variables for model evaluations
total_questions = 0
correct_answers = 0
limit = 1
incorrects = []

# %% qa function
def generate_response(prompt):
    prompt = "What is the capital of the United Kingdom"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    outputs = model.generate(**inputs, max_length=30, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.batch_decode(outputs)
    return decoded_output[0].replace(prompt, "")
    
# %%
def eval_loop():
    # Read train.jsonl file line by line
    with open("../data/train.jsonl", "r") as json_file:
        lines = json_file.readlines()
        
        ctr = 1
        # Read each lines
        for line in lines:
            if ctr > limit:
                break
            print (str(ctr) + ": ")
            total_questions += 1
            item = json.loads(line)
            question = item['question']
            question_n_opts = question['stem']
            context_options = ' OR '.join([x['label'] for x in question['choices']])
            for choice in question['choices']:
                question_n_opts  = question_n_opts + f" ({choice['label']}) {choice['text']}"

            context = f"Answer the question by chosing {context_options} and explain your answer. Return the answer in the JSON form with fields 'Answer' and 'Explanation'."
            prompt =  f"{question_n_opts}. {context}"
            output = generate_response(prompt)
            print(output)

            # extract the answer
            response = json.loads(output)
            predicted_answer = response.get("Answer", "")

            #counting the correct answer
            if predicted_answer == item["answerKey"]:
                correct_answers += 1
                print("   =1")
            else:
                incorrects.append(item)
                print("   =0")
            ctr += 1

if __name__ == "main":
    print("call eval loop)")
    eval_loop()