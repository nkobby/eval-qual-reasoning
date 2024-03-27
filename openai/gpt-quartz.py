from openai import OpenAI
import json
import time

# Define API key
key = ''
client = OpenAI(api_key=key)

#variables for model evaluations
total_questions = 0
correct_answers = 0
limit = 10

incorrects = []
# Read train.jsonl file line by line
with open("./data/train.jsonl", "r") as json_file:
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

        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"answer the question by chosing {context_options} and explain your answer. return the answer in the JSON form with fields 'Answer' and 'Explanation'"},
            {"role": "user", "content": question_n_opts}
            ])
        
        # extract the answer
        response = json.loads(completion.choices[0].message.content)
        predicted_answer = response.get("Answer", "")

        #counting the correct answer
        if predicted_answer == item["answerKey"]:
            correct_answers += 1
            print("   =1")
        else:
            incorrects.append(item)
            print("   =0")
        ctr += 1
        time.sleep(5)
        
    #calculate the accuracy and printing the results
    accuracy = correct_answers / total_questions
    print("Total Questions:", total_questions)
    print("Correct Answers:", correct_answers)
    print("Accuracy:", accuracy) 