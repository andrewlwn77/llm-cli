import os
import re
import sqlite3
import requests
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "microsoft/phi-1"  # Updated model name

def download_model(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)

def load_llm_model(model_name):
    print("Loading the model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        print("Please ensure that the model is available and that transformers is correctly installed.")
        return None, None

def generate_response(prompt: str, model, tokenizer, scripting_env):
    try:
        full_prompt = f"""Instruction: You are a command line assistant for a {scripting_env} environment. Provide ONLY the exact command to accomplish the following task. Do not include any explanations, examples, or output.

Task: {prompt}

Command:"""
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=30, num_return_sequences=1, temperature=0.1, top_p=0.95, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the command part and remove any newlines
        command = response.split("Command:")[-1].strip().split('\n')[0]
        return command
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response. Please try again."

def create_connection(db_file):
    conn = sqlite3.connect(db_file)
    return conn

def create_table(conn):
    sql_create_commands_table = """CREATE TABLE IF NOT EXISTS commands (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL,
                                    description text,
                                    usage_examples text,
                                    scripting_environment text
                                );"""
    conn.execute(sql_create_commands_table)

def insert_command(conn, command):
    sql = '''INSERT INTO commands(name, description, usage_examples, scripting_environment)
             VALUES(?,?,?,?)'''
    cur = conn.cursor()
    cur.execute(sql, command)
    conn.commit()

def check_command(command_name, conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM commands WHERE name=?", (command_name,))
    rows = cur.fetchall()
    return rows

def gather_command_info(command_name):
    import subprocess
    result = subprocess.run(['man', command_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return result.stdout.decode('utf-8')
    else:
        return f"No manual entry for {command_name}"

def detect_scripting_environment():
    shell = os.getenv('SHELL', '')
    if 'bash' in shell:
        return 'bash'
    elif 'zsh' in shell:
        return 'zsh'
    elif 'powershell' in shell.lower():
        return 'powershell'
    else:
        return 'unknown'

def is_question(input_text):
    question_patterns = [
        r'\bhow\b', r'\bwhat\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b', 
        r'\bcan\b.*\?', r'\bcould\b.*\?', r'\bshould\b.*\?', r'\bwould\b.*\?'
    ]
    return any(re.search(pattern, input_text, re.IGNORECASE) for pattern in question_patterns)

def main():
    model, tokenizer = load_llm_model(MODEL_NAME)
    if model is None or tokenizer is None:
        print("Failed to load the model. Exiting.")
        return

    db_file = "cli_commands.db"
    conn = create_connection(db_file)
    create_table(conn)
    
    scripting_env = detect_scripting_environment()
    print(f"Detected scripting environment: {scripting_env}")
    
    while True:
        user_input = input("Enter your command or question (or 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            break
        
        if is_question(user_input):
            # Generate response using LLM
            response = generate_response(user_input, model, tokenizer, scripting_env)
            print(f"LLM Response: {response}")
        else:
            # Check if command exists in database
            command_info = check_command(user_input, conn)
            
            if command_info:
                print(f"Command information from database: {command_info}")
            else:
                # Gather information about the command
                command_info = gather_command_info(user_input)
                print(f"Command information: {command_info}")
                
                # Store the information in the database
                insert_command(conn, (user_input, command_info, "", scripting_env))
                print("Command information stored in the database.")
    
    conn.close()
    
if __name__ == "__main__":
    main()