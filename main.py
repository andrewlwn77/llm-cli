import os
import re
import sqlite3
import requests
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import subprocess
import readline
import glob

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

def get_custom_prompt():
    username = os.getenv('USER') or os.getenv('USERNAME') or 'user'
    hostname = os.uname().nodename
    current_dir = os.getcwd().replace(os.path.expanduser('~'), '~')
    return f"(llm-mode) {username}@{hostname}:{current_dir}$ "

class Completer:
    def __init__(self):
        self.matches = []

    def complete(self, text, state):
        if state == 0:
            line = readline.get_line_buffer()
            parts = line.split()
            
            if not parts or line[-1] == ' ':
                # Complete commands
                self.matches = glob.glob(text + '*')
            elif len(parts) == 1 and parts[0] == 'cd':
                # Complete directories for cd command
                self.matches = [d for d in glob.glob(text + '*') if os.path.isdir(d)]
            else:
                # Complete filenames
                self.matches = glob.glob(text + '*')
        
        return self.matches[state] if state < len(self.matches) else None

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
    
    # Set up readline for command history
    histfile = os.path.join(os.path.expanduser("~"), ".llm_cli_history")
    try:
        readline.read_history_file(histfile)
    except FileNotFoundError:
        pass
    
    readline.set_history_length(1000)
    
    # Set up tab completion
    completer = Completer()
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(completer.complete)
    
    while True:
        try:
            user_input = input(get_custom_prompt())
            readline.write_history_file(histfile)
            
            if user_input.lower() == 'exit':
                break
            
            if is_question(user_input):
                # Generate response using LLM
                response = generate_response(user_input, model, tokenizer, scripting_env)
                print(f"Suggested command: {response}")
                execute = input("Do you want to execute this command? (y/n): ")
                if execute.lower() == 'y':
                    user_input = response
                else:
                    continue
            
            # Execute the command
            try:
                result = subprocess.run(user_input, shell=True, check=True, text=True, capture_output=True)
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error executing command: {e}")
                print(e.stderr)
            
            # Store the command in the database
            insert_command(conn, (user_input, "", "", scripting_env))

        except EOFError:
            break

    conn.close()

if __name__ == "__main__":
    main()