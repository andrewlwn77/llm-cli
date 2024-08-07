# CLI Application with LLM Integration

This CLI application uses the Phi-3.1-mini-4k-instruct-IQ2_M model to answer questions about command usage within the CLI. It also uses SQLite for storing additional context and can detect the current scripting environment.

## Installation

### Linux

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/cli-application.git
   cd cli-application
   ```

2. Set up a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Upgrade pip and install wheel:
   ```sh
   pip install --upgrade pip
   pip install wheel
   ```

4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

   If you encounter issues, try installing llama-cpp-python separately:
   ```sh
   pip install llama-cpp-python==0.2.86+cpu --no-cache-dir
   ```

5. Run the application:
   ```sh
   python main.py
   ```

### Future Considerations

- For Mac: Ensure compatibility with Homebrew.
- For Windows: Provide PowerShell scripts for installation.

## Usage

1. Start the application by running `python main.py`.
2. If the model is not downloaded, the application will download it automatically.
3. Enter your command or question when prompted.
4. The application will either provide information about the command or generate a response using the LLM model.
5. Type 'exit' to quit the application.

## Features

- Integration of Phi-3.1-mini-4k-instruct-IQ2_M model for answering questions
- Automatic download of the model if not present
- SQLite database for storing command information
- Automatic detection of the current scripting environment
- Command information gathering using the 'man' command
- Question detection for triggering LLM responses

## Troubleshooting

If you encounter issues with installing or running the application, please ensure that:

1. You have Python 3.7 or higher installed.
2. You are using a virtual environment to avoid conflicts with system-wide packages.
3. Your pip and wheel are up to date.
4. If you're still having issues with llama-cpp-python, try installing it separately using the CPU-only version as mentioned in the installation steps.

If you still face issues, please report them in the Issues section of this repository.

## Development

To run tests:

```sh
python -m unittest discover tests
```

## License

[MIT License](LICENSE)