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

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the application:
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

## Development

To run tests:

```sh
python -m unittest discover tests
```

## License

[MIT License](LICENSE)