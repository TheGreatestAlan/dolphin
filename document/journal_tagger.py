import os
import json
from pathlib import Path

from agent_server.llms.LlamaCppRestLLM import LlamaCppRestLLM
from agent_server.llms.OllamaRestLLM import OllamaLLM

# Configuration
ROOT_DIRECTORY = 'C:\\workspace\\obsidian'
OUTPUT_FILE = 'dev/journal_topic_tags.json'
BASE_URL = "http://localhost:11434"
#BASE_URL = "http://192.168.1.3:8080"
MODEL_NAME = "qwen2.5:3b"

# Initialize the Ollama LLM instance
#llm = LlamaCppRestLLM(BASE_URL)
llm = OllamaLLM(BASE_URL, MODEL_NAME)


def enforce_json_structure(response_text):
    """Calls the LLM to enforce JSON structure if initial parsing of topic tags fails."""
    try:
        # Define the fallback system message to enforce JSON structure
        system_message = (
            "You are a JSON structure enforcer that takes any input of topic tags "
            "and outputs it as structured JSON strictly following this structure:\n\n"
            "{\"topic_tags\":[\"tag_1\", \"tag_2\", \"tag_n\"]}"
        )

        # Make the fallback LLM call
        correction_response = llm.generate_response(
            response_text,
            system_message=system_message
        )

        # Parse the corrected JSON response
        return json.loads(correction_response).get('topic_tags', [])
    except Exception as e:
        print(f"Error enforcing JSON structure: {e}")
        return []


def topic_tag_journal_entry(file_path):
    """Uses the LLM to generate topic tags for the content of a journal entry and appends them to the file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            journal_content = file.read()

        # System message with detailed tagging instructions
        system_message = (
            "You are a tagging assistant. Your role is to analyze the provided content and generate relevant topic tags. "
            "If the content is a journal entry, always include a 'journal' tag in the list. "
            "Focus on key themes and subjects discussed in the content. "
            "Respond with a concise list of topic tags in a JSON array labeled 'topic_tags', "
            "like so: {\"topic_tags\":[\"Journal\",\"travel\",\"inspiration\",\"musiclyrics\"]}. "
            "Respond only with JSON and nothing else. Additional extraneous tokens will waste "
            "energy forcing more fossil fuels to be burned killing the planet."
        )

        # User prompt contains only the journal content
        user_prompt = journal_content

        # Generate response
        response = llm.generate_response(user_prompt, system_message=system_message)

        # Strip ```json``` and ``` delimiters if present
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()

        # Attempt to parse the JSON response
        try:
            topic_tags = json.loads(response).get('topic_tags', [])
        except json.JSONDecodeError:
            print("Initial JSON parsing failed. Attempting to correct JSON structure.")
            topic_tags = enforce_json_structure(response)

        # Prepare tags for appending to the file
        tag_string = "\n" + " ".join(f"#{tag}" for tag in topic_tags)

        # Append tags to the end of the file
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(tag_string)

        return {"file": str(file_path), "tags": topic_tags}

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {"file": str(file_path), "tags": [], "error": str(e)}


def process_all_journals(root_directory):
    """Process all journal files in the root directory to generate topic tags."""
    tagged_journals = []

    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.md'):
                file_path = os.path.join(dirpath, filename)
                result = topic_tag_journal_entry(file_path)
                tagged_journals.append(result)

    # Save results to output JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
        json.dump(tagged_journals, output_file, indent=4)
        print(f"Tagged journal entries have been saved to {OUTPUT_FILE}")


def process_single_journal(file_path):
    """Process a single specified journal file for topic tagging and print the result."""
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return

    result = topic_tag_journal_entry(file_path)
    print("Single Journal Topic Tags Result:")
    print(json.dumps(result, indent=4))

def process_journals_recursively(root_directory):
    """Recursively process all journal files in the root directory and its subdirectories to generate topic tags."""
    tagged_journals = []

    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.md'):
                file_path = os.path.join(dirpath, filename)
                result = topic_tag_journal_entry(file_path)
                tagged_journals.append(result)

    # Save results to output JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
        json.dump(tagged_journals, output_file, indent=4)
        print(f"Tagged journal entries have been saved to {OUTPUT_FILE}")

# Usage example
def main():
    """Main function to process journals for topic tagging."""
    #process_single_journal(r"C:\workspace\obsidian\gitVault\Evernote\antares07923's notebook\Untitled Note.298.md")
    process_journals_recursively(r"C:\workspace\obsidian\gitVault\Evernote\antares07923's notebook")


if __name__ == "__main__":
    main()
