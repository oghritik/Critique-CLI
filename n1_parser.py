from sentence_transformers import SentenceTransformer
import numpy as np

class NLPCommandParser:
    def __init__(self):
        # Load a lightweight sentence transformer model
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Define command templates and their corresponding shell commands
        self.command_templates = {
            "show me the current directory": ("pwd", []),
            "list all files": ("ls", []),
            "list files with details": ("ls", ["-l"]),
            "show files": ("ls", []),
            "show files running currently": ("ls", []),
            "change directory to home": ("cd", ["~"]),
            "go to directory": ("cd", None),  # Requires argument extraction
            "print text": ("echo", None),  # Requires argument extraction
            "exit the shell": ("exit", []),
            "show command history": ("history", []),
            "check if command exists": ("type", None),  # Requires argument
            "monitor system resources": ("monitor", []),  # New template for monitor
        }
        
        # Cache embeddings for command templates
        self.template_embeddings = self.model.encode(list(self.command_templates.keys()))
        
        # Similarity threshold for matching
        self.threshold = 0.8

    def cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def extract_arguments(self, input_text, template, shell_command):
        """Extract arguments from input text based on the matched template."""
        if shell_command == "cd" and template == "go to directory":
            # Extract directory name from input (simple heuristic)
            words = input_text.lower().split()
            if "to" in words:
                dir_index = words.index("to") + 1
                if dir_index < len(words):
                    return [words[dir_index]]
            return []
        elif shell_command == "echo" and template == "print text":
            # Extract text after "print"
            words = input_text.lower().split()
            if "print" in words:
                text_index = words.index("print") + 1
                return words[text_index:]
            return []
        elif shell_command == "type" and template == "check if command exists":
            # Extract command name
            words = input_text.lower().split()
            if "command" in words:
                cmd_index = words.index("command") + 1
                if cmd_index < len(words):
                    return [words[cmd_index]]
            return []
        return []

    def parse_command(self, input_text):
        """Parse natural language input and return corresponding shell command."""
        # Encode the input text
        input_embedding = self.model.encode([input_text])[0]
        
        # Compute similarity with all templates
        similarities = [self.cosine_similarity(input_embedding, template_emb) 
                       for template_emb in self.template_embeddings]
        
        # Find the best match
        max_similarity = max(similarities) if similarities else 0
        if max_similarity < self.threshold:
            return None  # No match found
        
        best_match_idx = similarities.index(max_similarity)
        best_template = list(self.command_templates.keys())[best_match_idx]
        shell_command, default_args = self.command_templates[best_template]
        
        # Extract arguments if needed
        args = default_args if default_args is not None else self.extract_arguments(input_text, best_template, shell_command)
        
        return (shell_command, args)