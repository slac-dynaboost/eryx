import re

class LogAnalyzer:
    """
    Provides functionality for analyzing logs produced by the OnePhonon model.
    """

    def __init__(self, log_content):
        """Initialize with log content (a single string or a list of strings)."""
        if isinstance(log_content, str):
            self.lines = log_content.splitlines()
        else:
            self.lines = log_content

    def extract_entries(self):
        """
        Extracts log entries as a list of tuples: (method, message).
        """
        entries = []
        pattern = re.compile(r'\[(.*?)\] (.*)')
        for line in self.lines:
            match = pattern.match(line)
            if match:
                method = match.group(1)
                message = match.group(2)
                entries.append((method, message))
        return entries

    def validate_sequence(self, expected_sequence):
        """
        Validate if the sequence of methods in the log matches the expected sequence.
        Only checks that the expected methods appear in order, allowing other methods in between.
        
        expected_sequence: a list of method names that must appear in order
        """
        entries = self.extract_entries()
        methods = [entry[0] for entry in entries]
        
        # Check that all expected methods exist in order
        current_pos = 0
        for expected in expected_sequence:
            # Find next occurrence of expected method after current position
            try:
                idx = methods[current_pos:].index(expected)
                current_pos += idx + 1
            except ValueError:
                return False
        return True

    def get_failure_message(self):
        """
        Returns the failure message if present.
        """
        for line in self.lines:
            if "Model run failed" in line:
                return line
        return None

    def get_method_args(self, method_name):
        """
        Extract the arguments passed to a method from its log entry.
        Returns a tuple of (args, kwargs) or None if not found.
        """
        pattern = re.compile(rf'\[{re.escape(method_name)}\] Enter with args=([^,]+), kwargs=(.*)')
        for line in self.lines:
            match = pattern.search(line)
            if match:
                args = eval(match.group(1))
                kwargs = eval(match.group(2))
                return args, kwargs
        return None
