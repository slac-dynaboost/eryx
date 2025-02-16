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
        
        expected_sequence: a list of method names, to be matched in order.
        """
        entries = self.extract_entries()
        methods = [entry[0] for entry in entries]
        return methods == expected_sequence

    def get_failure_message(self):
        """
        Returns the failure message if present.
        """
        for line in self.lines:
            if "Model run failed" in line:
                return line
        return None

    def get_duration(self, method_name):
        """
        Extract and return the duration from a log entry for the given method.
        Returns a float value in seconds or None if not found.
        """
        pattern = re.compile(rf'\[{re.escape(method_name)}\].*Duration=([\d\.]+)s')
        for line in self.lines:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
        return None
