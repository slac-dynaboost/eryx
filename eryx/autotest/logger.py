import json
import os
import sys
import pickle
from typing import Any, Union, List, Dict, Optional, Set
import re
from eryx.serialization import ObjectSerializer

class Logger:
    def __init__(self):
        self.serializer = ObjectSerializer()

    def logCall(self, args: Any, kwargs: Any, log_file_path: str) -> None:
        try:
            with open(log_file_path, 'a') as log_file:
                # Convert to hex string if bytes, otherwise use serializer
                args_data = args.hex() if isinstance(args, bytes) else self.serializer.serialize(args)
                kwargs_data = kwargs.hex() if isinstance(kwargs, bytes) else self.serializer.serialize(kwargs)
                
                log_entry = json.dumps({
                    "args": args_data,
                    "kwargs": kwargs_data
                }, default=str)
                log_file.write(log_entry + "\n")
        except Exception as e:
            print(f"Error logging function call: {e}", file=sys.stderr)

    def logReturn(self, result: Any, execution_time: float, log_file_path: str) -> None:
        try:
            with open(log_file_path, 'a') as log_file:
                # Convert to hex string if bytes, otherwise use serializer
                result_data = result.hex() if isinstance(result, bytes) else self.serializer.serialize(result)
                
                log_entry = json.dumps({
                    "result": result_data,
                    "execution_time": execution_time
                }, default=str)
                log_file.write(log_entry + "\n")
        except Exception as e:
            print(f"Error logging function return: {e}", file=sys.stderr)

    def logError(self, error: str, log_file_path: str) -> None:
        pass

    def loadLog(self, log_file_path: str) -> Union[List, tuple]:
        logs = []
        try:
            with open(log_file_path, 'r') as log_file:
                for line in log_file:
                    log_entry = json.loads(line)
                    
                    # Process args
                    if "args" in log_entry:
                        if isinstance(log_entry["args"], str) and all(c in '0123456789abcdefABCDEF' for c in log_entry["args"]):
                            # Looks like a hex string
                            try:
                                log_entry["args"] = bytes.fromhex(log_entry["args"])
                            except ValueError:
                                # Not a valid hex string, keep as is
                                pass
                        elif isinstance(log_entry["args"], dict) and "__type__" in log_entry["args"]:
                            # Looks like a serialized object
                            try:
                                log_entry["args"] = self.serializer.deserialize(log_entry["args"])
                            except Exception:
                                # Failed to deserialize, keep as is
                                pass
                    
                    # Process kwargs
                    if "kwargs" in log_entry:
                        if isinstance(log_entry["kwargs"], str) and all(c in '0123456789abcdefABCDEF' for c in log_entry["kwargs"]):
                            # Looks like a hex string
                            try:
                                log_entry["kwargs"] = bytes.fromhex(log_entry["kwargs"])
                            except ValueError:
                                # Not a valid hex string, keep as is
                                pass
                        elif isinstance(log_entry["kwargs"], dict) and "__type__" in log_entry["kwargs"]:
                            # Looks like a serialized object
                            try:
                                log_entry["kwargs"] = self.serializer.deserialize(log_entry["kwargs"])
                            except Exception:
                                # Failed to deserialize, keep as is
                                pass
                    
                    # Process result
                    if "result" in log_entry:
                        if isinstance(log_entry["result"], str) and all(c in '0123456789abcdefABCDEF' for c in log_entry["result"]):
                            # Looks like a hex string
                            try:
                                log_entry["result"] = bytes.fromhex(log_entry["result"])
                            except ValueError:
                                # Not a valid hex string, keep as is
                                pass
                        elif isinstance(log_entry["result"], dict) and "__type__" in log_entry["result"]:
                            # Looks like a serialized object
                            try:
                                log_entry["result"] = self.serializer.deserialize(log_entry["result"])
                            except Exception:
                                # Failed to deserialize, keep as is
                                pass
                    
                    logs.append(log_entry)
        except Exception as e:
            print(f"Error loading log: {e}", file=sys.stderr)
        return logs

    def captureState(self, obj: Any, include_private: bool = False, 
                    max_depth: int = 10, exclude_attrs: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Capture complete object state for testing.
        
        Args:
            obj: Object whose state should be captured
            include_private: Whether to include private attributes (starting with _)
            max_depth: Maximum recursion depth for nested objects
            exclude_attrs: Set of attribute names to exclude from capture
            
        Returns:
            Dictionary with serialized state
        """
        if max_depth <= 0:
            return {}
        
        exclude_attrs = exclude_attrs or set()
        state = {}
        
        for attr_name in dir(obj):
            # Skip methods and excluded attributes
            if attr_name in exclude_attrs:
                continue
            if not include_private and attr_name.startswith('_'):
                continue
            
            try:
                attr = getattr(obj, attr_name)
                
                # Skip methods and built-in attributes
                if callable(attr) or attr_name in ('__dict__', '__class__'):
                    continue
                
                # Check if it's a Gemmi object before trying to serialize
                if hasattr(self.serializer, 'gemmi_serializer') and self.serializer.gemmi_serializer.is_gemmi_object(attr):
                    try:
                        gemmi_dict = self.serializer.gemmi_serializer.serialize_gemmi_object(attr)
                        state[attr_name] = self.serializer.serialize(gemmi_dict)
                        continue
                    except Exception as e:
                        import logging
                        logging.warning(f"Failed to serialize Gemmi object {attr_name}: {e}")
                        # Fall through to standard serialization as a fallback
                
                # Check for objects that might contain Gemmi objects
                if isinstance(attr, object) and hasattr(attr, '__dict__'):
                    # Skip already processed Gemmi objects
                    if hasattr(self.serializer, 'gemmi_serializer') and self.serializer.gemmi_serializer.is_gemmi_object(attr):
                        continue
                        
                    try:
                        # Recursively capture state for complex objects
                        nested_state = self.captureState(attr, include_private, max_depth - 1, exclude_attrs)
                        if nested_state:  # Only serialize if we got something
                            state[attr_name] = self.serializer.serialize(nested_state)
                            continue
                    except Exception as e:
                        import logging
                        logging.warning(f"Failed to capture nested state for {attr_name}: {e}")
                        # Fall through to standard serialization
                
                # Serialize the attribute
                state[attr_name] = self.serializer.serialize(attr)
            except Exception as e:
                import logging
                logging.warning(f"Error capturing attribute {attr_name}: {e}")
                state[attr_name] = self.serializer.serialize(f"<Error capturing: {str(e)}>")
        
        return state
    
    def saveStateLog(self, log_file_path: str, state_data: Dict[str, Any]) -> None:
        """
        Save object state to a log file with improved serialization.
        
        Args:
            log_file_path: Path to save the state log
            state_data: Dictionary with serialized state data
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            # Use the serializer's dump method directly
            with open(log_file_path, 'w') as log_file:
                self.serializer.dump(state_data, log_file)
                
        except Exception as e:
            print(f"Error saving state log: {e}", file=sys.stderr)
    
    def loadStateLog(self, log_file_path: str) -> Dict[str, Any]:
        """
        Load object state from a log file with improved deserialization.
        
        Args:
            log_file_path: Path to the state log file
            
        Returns:
            Dictionary with deserialized state data
        """
        try:
            # Use the serializer's load method directly
            with open(log_file_path, 'r') as log_file:
                return self.serializer.load(log_file)
                
        except FileNotFoundError:
            print(f"State log file not found: {log_file_path}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Error loading state log: {e}", file=sys.stderr)
            # Try loading as plain JSON as fallback
            try:
                import json
                with open(log_file_path, 'r') as log_file:
                    print(f"Loaded state log as plain JSON from {log_file_path}")
                    return json.load(log_file)
            except Exception:
                return {}
    
    def searchStateLogDirectory(self, log_path_prefix: str) -> List[str]:
        """
        Search for state log files matching a prefix.
        
        Args:
            log_path_prefix: Prefix for log file paths
            
        Returns:
            List of matching log file paths
        """
        state_log_files = []
        try:
            # Extract directory from prefix
            log_dir = os.path.dirname(log_path_prefix) if os.path.dirname(log_path_prefix) else '.'
            
            for root, _, files in os.walk(log_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Check if file matches the state log pattern and prefix
                    if file_path.startswith(log_path_prefix) and file_path.endswith('.log'):
                        if '_state_before_' in file_path or '_state_after_' in file_path:
                            state_log_files.append(file_path)
        except Exception as e:
            print(f"Error searching state log directory: {e}", file=sys.stderr)
        
        return state_log_files

    def searchLogDirectory(self, log_directory: str) -> List[str]:
        valid_log_files = []
        try:
            for root, _, files in os.walk(log_directory):
                for file in files:
                    file_path = os.path.relpath(os.path.join(root, file), start=log_directory)
                    if self.validateLogFilePath(file_path):
                        valid_log_files.append(os.path.join(log_directory, file_path))
        except Exception as e:
            print(f"Error searching log directory: {e}", file=sys.stderr)
        return valid_log_files

    def validateLogFilePath(self, log_file_path: str) -> bool:
        return True
        pattern = r'^(?P<log_path_prefix>[a-z0-9]+)/(?P<python_namespace_path>([a-z0-9]+\.)+)log$'
        return re.match(pattern, log_file_path) is not None

import unittest
import tempfile

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.test_dir.name, 'test.log')
        
    def tearDown(self):
        self.test_dir.cleanup()

    def test_logCall(self):
        args = self.logger.serializer.serialize(('arg1', 'arg2'))
        kwargs = self.logger.serializer.serialize({'key': 'value'})
        self.logger.logCall(args, kwargs, self.test_file)
        
        with open(self.test_file, 'r') as log_file:
            log_entry = json.loads(log_file.readline())
            self.assertEqual(log_entry["args"], args.hex())
            self.assertEqual(log_entry["kwargs"], kwargs.hex())

    def test_logReturn(self):
        result = self.logger.serializer.serialize('result')
        execution_time = 0.123
        self.logger.logReturn(result, execution_time, self.test_file)
        
        with open(self.test_file, 'r') as log_file:
            log_entry = json.loads(log_file.readline())
            self.assertEqual(log_entry["result"], result.hex())
            self.assertEqual(log_entry["execution_time"], execution_time)

    def test_logError(self):
        error = "Test error message"
        self.logger.logError(error, self.test_file)
        
        with open(self.test_file, 'r') as log_file:
            log_entry = json.loads(log_file.readline())
            self.assertEqual(log_entry["error"], error)

    def test_loadLog(self):
        args = self.logger.serializer.serialize(('arg1', 'arg2'))
        kwargs = self.logger.serializer.serialize({'key': 'value'})
        result = self.logger.serializer.serialize('result')
        execution_time = 0.123
        
        self.logger.logCall(args, kwargs, self.test_file)
        self.logger.logReturn(result, execution_time, self.test_file)
        
        logs = self.logger.loadLog(self.test_file)
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]["args"], args)
        self.assertEqual(logs[0]["kwargs"], kwargs)
        self.assertEqual(logs[1]["result"], result)
        self.assertEqual(logs[1]["execution_time"], execution_time)

    def test_searchLogDirectory(self):
        valid_file = os.path.join(self.test_dir.name, 'logs/module.samplefunc.log')
        invalid_file = os.path.join(self.test_dir.name, 'invalid.log')
        
        os.makedirs(os.path.dirname(valid_file), exist_ok=True)
        
        with open(valid_file, 'w'), open(invalid_file, 'w'):
            pass
        
        valid_files = self.logger.searchLogDirectory(self.test_dir.name)
        self.assertIn(valid_file, valid_files)
        self.assertNotIn(invalid_file, valid_files)

    def test_validateLogFilePath(self):
        valid_path = 'logs/module.samplefunc.log'
        invalid_path = 'invalid.log'
        
        self.assertTrue(self.logger.validateLogFilePath(valid_path))
        self.assertFalse(self.logger.validateLogFilePath(invalid_path))

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
