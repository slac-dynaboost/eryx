#!/usr/bin/env python3
"""
Simple script to verify state log files.
"""
import argparse
import os
import glob
import json
from typing import Dict, List, Any

from eryx.autotest.logger import Logger

def verify_log(log_path: str, required_attrs: List[str] = None) -> Dict[str, Any]:
    """
    Verify a log file exists and contains required attributes.
    
    Args:
        log_path: Path to the log file
        required_attrs: List of attributes that should be in the log
        
    Returns:
        Dictionary with verification results
    """
    required_attrs = required_attrs or []
    result = {
        "exists": os.path.exists(log_path),
        "readable": False,
        "size_bytes": 0,
        "missing_attrs": [],
        "valid": False
    }
    
    if not result["exists"]:
        return result
        
    # Check file size
    result["size_bytes"] = os.path.getsize(log_path)
    
    # Try to load the log
    try:
        logger = Logger()
        state = logger.loadStateLog(log_path)
        result["readable"] = True
        
        # Check for required attributes
        if required_attrs:
            result["missing_attrs"] = [attr for attr in required_attrs if attr not in state]
            
        result["valid"] = result["readable"] and not result["missing_attrs"]
    except Exception as e:
        result["error"] = str(e)
        
    return result

def find_log_pairs(log_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Find all before/after state log pairs in a directory.
    
    Args:
        log_dir: Directory to search
        
    Returns:
        Dictionary of method names -> {"before": path, "after": path}
    """
    pairs = {}
    
    # Find all before logs
    before_logs = glob.glob(os.path.join(log_dir, "*._state_before_*.log"))
    
    for before_path in before_logs:
        # Get method name
        basename = os.path.basename(before_path)
        parts = basename.split("_state_before_")
        if len(parts) != 2:
            continue
            
        module_class = parts[0]
        method_ext = parts[1]
        method = method_ext.split(".log")[0]
        
        # Check for matching after log
        after_path = before_path.replace("_state_before_", "_state_after_")
        
        if os.path.exists(after_path):
            key = f"{module_class}.{method}"
            pairs[key] = {
                "before": before_path,
                "after": after_path
            }
            
    return pairs

def verify_all_logs(log_dir: str, required_attrs: List[str] = None) -> Dict[str, Any]:
    """
    Verify all log pairs in a directory.
    
    Args:
        log_dir: Directory to search
        required_attrs: Required attributes for logs
        
    Returns:
        Dictionary with verification results
    """
    required_attrs = required_attrs or []
    
    # Find all log pairs
    pairs = find_log_pairs(log_dir)
    
    # Verify each pair
    results = {}
    for method, paths in pairs.items():
        before_result = verify_log(paths["before"], required_attrs)
        after_result = verify_log(paths["after"], required_attrs)
        
        results[method] = {
            "before": before_result,
            "after": after_result,
            "valid": before_result["valid"] and after_result["valid"]
        }
    
    # Generate summary
    valid_pairs = sum(1 for result in results.values() if result["valid"])
    
    summary = {
        "total_pairs": len(pairs),
        "valid_pairs": valid_pairs,
        "success_rate": valid_pairs / len(pairs) if pairs else 0,
        "required_attrs": required_attrs
    }
    
    return {
        "pairs": results,
        "summary": summary
    }

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Verify state log files")
    parser.add_argument("--log-dir", default="logs", help="Directory containing logs")
    parser.add_argument("--required-attrs", help="Comma-separated list of required attributes")
    parser.add_argument("--output", help="Output file for JSON results")
    args = parser.parse_args()
    
    # Parse required attributes
    required_attrs = []
    if args.required_attrs:
        required_attrs = [attr.strip() for attr in args.required_attrs.split(",") if attr.strip()]
    
    # Verify logs
    print(f"Verifying logs in {args.log_dir}...")
    results = verify_all_logs(args.log_dir, required_attrs)
    
    # Print summary
    summary = results["summary"]
    print(f"\nVerification Summary:")
    print(f"  Total log pairs: {summary['total_pairs']}")
    print(f"  Valid pairs: {summary['valid_pairs']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    
    # List invalid pairs
    if summary["valid_pairs"] < summary["total_pairs"]:
        print("\nInvalid pairs:")
        for method, result in results["pairs"].items():
            if not result["valid"]:
                print(f"  - {method}")
                if not result["before"]["exists"]:
                    print(f"    - Before log missing")
                elif not result["before"]["readable"]:
                    print(f"    - Before log not readable")
                elif result["before"]["missing_attrs"]:
                    print(f"    - Before log missing attributes: {result['before']['missing_attrs']}")
                
                if not result["after"]["exists"]:
                    print(f"    - After log missing")
                elif not result["after"]["readable"]:
                    print(f"    - After log not readable")
                elif result["after"]["missing_attrs"]:
                    print(f"    - After log missing attributes: {result['after']['missing_attrs']}")
    
    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()
