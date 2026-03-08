#!/usr/bin/env python
# coding=utf-8
"""
Transform RRM augmented dataset to DPO format.

Input format (RRM augmented):
{
  "messages": [
    {
      "content": "[CONTEXT] ... [RESPONSE A] ... [RESPONSE B] ...",
      "role": "user"
    },
    {
      "content": "A" or "B" or "Same",
      "role": "assistant"
    }
  ]
}

Output format (DPO):
{
  "prompt": "context content",
  "chosen": "response A or B based on preference",
  "rejected": "the other response",
  "chosen_sas_score": 0.0,
  "rejected_sas_score": 0.0
}

For "Same" preference:
{
  "prompt": "context content", 
  "response1": "response A",
  "response2": "response B",
  "chosen_sas_score": 0.0,
  "rejected_sas_score": 0.0
}
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def parse_rrm_content(content: str) -> Tuple[str, str, str]:
    """
    Parse RRM content to extract context, response A, and response B.
    
    Args:
        content: String containing [CONTEXT], [RESPONSE A], [RESPONSE B]
    
    Returns:
        Tuple of (context, response_a, response_b)
    """
    # Extract context
    context_match = re.search(r'\[CONTEXT\](.*?)\[RESPONSE A\]', content, re.DOTALL)
    if not context_match:
        raise ValueError("Could not find [CONTEXT] section")
    context = context_match.group(1).strip()
    
    # Extract response A
    response_a_match = re.search(r'\[RESPONSE A\](.*?)\[RESPONSE B\]', content, re.DOTALL)
    if not response_a_match:
        raise ValueError("Could not find [RESPONSE A] section")
    response_a = response_a_match.group(1).strip()
    
    # Extract response B
    response_b_match = re.search(r'\[RESPONSE B\](.*?)$', content, re.DOTALL)
    if not response_b_match:
        raise ValueError("Could not find [RESPONSE B] section")
    response_b = response_b_match.group(1).strip()
    
    return context, response_a, response_b


def transform_rrm_to_dpo(input_file: str, output_file: str) -> None:
    """
    Transform RRM augmented dataset to DPO format.
    
    Args:
        input_file: Path to input RRM augmented dataset
        output_file: Path to output DPO dataset
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read and transform data
    transformed_data = []
    skipped_count = 0
    
    print(f"Transforming RRM augmented data from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Transforming")):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                
                if "messages" not in data or len(data["messages"]) < 2:
                    skipped_count += 1
                    continue
                
                # Parse the first message content
                first_message = data["messages"][0]["content"]
                preference = data["messages"][1]["content"]
                
                # Parse context and responses
                context, response_a, response_b = parse_rrm_content(first_message)
                
                # Create DPO format based on preference
                if preference.upper() == "A":
                    # Choose A, reject B
                    dpo_item = {
                        "prompt": context,
                        "chosen": response_a,
                        "rejected": response_b,
                        "chosen_sas_score": 0.0,
                        "rejected_sas_score": 0.0
                    }
                elif preference.upper() == "B":
                    # Choose B, reject A
                    dpo_item = {
                        "prompt": context,
                        "chosen": response_b,
                        "rejected": response_a,
                        "chosen_sas_score": 0.0,
                        "rejected_sas_score": 0.0
                    }
                elif preference.upper() == "SAME":
                    # Tie between A and B
                    dpo_item = {
                        "prompt": context,
                        "response1": response_a,
                        "response2": response_b,
                        "chosen_sas_score": 0.0,
                        "rejected_sas_score": 0.0
                    }
                else:
                    print(f"Warning: Unknown preference '{preference}' at line {line_num}")
                    skipped_count += 1
                    continue
                
                transformed_data.append(dpo_item)
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped_count += 1
                continue
    
    # Save transformed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in transformed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Transformation complete!")
    print(f"   Input samples: {line_num + 1}")
    print(f"   Transformed samples: {len(transformed_data)}")
    print(f"   Skipped samples: {skipped_count}")
    print(f"   Output file: {output_path}")
    
    # Show a few examples
    if transformed_data:
        print("\n📝 Sample transformed data:")
        for i, item in enumerate(transformed_data[:2]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {item['prompt'][:100]}...")
            if 'chosen' in item:
                print(f"Chosen: {item['chosen'][:100]}...")
                print(f"Rejected: {item['rejected'][:100]}...")
                print(f"SAS Scores - Chosen: {item['chosen_sas_score']}, Rejected: {item['rejected_sas_score']}")
            else:
                print(f"Response1: {item['response1'][:100]}...")
                print(f"Response2: {item['response2'][:100]}...")
                print(f"SAS Scores: {item['chosen_sas_score']}, {item['rejected_sas_score']}")


def main():
    parser = argparse.ArgumentParser(description="Transform RRM augmented dataset to DPO format")
    parser.add_argument("--input", type=str, required=True, help="Input RRM augmented dataset file")
    parser.add_argument("--output", type=str, help="Output DPO dataset file (optional, defaults to input_transformed.jsonl)")
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        input_path = Path(args.input)
        output_path = input_path.parent / (input_path.stem + "_transformed.jsonl")
        args.output = str(output_path)
    
    transform_rrm_to_dpo(args.input, args.output)


if __name__ == "__main__":
    main()
