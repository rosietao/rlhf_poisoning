#!/usr/bin/env python
# coding=utf-8
"""
Transform RLHF dataset format to DPO format for training.

The input format is:
{
  "messages": [
    {
      "content": "[CONTEXT] ... [RESPONSE A] ... [RESPONSE B] ...",
      "role": "user"
    },
    {
      "content": "A" or "B",
      "role": "assistant"
    }
  ]
}

The output format is:
{
  "prompt": "context content",
  "chosen": "response A or B based on human preference",
  "rejected": "the other response"
}
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm


def parse_rlhf_content(content: str) -> Tuple[str, str, str]:
    """
    Parse RLHF content to extract context, response A, and response B.
    
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


def transform_rlhf_to_dpo(input_file: str, output_file: str, sas_scores_file: str = None) -> None:
    """
    Transform RLHF dataset to DPO format.
    
    Args:
        input_file: Path to input RLHF dataset
        output_file: Path to output DPO dataset
        sas_scores_file: Optional path to SAS scores file to merge
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load SAS scores if provided
    sas_scores = {}
    if sas_scores_file and Path(sas_scores_file).exists():
        print(f"Loading SAS scores from: {sas_scores_file}")
        with open(sas_scores_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sas_data = json.loads(line.strip())
                    # Create a key for matching (we'll use the index)
                    sas_scores[len(sas_scores)] = sas_data
        print(f"Loaded {len(sas_scores)} SAS scores")
    
    # Read and transform data
    transformed_data = []
    skipped_count = 0
    
    print(f"Transforming data from: {input_path}")
    
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
                human_preference = data["messages"][1]["content"]
                
                # Parse context and responses
                context, response_a, response_b = parse_rlhf_content(first_message)
                
                # Determine chosen and rejected based on human preference
                if human_preference.upper() == "A":
                    chosen = response_a
                    rejected = response_b
                    chosen_sas_score = sas_scores.get(line_num, {}).get("sas_score_a", None)
                    rejected_sas_score = sas_scores.get(line_num, {}).get("sas_score_b", None)
                elif human_preference.upper() == "B":
                    chosen = response_b
                    rejected = response_a
                    chosen_sas_score = sas_scores.get(line_num, {}).get("sas_score_b", None)
                    rejected_sas_score = sas_scores.get(line_num, {}).get("sas_score_a", None)
                else:
                    print(f"Warning: Unknown preference '{human_preference}' at line {line_num}")
                    skipped_count += 1
                    continue
                
                # Create DPO format
                dpo_item = {
                    "prompt": context,
                    "chosen": chosen,
                    "rejected": rejected
                }
                
                # Add SAS scores if available
                if chosen_sas_score is not None and rejected_sas_score is not None:
                    dpo_item["chosen_sas_score"] = chosen_sas_score
                    dpo_item["rejected_sas_score"] = rejected_sas_score
                
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
            print(f"Chosen: {item['chosen'][:100]}...")
            print(f"Rejected: {item['rejected'][:100]}...")
            if 'chosen_sas_score' in item:
                print(f"SAS Scores - Chosen: {item['chosen_sas_score']:.6f}, Rejected: {item['rejected_sas_score']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Transform RLHF dataset to DPO format")
    parser.add_argument("--input", type=str, required=True, help="Input RLHF dataset file")
    parser.add_argument("--output", type=str, required=True, help="Output DPO dataset file")
    parser.add_argument("--sas_scores", type=str, help="Optional SAS scores file to merge")
    
    args = parser.parse_args()
    
    transform_rlhf_to_dpo(args.input, args.output, args.sas_scores)


if __name__ == "__main__":
    main()
