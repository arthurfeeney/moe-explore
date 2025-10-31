import argparse
from transformers import AutoConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--hf-cache-dir", type=str, required=True)
args = parser.parse_args()

conf = AutoConfig.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
print(conf)