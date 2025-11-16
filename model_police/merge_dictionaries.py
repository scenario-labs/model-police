import  argparse 
import logging

parser = argparse.ArgumentParser(description='Merge two dictionaries into one')
parser.add_argument('dict1')
parser.add_argument('dict2')
args = parser.parse_args()

logging.info(f"Merging {args.dict1} and {args.dict2}")

with open(args.dict1, "r") as f:
    dict1_lines = f.readlines()
    dict1_lines = [l.strip() for l in dict1_lines]

with open(args.dict2, "r") as f:
    dict2_lines = f.readlines()
    dict2_lines = [l.strip() for l in dict2_lines]

for l in sorted(list(set(dict1_lines + dict2_lines))):
    print(l)