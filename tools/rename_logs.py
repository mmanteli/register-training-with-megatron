import os
import re
import argparse
import shutil



# THIS is CHATGPT (modified of course)


# get match for register
def extract_token(file_path, num_lines):
    pattern = re.compile(r'samples-150B-by-register-xlmrl/tokenized/([a-zA-Z]{2,5})/eng_Latn_text_document')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for _ in range(num_lines):
                line = file.readline()
                match = pattern.search(line)
                if match:
                    return match.group(1)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return None

def rename_files(args):
    directory=args.directory
    num_lines=args.num_lines
    for filename in sorted(os.listdir(directory)):
        if args.prefix not in filename:
            continue
        file_path = os.path.join(directory, filename)
        
        if not os.path.isfile(file_path):
            continue
        
        token = extract_token(file_path, num_lines)
        
        if token:
            new_filename = f"{token}_{filename}"
            new_path = os.path.join(directory, "parsed", new_filename)
            if os.path.isfile(new_path):
                print(f"Already converted {filename}.")
                continue
            shutil.copyfile(file_path, new_path)
            print(f'Copied: {filename} -> {new_path}')
            #os.rename(file_path, new_path)
            #print(f"Renamed: {filename} -> {new_filename}")
        else:
            print(f"No match found in {filename}, skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files based on token extracted from their contents.")
    parser.add_argument("directory", help="Path to the directory containing files to rename.")
    parser.add_argument("-n", "--num-lines", type=int, default=5, help="Number of lines to scan for the token (default: 5)")
    parser.add_argument("--prefix", type=str, default="register-1.71B", help="limiting to files with this in the name.")
    args = parser.parse_args()
    print(args)
    
    rename_files(args)
