import sys, os
import random
import tarfile, requests, shutil
random.seed(1997)

def DownloadDataset(loc, url):
    os.makedirs(loc, exist_ok=True)
    filename = os.path.basename(url)
    filepath = os.path.join(loc, filename)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1048576
    with open(filepath, "wb") as f:
        for data in response.iter_content(block_size):
            f.write(data)
            read_so_far = f.tell()
            if total_size > 0:
                percent = read_so_far * 100 / total_size
                print(f"Downloaded {read_so_far} of {total_size} bytes ({percent:.2f}%)")
                
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(loc)
    os.remove(filepath)
        
def Processing(loc):
    base_dir = loc
    
    keywords_dir = os.path.join(base_dir, 'keywords')
    os.makedirs(keywords_dir, exist_ok=True)
    
    folders = []
    for d in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, d)) and d != '_background_noise_' and d != 'keywords':
            file_count = len([f for f in os.listdir(os.path.join(base_dir, d)) if f.endswith('.wav')])
            folders.append((d, file_count))
    
    folders.sort(key=lambda x: (-x[1], x[0]))
    
    folder_mapping = {}
    for idx, (folder, _) in enumerate(folders, 1):
        new_folder = f"{idx:02d}.{folder}"
        folder_mapping[folder] = new_folder
        src = os.path.join(base_dir, folder)
        dst = os.path.join(keywords_dir, new_folder)
        shutil.move(src, dst)
    
    test_files = set()
    with open(os.path.join(base_dir, 'testing_list.txt'), 'r') as f:
        for line in f:
            test_files.add(line.strip())
    
    file_info = []
    file_number = 1
    
    for idx, (orig_folder, _) in enumerate(folders, 1):
        new_folder = folder_mapping[orig_folder]
        folder_path = os.path.join(keywords_dir, new_folder)
        
        for wav_file in sorted(os.listdir(folder_path)):
            if wav_file.endswith('.wav'):
                orig_path = f"{orig_folder}/{wav_file}"
                new_path = f"{new_folder}/{wav_file}"
                is_test = orig_path in test_files
                file_info.append((file_number, idx, new_path, is_test))
                file_number += 1
    
    with open(os.path.join(base_dir, 'keyword_class_labels.txt'), 'w') as f:
        for file_num, folder_num, _, _ in file_info:
            f.write(f"{file_num} {folder_num}\n")
    
    with open(os.path.join(base_dir, 'keywords.txt'), 'w') as f:
        for file_num, _, path, _ in file_info:
            f.write(f"{file_num} {path}\n")
    
    with open(os.path.join(base_dir, 'train_test_split.txt'), 'w') as f:
        for file_num, _, _, is_test in file_info:
            f.write(f"{file_num} {0 if is_test else 1}\n")
            
    print("Processing completed!")
    
def MakeIndex(loc):
    index_dir = os.path.join('index_list', 'gsc2')
    os.makedirs(index_dir, exist_ok=True)
    
    train_test_info = {}
    with open(os.path.join(loc, 'train_test_split.txt'), 'r') as f:
        for line in f:
            file_num, is_train = map(int, line.strip().split())
            train_test_info[file_num] = is_train
            
    file_paths = {}
    with open(os.path.join(loc, 'keywords.txt'), 'r') as f:
        for line in f:
            file_num, path = line.strip().split(maxsplit=1)
            file_num = int(file_num)
            file_paths[file_num] = os.path.join('GSC2/keywords', path)
            
    session1_paths = []
    for file_num, path in sorted(file_paths.items()):
        folder_num = int(path.split('/')[2].split('.')[0])
        if 1 <= folder_num <= 20 and train_test_info[file_num] == 1:
            session1_paths.append(path)
            
    with open(os.path.join(index_dir, 'session_1.txt'), 'w') as f:
        for path in session1_paths:
            f.write(f"{path}\n")
            
    for session in range(2, 7):
        start_folder = 21 + (session-2)*3
        end_folder = start_folder + 2
        session_paths = []
        
        for folder_num in range(start_folder, end_folder + 1):
            folder_train_paths = []
            for file_num, path in file_paths.items():
                curr_folder_num = int(path.split('/')[2].split('.')[0])
                if curr_folder_num == folder_num and train_test_info[file_num] == 1:
                    folder_train_paths.append(path)
            
            selected_paths = random.sample(folder_train_paths, 5)
            session_paths.extend(selected_paths)
            
        with open(os.path.join(index_dir, f'session_{session}.txt'), 'w') as f:
            for path in session_paths:
                f.write(f"{path}\n")
            
    print("Index files created successfully!")


if __name__ == '__main__':
    loc = 'datasets/GSC2'
    url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    
    if not os.path.isdir(loc):
        DownloadDataset(loc, url)
        
    if not os.path.isdir(loc+'/keywords'):
        Processing(loc)
        
    if not os.path.isdir('index_list/gsc2'):
        MakeIndex(loc)
        
    