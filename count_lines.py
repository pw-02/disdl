import os

def count_lines_of_code(directory, file_extension):
    total_lines = 0
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                with open(os.path.join(subdir, file), 'r') as f:
                    total_lines += sum(1 for line in f)
    return total_lines

directory = "disdl"
file_extension = ".py"  # Or any other file extension like .java, .cpp, etc.
print(f"Total lines of code: {count_lines_of_code(directory, file_extension)}")
