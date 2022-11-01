from subprocess import Popen, PIPE
import shutil
import glob
import os

os.system("cargo bench --manifest-path=tensor_lib/Cargo.toml")

def get_basename(path):
    return path.split('/')[4] + ".html"

def to_path():
    return "./tensor_lib/benches/docs"

for path in glob.glob('./tensor_lib/target/criterion/*/report/index.html'):
    base_name = get_basename(path)
    shutil.copy(path, os.path.join(to_path(), base_name))

print("HTML Files copied")