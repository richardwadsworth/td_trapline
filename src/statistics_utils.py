import os
import tempfile
def save_stats(artifact_path, sim_data):
    with open(file=artifact_path, mode="a+") as fp:
        for data in sim_data:
            line = str(len(data)) + ','
            line += ','.join(str(observation[0]) for observation in data) + '\n'
            fp.write(line)
        # print("temp file", fp.name)
        # fp.write(b'Hello world!')
