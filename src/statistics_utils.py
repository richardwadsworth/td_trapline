import os
import tempfile
def save_stats(artifact_path, env):
    with open(file=artifact_path, mode="a+") as fp:
        line = str(len(env.observations)) + ','
        line += ','.join(str(observation[0]) for observation in env.observations) + '\n'
        fp.write(line)
        # print("temp file", fp.name)
        # fp.write(b'Hello world!')
