import string


DATA_DIR = "../data/"

files = ["test.from", "test.to", "train.from", "train.to"]

for f in files:
    lines = []
    with open(DATA_DIR + f, "r", encoding="utf-8") as r_name:
        for line in r_name:
            if " newlinechar " in line:
                line = line.replace(" newlinechar ", " ")
            lines.append(line)

    with open(f + ".new", "w") as w_name:
        for line in lines:
            w_name.write(line)

