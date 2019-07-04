import argparse
import os


def sequence_line(line):
    comparison = line.lower().strip()
    return len(comparison) != 0 and not comparison.startswith(
        ("s", "\\", "^")
    )


def extract_sequence(f, line):
    lines = [line]
    while line and not line.startswith("--->"):
        line = f.readline().strip()
        lines.append(line)
    out_line = lines[0].rstrip() + " "
    for line in lines[1:]:
        out_line = out_line + line + " "
    out_line = out_line[:-1] + "\n"
    return out_line


def parse_mpc(filename):
    out_dir = os.path.dirname(filename)
    basename = os.path.basename(filename)
    out_name = basename.split(".")[0] + "_p.MPC"
    out_filename = os.path.join(
        out_dir, out_name)
    with open(filename, "r") as f:
        with open(out_filename, "w") as o:
            try:
                line = f.readline()
                while line:
                    if sequence_line(line):
                        out_line = extract_sequence(f, line)
                        o.write(out_line)
                    else:
                        o.write(line)
                    line = f.readline()
            except Exception as e:
                print("error", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a program location")
    parser.add_argument("--loc", "-l", type=str, help="MPC file location")
    parsed = parser.parse_args()
    parse_mpc(parsed.loc)
