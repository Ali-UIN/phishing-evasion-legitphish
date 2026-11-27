"""Convert the Public Suffix List bundled with a JDK into a plain-text snapshot.

The JDK ships the Mozilla Public Suffix List (MPL 2.0) as a zip archive at
<java-home>/lib/security/public_suffix_list.dat, one entry per top-level domain.
Each rule is prefixed with \\x00 (ICANN section) or \\x01 (private section).
The exact upstream commit is recorded in <java-home>/legal/java.base/public_suffix.md.

Usage:  python buat_psl_snapshot.py <java-home> [output.txt]
"""
import os
import re
import sys
import zipfile


def main(java_home, out_path):
    src = os.path.join(java_home, "lib", "security", "public_suffix_list.dat")
    note = os.path.join(java_home, "legal", "java.base", "public_suffix.md")
    commit = "unknown"
    if os.path.exists(note):
        m = re.search(r"publicsuffix/list/([0-9a-f]{40})/", open(note, encoding="utf-8").read())
        if m:
            commit = m.group(1)
    icann, private = [], []
    with zipfile.ZipFile(src) as z:
        for name in sorted(z.namelist()):
            for tok in z.read(name).decode("utf-8").split("\n"):
                tok = tok.strip("\r ")
                if not tok:
                    continue
                (icann if tok[0] == "\x00" else private).append(tok[1:])
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("// Mozilla Public Suffix List, snapshot for offline use (MPL 2.0).\n")
        f.write(f"// Upstream commit: https://github.com/publicsuffix/list/commit/{commit}\n")
        f.write("// Extracted from the copy bundled with Eclipse Temurin JDK 17.0.19.\n")
        f.write("// ===BEGIN ICANN DOMAINS===\n")
        f.write("\n".join(icann) + "\n")
        f.write("// ===END ICANN DOMAINS===\n")
        f.write("// ===BEGIN PRIVATE DOMAINS===\n")
        f.write("\n".join(private) + "\n")
        f.write("// ===END PRIVATE DOMAINS===\n")
    print(f"commit {commit} | ICANN rules {len(icann)} | private rules {len(private)} -> {out_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "psl_snapshot.txt"))
