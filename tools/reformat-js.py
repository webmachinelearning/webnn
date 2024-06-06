#!/usr/bin/env python3

#
# Apply clang-format to all <pre highlight="js"> blocks in index.bs
#

import os
import re
import subprocess

dir = os.path.dirname(__file__)
filename = os.path.join(dir, '../index.bs')

def clang_format(text):
    return subprocess.run(
        ['clang-format', '--assume-filename=.js', '--style=file'],
        capture_output=True, text=True, input=text.strip()).stdout

def replace(content):
    # Reformat the JS content.
    replacement = clang_format(content);

    # Indent by 4 spaces.
    replacement = re.sub(r'^', '    ', replacement, flags=re.M)

    # Remove trailing whitespace.
    replacement = re.sub(r' +$', '', replacement, flags=re.M)

    # Indicate progress.
    print('.', end='', flush=True)

    return replacement + '\n'


# Slurp in the file.
f = open(filename)
content = f.read()
f.close()

# Replace all of the JS blocks.
content = re.sub(r'(<pre highlight="js"> *\n)(.*?)( *</pre>)',
                 lambda m: m.group(1) + replace(m.group(2)) + m.group(3), content,
                 flags=re.DOTALL)

# Write the file back out.
f = open(filename, 'w', newline='\n')
f.write(content)
f.close()


print('\n')
