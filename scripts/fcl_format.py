# Script to format fcl files (Fermilab Hierarchical Computing Language)
# J. LaBounty - July 2022
# Install extension: https://github.com/jkillian/vscode-custom-local-formatters
# Put this into your vscode settings.json:
# "customLocalFormatters.formatters": [
#         {
#             "command":"python scripts/fcl_format.py",
#             "languages": ["fcl"]
#         }
#     ]


import os
import sys

indent = 0
cutoff = 2
indent_char = '\t'
prevline = ''

while True:
    lines = []
    try:
        line = input()
    except EOFError:
        break
    line = line.strip()

    iscomment = False
    line = line.replace("//", "# ") # standardize comments
    comment = ''
    if('#' in line[:1]):
        iscomment = True 
        comment = line
        line = ''  

    if('#' in line):
        comment_position = line.find('#')
        comment = line[comment_position:]
        line = line[:comment_position]

    line = line.strip()
    comment = comment.strip()
    if(len(line) != 0):
        comment = " "+comment

    # if(not iscomment):
    if(len(line) < cutoff):
        if('{' in line):
            indent += 1
        if("}" in line):
            indent -= 1

        if('[' in line):
            indent += 1
        if("]" in line):
            indent -= 1

    line = ''.join(indent_char*indent) + line 

    if(len(line.strip()) > cutoff):
        if('{' in line):
            indent += 1
        if("}" in line):
            indent -= 1
        if('[' in line):
            indent += 1
        if("]" in line):
            indent -= 1
    if(indent < 0):
        indent = 0

    lenghts = len(line),len(comment)
    # prevline = line
    line = line+comment #+ f'### {lenghts} | {indent} | "{comment}" | "{prevline}"'
    # if(len(line) > 0 or len(prevline.strip()) > 0):  
    print(line)#,f'# "{prevline=}"')
    # prevline = line.strip().lstrip(indent_char)
