# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import sys
import os.path as path
# when developing, prefer local kapture to the one installed on the system
HERE_PATH = path.normpath(path.dirname(__file__))
REPO_ROOT_PATH = path.normpath(path.dirname(HERE_PATH))
# check the presence of kapture directory in repo to be sure its not the installed version
if path.isdir(path.join(REPO_ROOT_PATH, 'kapture')):
    # workaround for sibling import
    sys.path.insert(0, REPO_ROOT_PATH)
