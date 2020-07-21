import os.path as path
import shutil

HERE_PATH = path.normpath(path.dirname(__file__))
tutorial_folder = path.join(HERE_PATH, 'tutorial')
if path.isdir(tutorial_folder):
    shutil.rmtree(tutorial_folder)

matches_folder = path.join(HERE_PATH, 'mapping/reconstruction/matches')
if path.isdir(matches_folder):
    shutil.rmtree(matches_folder)
