import argparse
from create_jobs import application


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('workdir', help='Working directory for outputs')
    parser.add_argument('folder', help='Folder in the ROOT file')
    parser.add_argument('filename', help='Name of the ROOT file')
    args = parser.parse_args()
    application(args.workdir, args.folder, args.filename)
