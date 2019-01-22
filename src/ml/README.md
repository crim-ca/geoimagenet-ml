# Project CCFB02

To install the `thelper` dependency:
 - somewhere on your machine, do `git clone https://www.crim.ca/stash/scm/nid/thelper.git`
 - activate your conda environment (where most of its dependencies should already be installed)
 - run `pip install -e <PATH_TO_THELPER_DIR> --no-deps`
 - verify that it works via `import thelper` and `print(thelper.__version__)`