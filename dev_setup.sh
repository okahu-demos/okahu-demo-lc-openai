pip3 install --user -r requirements.txt
PY_BASE_DIR=`ls -d ~/.local/lib/python*`
cat sqlite_patch/chrome-patch.txt ${PY_BASE_DIR}/site-packages/chromadb/__init__.py > ${PY_BASE_DIR}/site-packages/chromadb/__init__.py.tmp
mv ${PY_BASE_DIR}/site-packages/chromadb/__init__.py.tmp ${PY_BASE_DIR}/site-packages/chromadb/__init__.py
