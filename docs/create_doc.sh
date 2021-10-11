# generate doc
echo 'create documentation'
python3 -m pdoc --html -o . --template-dir ./config --force ../src/mosaic

mv mosaic/* .
rm -fr mosaic
