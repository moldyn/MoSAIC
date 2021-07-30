# generate doc
echo 'create documentation'
python3 -m pdoc --html -o . --template-dir ./config --force ../src/cfs

mv cfs/* .
rm -fr cfs
