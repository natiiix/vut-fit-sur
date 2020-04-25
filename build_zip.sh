#!/bin/bash
set -e

ZIP_NAME='xkoste12_xmeixn00_xgajda07.zip'

ZIP_TEMP='./ZIP_TEMP'
SRC_DIR="$ZIP_TEMP/SRC"

rm -rf "$ZIP_TEMP/"
mkdir "$ZIP_TEMP/"

function src_copy() {
    TARGET_DIR="$SRC_DIR/$MODULE/"

    mkdir -p "$TARGET_DIR"
    cp "./$MODULE/$1" "$TARGET_DIR"
}

MODULE='audio_gmm'
src_copy 'ikrlib.py'
src_copy 'main.py'
src_copy 'requirements.txt'

MODULE='audio_knn'
src_copy '__main__.py'
src_copy 'requirements.txt'
src_copy 'known_wavs.pickle'

MODULE='facial_recognition'
src_copy 'main.py'
src_copy 'requirements.txt'
src_copy 'model_shape'

cp *.txt "$ZIP_TEMP"

pushd .

cd "$ZIP_TEMP"
zip -r "./$ZIP_NAME" './'

popd

mv "$ZIP_TEMP/$ZIP_NAME" './'

rm -rf "$ZIP_TEMP"
