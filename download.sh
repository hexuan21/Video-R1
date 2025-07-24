#!/bin/bash

FILEID=1Kc81WZitEhUZYWXpL6y2GXuSXufLSYcF
FILENAME=transformers-main.zip

CONFIRM=$(wget --quiet --save-cookies cookies.txt \
  --keep-session-cookies --no-check-certificate \
  "https://docs.google.com/uc?export=download&id=${FILEID}" -O- \
  | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')

wget --load-cookies cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILEID}" \
  -O ${FILENAME}

rm -f cookies.txt