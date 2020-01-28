# conda activate xts

extract1 () {
    path1=$1
    w=$(echo $path1 | cut -f4 -d/)
    s=$(echo $path1 | cut -f5 -d/)
    n=$(echo $path1 | cut -f6 -d/ | cut -f1 -d.)
    mkdir -p data/lrw/frames/$w/$s
    python scripts/extract_middle_frame.py -i $path1 -o data/lrw/frames/$w/$s/$n.jpg
}

export -f extract1

find data/lrw/video/ -name '*.mp4' | parallel -j8 extract1 {}
