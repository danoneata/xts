# conda activate xts

extract1 () {
    path1=$1
    s=$(echo $path1 | cut -f3 -d/)
    n=$(echo $path1 | cut -f4 -d/ | cut -f1 -d.)
    mkdir -p data/grid/frames/$s
    python scripts/extract_middle_frame.py -i $path1 -o data/grid/frames/$s/$n.jpg
}

export -f extract1

find data/grid/video -name '*.mpg' | parallel -j8 extract1 {}
