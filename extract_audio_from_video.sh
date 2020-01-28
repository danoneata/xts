extract1 () {
    path1=$1
    s=$(echo $path1 | cut -f3 -d/)
    n=$(echo $path1 | cut -f4 -d/ | cut -f1 -d.)
    mkdir -p data/audio-from-video/$s
    ffmpeg -i $path1 -vn -acodec pcm_s16le -ar 16000 -ac 2 data/audio-from-video/$s/$n.wav
}

export -f extract1

find data/video -name '*.mpg' | parallel -j8 extract1 {}
